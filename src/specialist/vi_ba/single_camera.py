"""One camera's lenses and IMU prepared for the joint graph."""

import ctypes
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import gtsam
import numpy as np
import numpy.typing as npt
from specialist_dataset import schema
from specialist_dataset.frames import LensFrames

from specialist.record.session import (
  SessionRecording,
  covered_samples,
  fps_from_times,
  nearest_samples,
)
from specialist.vi_ba.build import LensChannel
from specialist.vi_ba.configs import CameraArgs
from specialist.vi_ba.features import Frame, available_ram, detect_orb
from specialist.vi_ba.finger_kinematics import FingerReproj, recover_gripper_tags
from specialist.vi_ba.geometry import average_pose, consensus_pose
from specialist.vi_ba.imu import (
  GRAVITY,
  GRAVITY_WORLD,
  ImuSamples,
  find_stationary_windows,
  make_preintegration_params,
  preintegrate_between_keyframes,
)
from specialist.vi_ba.init import (
  apply_alignment_to_poses,
  body_poses_from_lens_poses,
  fill_lens_poses,
  visual_inertial_align,
)
from specialist.vi_ba.tag_detection import (
  TagDetection,
  detect_all_tags,
  detect_tags_following_split,
  lens_T_tag_per_kf,
)
from specialist.vi_ba.tag_discovery import (
  GripperTags,
  TagDiscoveryError,
  discover_gripper_tags,
  discovery_samples,
)
from specialist.vi_ba.tracks import (
  add_revisit_landmarks,
  cap_landmarks,
  filter_tracks_by_parallax,
  revisit_tracks,
  run_lens_frontend,
  split_tracks_to_window,
  subsample_obs,
  triangulate_lens,
)

Float64Array = npt.NDArray[np.float64]

# Keyframes decoded at once from the front and the rear lens.
_PREP_BAND = 128
_REAR_BAND = 64

# Share of the available memory the concurrent decodes may claim between them.
_PREP_MEMORY_SHARE = 0.35

# Frames that may wait on the per-frame job workers, and the workers.
_FRAME_JOBS_PENDING = 32
_FRAME_JOB_WORKERS = 8
# Runs the rear lens walk splits into, each with its own decoder.
_REAR_WALK_SEGMENTS = 2
# Share of the keyframes, at least eight, an anchor-pose seed must register.
_SEED_SHARE = 0.25


__all__ = [
  "PreparedCamera",
  "cameras_that_fit",
  "discover_camera_tags",
  "prepare_camera",
  "renumber_landmarks",
  "bound_malloc_arenas",
  "stage_timer",
]


def _resident() -> int | None:
  """The process's anonymous resident memory in bytes, where the platform reports it."""
  try:
    for line in Path("/proc/self/status").read_text().splitlines():
      if line.startswith("RssAnon:"):
        return int(line.split()[1]) * 1024
  except OSError, ValueError, IndexError:
    pass
  return None


def stage_timer(scope: str | None = None) -> Callable[[str], None]:
  """A ``[scope/timing] <stage> +<seconds> rss=<GB>`` line per finished stage."""
  tag = f"{scope}/timing" if scope else "timing"
  last = time.perf_counter()

  def mark(stage: str) -> None:
    nonlocal last
    now = time.perf_counter()
    resident = _resident()
    memory = "" if resident is None else f" rss={resident / 2**30:.1f}GB"
    print(f"  [{tag}] {stage} +{now - last:.1f}s{memory}")
    last = now

  return mark


class _FrameJobs:
  """Tag detection and revisit ORB per frame, on workers beside the walk."""

  def __init__(self, n_frames: int, orb_features: int) -> None:
    self._orb_features = orb_features
    self.tags: list[list[TagDetection]] = [[] for _ in range(n_frames)]
    self._orb: list[Frame | None] = [None] * n_frames
    self._pool = ThreadPoolExecutor(max_workers=_FRAME_JOB_WORKERS)
    self._slots = threading.BoundedSemaphore(_FRAME_JOBS_PENDING)
    self._futures: list[Future[None]] = []

  @property
  def orb(self) -> list[Frame]:
    """Every frame's ORB keypoints and descriptors; empty with no budget."""
    return [f for f in self._orb if f is not None]

  def submit(self, k: int, frame: np.ndarray) -> None:
    # The previous band's freed frames go back to the OS as a band starts.
    if k % _PREP_BAND == 0:
      release_freed_memory()
    self._slots.acquire()
    self._futures.append(self._pool.submit(self._run, k, frame))

  def _run(self, k: int, frame: np.ndarray) -> None:
    try:
      self.tags[k] = detect_all_tags([frame])[0]
      if self._orb_features > 0:
        self._orb[k] = detect_orb(frame, self._orb_features, 20)
    finally:
      self._slots.release()

  def __enter__(self) -> _FrameJobs:
    return self

  def __exit__(self, *exc: object) -> None:
    self._pool.shutdown(wait=True)
    for future in self._futures:
      future.result()


def renumber_landmarks(
  points: dict[int, Float64Array], tracks: dict[int, dict[int, Float64Array]]
) -> tuple[dict[int, Float64Array], dict[int, dict[int, Float64Array]]]:
  """Dense ``0..n-1`` ids for ``points`` in id order; tracks without a point drop."""
  dense = {old: new for new, old in enumerate(sorted(points))}
  return (
    {dense[t]: p for t, p in points.items()},
    {dense[t]: obs for t, obs in tracks.items() if t in dense},
  )


def bound_malloc_arenas(limit: int) -> bool:
  """Cap glibc's malloc arenas at ``limit``; whether the allocator took it."""
  try:
    libc = ctypes.CDLL("libc.so.6")
    m_arena_max = -8
    return bool(libc.mallopt(m_arena_max, limit))
  except OSError, AttributeError:
    return False


def release_freed_memory() -> None:
  """Return freed heap to the OS; a no-op without glibc's ``malloc_trim``."""
  try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
  except OSError, AttributeError:
    pass


@dataclass
class PreparedCamera:
  """Everything one camera contributes to the joint graph + viz."""

  slot: str
  body_poses: list[gtsam.Pose3]
  velocities: list[Float64Array]
  biases: list[gtsam.imuBias.ConstantBias]
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements]
  front: LensChannel
  rear: LensChannel
  # Front-lens anchor PnP, ``{tag_id: {kf: lens_T_tag}}``.
  anchor_obs_by_tag: dict[int, dict[int, gtsam.Pose3]]
  # Each sighted anchor's pose in this camera's local world.
  anchor_locals_by_tag: dict[int, gtsam.Pose3]
  body_T_base_tag: gtsam.Pose3
  keyframe_times: Float64Array
  # Session-clock frame per keyframe; ``schema.keyframe_index`` maps it to the timeline.
  keyframe_frames: npt.NDArray[np.int64]
  # URDF joint name -> kinematic inputs for the joint-angle reprojection factors.
  finger_reproj: dict[str, FingerReproj]
  # This camera's own tag ids, discovered from its rear lens.
  tags: GripperTags
  # Body-frame, rescaled IMU samples, for IMU-rate propagation between keyframes.
  imu: ImuSamples
  # Per-joint raw ``(times, angles)`` at the finger rate; export pins them to the solve.
  finger_frame_angles: dict[str, tuple[Float64Array, Float64Array]] = field(
    default_factory=dict
  )
  # Joint name -> solved angle about the joint z-axis per keyframe, NaN where unseen.
  joint_angles: dict[str, list[float]] = field(default_factory=dict)

  @property
  def n_keyframes(self) -> int:
    return len(self.body_poses)

  @property
  def lenses(self) -> tuple[LensChannel, LensChannel]:
    """Both lens channels, front first."""
    return (self.front, self.rear)


def _choose_prep_anchor(
  tag_ids: tuple[int, ...],
  obs_by_tag: dict[int, dict[int, gtsam.Pose3]],
) -> int:
  """The id sighted at most keyframes; a tie goes to the earlier ``tag_ids`` entry."""
  return max(tag_ids, key=lambda tid: len(obs_by_tag[tid]))


def _imu_and_initial_gravity(
  imu: ImuSamples,
  raw_factory_bias: gtsam.imuBias.ConstantBias,
  body_R_imu: Float64Array,
  keyframe_times: Float64Array,
  body_poses_prealign: list[gtsam.Pose3],
) -> tuple[ImuSamples, gtsam.imuBias.ConstantBias, Float64Array | None]:
  """IMU scaled to |g| at rest, the body-frame bias, and world gravity or ``None``."""
  bias0 = gtsam.imuBias.ConstantBias(
    body_R_imu @ np.asarray(raw_factory_bias.accelerometer(), dtype=np.float64),
    body_R_imu @ np.asarray(raw_factory_bias.gyroscope(), dtype=np.float64),
  )
  windows = find_stationary_windows(imu, bias0)
  gravity_world_initial: Float64Array | None = None
  if windows:
    first_start, first_end = windows[0]
    stationary_mean = (imu.accel[first_start:first_end] - bias0.accelerometer()).mean(
      axis=0
    )
    stationary_mag = float(np.linalg.norm(stationary_mean))
    if 8.0 < stationary_mag < 11.0 and abs(stationary_mag - GRAVITY) > 0.05:
      accel_scale = GRAVITY / stationary_mag
      imu = ImuSamples(
        t=imu.t, gyro=imu.gyro, accel=(imu.accel * accel_scale).astype(np.float64)
      )
      stationary_mean = stationary_mean * accel_scale
    gravity_body = -stationary_mean / float(np.linalg.norm(stationary_mean)) * GRAVITY
    t_stationary_mid = 0.5 * (imu.t[first_start] + imu.t[first_end - 1])
    kf_for_g = int(np.argmin(np.abs(keyframe_times - t_stationary_mid)))
    gravity_world_initial = (
      body_poses_prealign[kf_for_g].rotation().matrix() @ gravity_body
    )
  return imu, bias0, gravity_world_initial


def _seed_body_poses(
  slot: str,
  anchor_obs_per_kf: dict[int, gtsam.Pose3],
  body_T_front_lens: gtsam.Pose3,
  n_keyframes: int,
) -> list[gtsam.Pose3]:
  """Body poses from the anchor sightings, gaps geodesic-filled."""
  n_seen = len(anchor_obs_per_kf)
  if n_seen < max(8, int(n_keyframes * _SEED_SHARE)):
    raise RuntimeError(
      f"{slot}: only {n_seen}/{n_keyframes} keyframes see the anchor; the "
      "trajectory seed needs a quarter of them and at least eight"
    )
  print(f"  [{slot}/vo] anchor-pose init from {n_seen}/{n_keyframes} anchor keyframes")
  lens_poses = {
    kf: lens_T_tag.inverse() for kf, lens_T_tag in anchor_obs_per_kf.items()
  }
  return body_poses_from_lens_poses(
    fill_lens_poses(lens_poses, n_keyframes), body_T_front_lens
  )


def _align_visual_inertial(
  slot: str,
  imu: ImuSamples,
  keyframe_times: Float64Array,
  bias0: gtsam.imuBias.ConstantBias,
  body_poses_prealign: list[gtsam.Pose3],
  gravity_world_initial: Float64Array | None,
) -> tuple[
  list[gtsam.Pose3],
  list[Float64Array],
  list[gtsam.PreintegratedCombinedMeasurements],
]:
  """Gravity-aligned metric body poses, velocities and keyframe preintegration."""

  # Gravity never enters the deltas, so one PIM serves alignment and export.
  pim = preintegrate_between_keyframes(
    make_preintegration_params(GRAVITY_WORLD), imu, keyframe_times, bias0
  )
  alignment = visual_inertial_align(
    body_poses_prealign, pim, fixed_gravity=gravity_world_initial
  )
  # Gravity takes the stationary IMU reading exactly; the weighted pin holds it only near.
  if gravity_world_initial is not None:
    alignment = replace(alignment, gravity_world=gravity_world_initial)
  print(f"  [{slot}/vi] scale={alignment.scale:.4f}")
  body_poses, world_R_prealign = apply_alignment_to_poses(
    body_poses_prealign, alignment
  )
  velocities = [world_R_prealign.matrix() @ v for v in alignment.velocities]
  return body_poses, velocities, pim


def cameras_that_fit(session: SessionRecording, slots: Sequence[str]) -> int:
  """How many cameras may prep at once within the memory on hand."""
  if len(slots) <= 1:
    return len(slots)
  height, width = session.decoded_shape(slots[0])
  bands = 2 * _PREP_BAND + 2 * _REAR_WALK_SEGMENTS * _REAR_BAND
  per_camera = bands * width * height
  budget = available_ram() * _PREP_MEMORY_SHARE
  affordable = int(budget / per_camera)
  return max(1, min(len(slots), affordable))


def keyframe_grid(
  times: Float64Array,
  covered: npt.NDArray[np.int64],
  fps: float,
) -> tuple[npt.NDArray[np.int64], int]:
  """The covered samples on the session-clock stride lattice, and their phase."""
  stride = schema.keyframe_stride(fps)
  session_frames = np.round(times * fps).astype(np.int64)
  on_lattice = np.flatnonzero(session_frames[covered] % stride == 0)
  phase = int(on_lattice[0]) if len(on_lattice) else 0
  return covered[phase::stride], phase


def finger_stride(keyframe_stride: int, fps: float) -> int:
  """The largest divisor of ``keyframe_stride`` up to ``fps / 30``."""
  return max(
    (
      d
      for d in range(1, keyframe_stride + 1)
      if keyframe_stride % d == 0 and d <= fps / 30.0
    ),
    default=1,
  )


def discover_camera_tags(args: CameraArgs) -> GripperTags:
  """Identify the camera's gripper tags from a sparse rear-lens read before prep."""
  rear = args.session.lens_video(args.slot, "rear")
  walk = LensFrames(rear, discovery_samples(rear.sync_samples), band=1)
  detections = detect_all_tags(walk)
  _, rear_lens = args.session.lens_setups(args.slot, int(walk[0].shape[1]))
  try:
    return discover_gripper_tags(
      detections,
      rear_lens.params,
      args.joints,
      tag_size=args.gripper.tag_size,
      anchor_tag_ids=set(args.anchor.tag_ids),
    )
  except TagDiscoveryError as exc:
    raise TagDiscoveryError(f"{args.slot}: {exc}") from exc


def prepare_camera(args: CameraArgs, tags: GripperTags) -> PreparedCamera:
  """Every per-camera stage; body poses land in the camera's gravity-aligned world."""
  slot = args.slot
  lap = stage_timer(slot)
  session = args.session
  front_video = session.lens_video(args.slot, "front")
  rear_video = session.lens_video(args.slot, "rear")
  fps = fps_from_times(front_video.times)

  keyframe_stride = schema.keyframe_stride(fps)
  raw_imu = session.imu(args.slot)
  covered = covered_samples(front_video.times, rear_video.times, raw_imu.t, fps)
  front_samples, grid_phase = keyframe_grid(front_video.times, covered, fps)
  grid_times = front_video.times[front_samples]

  # Rear: a dense walk beside the front walk; the keyframes are a stride of it.
  stride = finger_stride(keyframe_stride, fps)
  dense_samples = nearest_samples(
    rear_video.times, front_video.times[covered][grid_phase::stride]
  )
  kf_stride = keyframe_stride // stride
  rear_frame_times = rear_video.times[dense_samples]

  def rear_frames(begin: int, end: int) -> LensFrames:
    return LensFrames(rear_video, dense_samples[begin:end], band=_REAR_BAND)

  with ThreadPoolExecutor(max_workers=1) as rear_walk:
    rear_tags_future = rear_walk.submit(
      detect_tags_following_split,
      rear_frames,
      len(dense_samples),
      kf_stride,
      {tag_id for tag_id in tags.fingers if tag_id is not None},
      _REAR_WALK_SEGMENTS,
    )
    # Front: one walk serves the flow tracks, the tag detections and the ORB.
    lap("inputs loaded")
    front_frames = LensFrames(front_video, front_samples, band=_PREP_BAND)
    decoded_width = int(front_frames[0].shape[1])
    front_lens, rear_lens = session.lens_setups(args.slot, decoded_width)
    with _FrameJobs(len(front_frames), args.frontend.orb_n_features) as jobs:
      keypoints, flow_tracks = run_lens_frontend(
        front_frames,
        front_lens.params,
        args.frontend,
        label=f"{slot}/front",
        on_frame=jobs.submit,
        band=front_frames.band,
      )
    lap("front lens walk")
  rear_tags_per_frame = rear_tags_future.result()
  front_tags, front_orb = jobs.tags, jobs.orb
  del front_frames, jobs
  rear_tags_per_kf = rear_tags_per_frame[::kf_stride]
  n_keyframes = min(len(front_tags), len(rear_tags_per_kf))
  release_freed_memory()
  lap("rear lens walk, past the front")
  kept = {
    t: o for t, o in flow_tracks.items() if len(o) >= args.frontend.min_track_length
  }
  rejected: dict[int, dict[int, Float64Array]] = {}
  if args.frontend.min_parallax > 0.0:
    kept, rejected = filter_tracks_by_parallax(
      kept, front_lens.params, args.frontend.min_parallax
    )
  del flow_tracks
  lap("front tracks")
  front_tags = front_tags[:n_keyframes]
  keyframe_times = grid_times[:n_keyframes]

  body_R_imu = session.body_R_imu(args.slot)
  # Raw-axis samples rotated into the body frame; gyro is already rad/s.
  imu = ImuSamples(
    t=raw_imu.t,
    gyro=raw_imu.gyro @ body_R_imu.T,
    accel=raw_imu.accel @ body_R_imu.T,
  )
  del raw_imu

  # Front lens only: the rear seat is a solved variable, so prep never reads it.
  obs_by_tag = {
    tag_id: lens_T_tag_per_kf(
      front_tags, tag_id, args.anchor.tag_size, front_lens.params
    )
    for tag_id in args.anchor.tag_ids
  }
  prep_anchor_tag_id = _choose_prep_anchor(args.anchor.tag_ids, obs_by_tag)
  anchor_obs_per_kf = obs_by_tag[prep_anchor_tag_id]
  reference_tag_id = args.anchor.tag_ids[0]
  if prep_anchor_tag_id != reference_tag_id:
    n_ref = len(obs_by_tag[reference_tag_id])
    print(
      f"  [{slot}/anchor] prep reference: tag {prep_anchor_tag_id} "
      f"({len(anchor_obs_per_kf)} sighted keyframes vs {n_ref} for "
      f"tag {reference_tag_id})"
    )
  lap("anchor pnp")

  body_poses_prealign = _seed_body_poses(
    slot, anchor_obs_per_kf, front_lens.body_T_lens, n_keyframes
  )
  lap("trajectory seed")

  imu, bias0, gravity_world_initial = _imu_and_initial_gravity(
    imu,
    session.factory_bias(args.slot),
    body_R_imu,
    keyframe_times,
    body_poses_prealign,
  )

  landmarks_prealign = triangulate_lens(
    kept,
    body_poses_prealign,
    front_lens.body_T_lens,
    front_lens.params,
    args.frontend.min_track_length,
    min_landmark_depth=0.01,
    max_landmark_depth=1000.0,
  )
  lap("prealign triangulation")

  body_poses, velocities, pim = _align_visual_inertial(
    slot,
    imu,
    keyframe_times,
    bias0,
    body_poses_prealign,
    gravity_world_initial,
  )
  lap("vi alignment")

  anchor_locals_by_tag: dict[int, gtsam.Pose3] = {}
  for tag_id, obs in obs_by_tag.items():
    if not obs:
      continue
    candidates = [
      body_poses[kf].compose(front_lens.body_T_lens).compose(lens_T_tag)
      for kf, lens_T_tag in obs.items()
    ]
    agreed = consensus_pose(candidates)
    if agreed is None:
      print(
        f"  [warn] [{slot}/anchor] tag {tag_id} sightings split; seeding on the median"
      )
      agreed = average_pose(candidates)
    anchor_locals_by_tag[tag_id] = agreed

  revisits = revisit_tracks(
    front_orb[:n_keyframes],
    body_poses,
    front_lens,
    max(kept, default=0),
    window=args.frontend.max_track_span,
  )
  # Nothing past revisit matching reads the ORB frames, the descriptors' last holder.
  front_orb.clear()
  release_freed_memory()
  lap("revisit")

  points = triangulate_lens(
    {t: kept[t] for t in kept if t in landmarks_prealign},
    body_poses,
    front_lens.body_T_lens,
    front_lens.params,
    args.frontend.min_track_length,
    max_landmark_depth=args.frontend.max_landmark_depth,
  )
  points = cap_landmarks(points, kept, args.frontend.max_landmarks)
  if revisits:
    # After the cap, so the revisit landmarks survive the landmark budget.
    add_revisit_landmarks(
      points,
      kept,
      revisits,
      body_poses,
      front_lens,
      args.frontend,
      slot=slot,
      anchor_kfs=set(anchor_obs_per_kf),
    )

  points, windowed = split_tracks_to_window(
    points,
    {t: kept[t] for t in points if t in kept},
    args.frontend.max_track_span,
  )
  tracks = {
    t: subsample_obs(obs, args.frontend.max_obs_per_track)
    for t, obs in windowed.items()
  }
  points, tracks = renumber_landmarks(points, tracks)

  lap("landmarks")

  v_max = float(max(np.linalg.norm(v) for v in velocities))
  print(f"  [{slot}/vi] LSQ |v|max={v_max:.2f} m/s")
  biases = [bias0 for _ in range(n_keyframes)]

  body_T_base_tag, finger_reproj, finger_frame_angles = recover_gripper_tags(
    tags,
    args.joints,
    rear_tags_per_kf,
    rear_lens,
    args.gripper.tag_size,
    n_keyframes,
    rear_tags_per_frame,
    rear_frame_times,
  )
  lap("gripper tags")

  return PreparedCamera(
    slot=slot,
    body_poses=body_poses,
    velocities=velocities,
    biases=biases,
    preintegrated=pim,
    front=LensChannel(
      lens="front",
      setup=front_lens,
      tags=front_tags,
      keypoints=keypoints[:n_keyframes],
      tracks=tracks,
      points=points,
      rejected=rejected,
    ),
    rear=LensChannel(lens="rear", setup=rear_lens, tags=rear_tags_per_kf[:n_keyframes]),
    anchor_obs_by_tag=obs_by_tag,
    anchor_locals_by_tag=anchor_locals_by_tag,
    body_T_base_tag=body_T_base_tag,
    keyframe_times=keyframe_times,
    keyframe_frames=np.round(keyframe_times * fps).astype(np.int64),
    finger_reproj=finger_reproj,
    tags=tags,
    imu=imu,
    finger_frame_angles=finger_frame_angles,
  )
