"""One camera's lenses and IMU prepared for the joint graph."""

from __future__ import annotations

import ctypes
from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeVar

import cv2
import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import MeiParams, mei_unproject
from specialist.record import schema
from specialist.record.frames import LensFrames, LensVideo
from specialist.record.session import (
  SessionRecording,
  covered_samples,
  fps_from_times,
  nearest_samples,
)
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.configs import (
  AnchorCfg,
  FrontendCfg,
  GripperCfg,
)
from specialist.vi_ba.features import (
  Frame,
  Rect,
  available_ram,
  detect_orb,
  mask_plane,
  match_pair,
  parallel_map,
)
from specialist.vi_ba.finger_kinematics import FingerReproj, build_finger_reproj
from specialist.vi_ba.imu import (
  GRAVITY,
  GRAVITY_WORLD,
  ImuSamples,
  find_stationary_windows,
  keyframe_delta_rotations,
  make_preintegration_params,
  preintegrate_between_keyframes,
)
from specialist.vi_ba.init import (
  Alignment,
  apply_alignment_to_poses,
  body_poses_from_lens_poses,
  incremental_sfm,
  interpolate_camera_poses,
  pick_init_pair,
  two_view_essential,
  visual_inertial_align,
)
from specialist.vi_ba.tag_detection import (
  TagDetection,
  configure_detector,
  detect_all_tags,
  lens_T_tag_per_kf,
  tag_half_angle,
)
from specialist.vi_ba.tracks import (
  filter_tracks_by_parallax,
  run_lens_frontend,
  triangulate_lens,
  unproject_tracks,
)
from specialist.vi_ba.urdf_joints import read_revolute_joints

Float64Array = npt.NDArray[np.float64]

_A = TypeVar("_A")

# ORB loop-closure track recovery (``_revisit_tracks``).
_REVISIT_N_NEIGHBORS = 8  # spatially-closest non-adjacent revisit kfs to re-match
_REVISIT_MATCH_RATIO = 0.8  # Lowe ratio-test threshold
_REVISIT_RANSAC_THRESHOLD = 4.5  # homography RANSAC reprojection tolerance, px
_REVISIT_MIN_VIEWS = 3  # keyframes a union-found track must span to be kept
_REVISIT_MIN_INLIERS = 10  # geometrically-verified matches required per pair

# Visual-odometry pose-spike rejection (``_reject_vo_pose_outliers``).
_VO_REJECT_FACTOR = 8.0  # residual/median-step ratio marking a PnP teleport

# Keyframes read at once from a banded lens.
_PREP_BAND = 256

# Share of the available memory the concurrent decodes may claim between them.
_PREP_MEMORY_SHARE = 0.35


__all__ = [
  "PreparedCamera",
  "CameraArgs",
  "cameras_that_fit",
  "prepare_camera",
]


def resolve_keyframe_every(keyframe_every: int, fps: float) -> int:
  """The keyframe stride: ``keyframe_every`` when positive, else the ~0.1 s lattice
  stride; raise when an override is off that lattice."""
  stride = schema.keyframe_stride(fps)
  if keyframe_every > 0 and keyframe_every % stride:
    raise ValueError(
      f"--keyframe-every {keyframe_every} does not land on the {stride}-frame "
      f"keyframe lattice at {fps:.2f} fps; use a multiple of {stride}"
    )
  return keyframe_every if keyframe_every > 0 else stride


def release_freed_memory() -> None:
  """Return freed heap to the OS; a no-op without glibc's ``malloc_trim``."""
  try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
  except (OSError, AttributeError):
    pass


def _per_band(
  fn: Callable[[list[np.ndarray]], list[_A]], frames: Sequence[np.ndarray]
) -> list[_A]:
  """Map ``fn`` band by band, so a banded sequence decodes each band once, in order."""
  out: list[_A] = []
  for begin in range(0, len(frames), _PREP_BAND):
    end = min(begin + _PREP_BAND, len(frames))
    out.extend(fn([frames[i] for i in range(begin, end)]))
  return out


def _enough_registered(
  n_registered: int, n_keyframes: int, ratio: float = 0.25
) -> bool:
  """A seed needs ``ratio`` of the keyframes, and at least eight."""
  return n_registered >= max(8, int(n_keyframes * ratio))


@dataclass
class CameraArgs:
  """Per-tracker prep knobs composed from the nested configs in ``configs.py``."""

  session: SessionRecording
  slot: str  # the camera's slot in the base recording
  base_tag_id: int
  anchor: AnchorCfg
  keyframe_every: int  # 0 = auto (round(fps/10))
  urdf_path: Path | None  # gripper URDF for finger joint kinematics
  # 0 disables the corresponding URDF revolute joint angle recovery.
  left_finger_tag_id: int
  right_finger_tag_id: int
  frontend: FrontendCfg
  gripper: GripperCfg
  # Set when sibling captures prep at once; it gates per-thread AprilTag detectors.
  parallel_siblings: bool = False


@dataclass
class PreparedCamera:
  """Everything one camera contributes to the joint graph + viz."""

  name: str
  body_poses: list[gtsam.Pose3]
  velocities: list[Float64Array]
  biases: list[gtsam.imuBias.ConstantBias]
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements]
  front_lens: LensSetup
  rear_lens: LensSetup
  front_tracks: dict[int, dict[int, Float64Array]]
  rear_tracks: dict[int, dict[int, Float64Array]]
  front_points: dict[int, Float64Array]
  rear_points: dict[int, Float64Array]
  # ``{tag_id: {kf: {side: lens_T_tag}}}``.
  anchor_obs_by_tag: dict[int, dict[int, dict[str, gtsam.Pose3]]]
  anchor_locals_by_tag: dict[int, gtsam.Pose3]
  body_T_base_tag: gtsam.Pose3 | None
  keyframe_times: Float64Array
  # Session-clock frame per keyframe; ``schema.keyframe_index`` maps it to the timeline.
  keyframe_frames: npt.NDArray[np.int64]
  front_keypoints: list[Float64Array]
  rear_keypoints: list[Float64Array]
  front_tags_per_kf: list[list[TagDetection]]
  rear_tags_per_kf: list[list[TagDetection]]
  front_rejected: dict[int, dict[int, Float64Array]]
  rear_rejected: dict[int, dict[int, Float64Array]]
  rear_static_mask_rects: list[tuple[int, int, int, int]]
  # URDF joint name -> kinematic inputs for the joint-angle reprojection factors.
  finger_reproj: dict[str, FingerReproj]
  left_finger_tag_id: int
  right_finger_tag_id: int
  # Body-frame, rescaled IMU samples, for IMU-rate propagation between keyframes.
  imu: ImuSamples | None = None
  # Per-joint raw ``(times, angles)`` at the finger rate; export pins them to the solve.
  finger_frame_angles: dict[str, tuple[Float64Array, Float64Array]] | None = None
  # Joint name -> solved angle about the joint z-axis per keyframe, NaN where unseen.
  joint_angles: dict[str, list[float]] = field(default_factory=dict)

  @property
  def n_keyframes(self) -> int:
    return len(self.body_poses)


def _build_non_anchor_tag_masks(
  shape: tuple[int, int],
  tags_per_kf: list[list[TagDetection]],
  exclude_tag_ids: set[int],
  pad_px: int,
) -> list[list[Rect]]:
  """Per keyframe, the padded, clipped box of each tag not in ``exclude_tag_ids``."""
  h, w = shape
  blocked: list[list[Rect]] = []
  for tags in tags_per_kf:
    rects: list[Rect] = []
    for tag in tags:
      if int(tag.tag_id) in exclude_tag_ids:
        continue
      corners = np.asarray(tag.corners, dtype=np.float64)
      x0 = max(int(np.floor(corners[:, 0].min())) - pad_px, 0)
      y0 = max(int(np.floor(corners[:, 1].min())) - pad_px, 0)
      x1 = min(int(np.ceil(corners[:, 0].max())) + pad_px, w)
      y1 = min(int(np.ceil(corners[:, 1].max())) + pad_px, h)
      if x1 > x0 and y1 > y0:
        rects.append((x0, y0, x1, y1))
    blocked.append(rects)
  return blocked


def _apply_static_mask_rects(
  blocked: list[list[Rect]], rects: list[Rect], shape: tuple[int, int]
) -> None:
  if not rects or not blocked:
    return
  h, w = shape
  for x1, y1, x2, y2 in rects:
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
      raise ValueError(
        f"mask_rear_rects entry {(x1, y1, x2, y2)} is empty after clamping to the "
        f"{w}x{h} frame — check the coordinate order (x1 y1 x2 y2) and bounds"
      )
    for frame_rects in blocked:
      frame_rects.append((x1c, y1c, x2c, y2c))


# Anchor-detection lens side: name, lens, per-kf tag detections, gyro increments.
_LensSide = tuple[str, LensSetup, list[list[TagDetection]], list[Float64Array]]


def _detect_anchor_obs(
  tag_id: int,
  tag_size: float,
  lens_sides: tuple[_LensSide, ...],
) -> dict[int, dict[str, gtsam.Pose3]]:
  """``{kf: {side: lens_T_tag}}`` for one anchor id over the given lenses, ungated."""
  obs: dict[int, dict[str, gtsam.Pose3]] = {}
  for side, lens, tags, rotations in lens_sides:
    for kf, pose in lens_T_tag_per_kf(
      tags,
      tag_id,
      tag_size,
      lens.params,
      kf_rotations=rotations,
    ).items():
      obs.setdefault(kf, {})[side] = pose
  return obs


def _choose_prep_anchor(
  tag_ids: tuple[int, ...],
  obs_by_tag: dict[int, dict[int, dict[str, gtsam.Pose3]]],
) -> int:
  """The id sighted at most keyframes; a tie goes to the earlier ``tag_ids`` entry."""
  return max(tag_ids, key=lambda tid: len(obs_by_tag.get(tid, {})))


def _print_anchor_geometry(
  slot: str,
  tag_id: int,
  lens_sides: tuple[_LensSide, ...],
  obs: dict[int, dict[str, gtsam.Pose3]],
) -> None:
  """Print per-anchor edge length (px), bearing half-angle and PnP incidence angle."""
  edges: list[float] = []
  half_angles: list[float] = []
  for _side, lens, tags, _rotations in lens_sides:
    for dets in tags:
      for det in dets:
        if int(det.tag_id) != tag_id:
          continue
        corners = det.corners
        edge = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        edges.append(float(edge.min()))
        half_angle = tag_half_angle(det, lens.params)
        if np.isfinite(half_angle):
          half_angles.append(float(np.degrees(half_angle)))
  if not edges:
    return
  incidences: list[float] = []
  for sides in obs.values():
    for pose in sides.values():
      t = np.asarray(pose.translation(), dtype=np.float64)
      dist = float(np.linalg.norm(t))
      if dist < 1e-9:
        continue
      cos_incidence = -float(pose.rotation().matrix()[:, 2] @ t) / dist
      incidences.append(float(np.degrees(np.arccos(np.clip(cos_incidence, -1.0, 1.0)))))

  def med_p90(values: list[float]) -> str:
    if not values:
      return "-/-"
    return f"{float(np.median(values)):.0f}/{float(np.percentile(values, 90)):.0f}"

  print(
    f"  [{slot}] anchor {tag_id} geometry: {len(edges)} sightings, "
    f"edge px min/median {min(edges):.0f}/{float(np.median(edges)):.0f}, "
    f"half-angle median/p90 {med_p90(half_angles)} deg, "
    f"incidence median/p90 {med_p90(incidences)} deg"
  )


def _first_sighting_local(
  obs: dict[int, dict[str, gtsam.Pose3]],
  front_lens: LensSetup,
  rear_lens: LensSetup,
  body_poses: list[gtsam.Pose3],
) -> gtsam.Pose3:
  """An anchor's local-world pose from its first sighting, front lens preferred."""
  first_kf = min(obs)
  lens, lens_T_tag = _side_lens_and_pose(obs[first_kf], front_lens, rear_lens)
  return body_poses[first_kf].compose(lens.body_T_lens).compose(lens_T_tag)


def _translation(pose: gtsam.Pose3) -> Float64Array:
  return np.asarray(pose.translation(), dtype=np.float64)


def _mean_pose(poses: list[gtsam.Pose3]) -> gtsam.Pose3:
  """Mean translation and log-mean rotation about the first pose."""
  ref = poses[0]
  translations = np.stack([_translation(p) for p in poses])
  log_sum = np.zeros(3)
  for p in poses:
    log_sum += gtsam.Rot3.Logmap(ref.rotation().between(p.rotation()))
  delta = gtsam.Rot3.Expmap(log_sum / len(poses))
  return gtsam.Pose3(
    ref.rotation().compose(delta), gtsam.Point3(*translations.mean(axis=0))
  )


def _bearings_by_track(
  tracks: dict[int, dict[int, Float64Array]], params: MeiParams
) -> dict[int, dict[int, Float64Array]]:
  """``{tid: {kf: unit bearing}}`` for the visual-odometry front-end."""
  return {
    tid: dict(zip(kfs, bearings, strict=True))
    for tid, kfs, bearings in unproject_tracks(tracks, params, min_obs=1)
  }


def _subsample_obs(
  track: dict[int, Float64Array], max_obs: int
) -> dict[int, Float64Array]:
  """Keep at most ``max_obs`` observations, the endpoints plus an even middle spread."""
  kfs = sorted(track)
  if max_obs <= 0 or len(kfs) <= max_obs:
    return track
  idx = np.unique(np.linspace(0, len(kfs) - 1, max_obs).round().astype(int))
  return {kfs[i]: track[kfs[i]] for i in idx}


def _cap_landmarks(
  points: dict[int, Float64Array],
  tracks: dict[int, dict[int, Float64Array]],
  cap: int,
) -> dict[int, Float64Array]:
  """Keep at most ``cap`` landmarks, longest tracks first; ``cap <= 0`` keeps all."""
  if cap <= 0 or len(points) <= cap:
    return points
  keep = set(sorted(points, key=lambda t: -len(tracks.get(t, {})))[:cap])
  return {t: p for t, p in points.items() if t in keep}


def _revisit_tracks(
  orb: list[Frame],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  base_track_id: int,
  max_center_dist: float,
  min_kf_gap: int,
) -> dict[int, dict[int, Float64Array]]:
  """Recover tracks broken by fast lifts via ORB loop closure."""
  centers = np.asarray([_translation(p.compose(lens.body_T_lens)) for p in body_poses])
  n = len(orb)
  parent: dict[tuple[int, int], tuple[int, int]] = {}

  def find(x: tuple[int, int]) -> tuple[int, int]:
    parent.setdefault(x, x)
    root = x
    while parent[root] != root:
      root = parent[root]
    while parent[x] != root:
      parent[x], x = root, parent[x]
    return root

  pairs: list[tuple[int, int]] = []
  for i in range(n):
    if len(orb[i].keypoints) == 0:
      continue
    dist = np.linalg.norm(centers - centers[i], axis=1)
    cand = [
      int(j)
      for j in np.argsort(dist)
      if j > i and (j - i) > min_kf_gap and dist[j] <= max_center_dist
    ][:_REVISIT_N_NEIGHBORS]
    pairs.extend((i, j) for j in cand)

  def verify_pair(i: int, j: int) -> npt.NDArray[np.int64] | None:
    m = match_pair(orb[i], orb[j], _REVISIT_MATCH_RATIO)
    if len(m) < _REVISIT_MIN_INLIERS:
      return None
    pa = orb[i].keypoints[m[:, 0]]
    pb = orb[j].keypoints[m[:, 1]]
    # The cv2 RNG is thread-local, so a per-pair seed keeps RANSAC deterministic.
    cv2.setRNGSeed(i * n + j)
    _, mask = cv2.findHomography(pa, pb, cv2.RANSAC, _REVISIT_RANSAC_THRESHOLD)
    if mask is None:
      return None
    inl = mask.ravel().astype(bool)
    if int(inl.sum()) < _REVISIT_MIN_INLIERS:
      return None
    return np.asarray(m[inl], dtype=np.int64)

  verified = parallel_map(verify_pair, [i for i, _ in pairs], [j for _, j in pairs])
  for (i, j), matches in zip(pairs, verified, strict=True):
    if matches is None:
      continue
    for a_idx, b_idx in matches:
      ra = find((i, int(a_idx)))
      rb = find((j, int(b_idx)))
      if ra != rb:
        parent[ra] = rb

  comps: dict[tuple[int, int], dict[int, int]] = {}
  for node in list(parent):
    comps.setdefault(find(node), {})[node[0]] = node[1]
  tracks: dict[int, dict[int, Float64Array]] = {}
  tid = base_track_id + 1
  for by_kf in comps.values():
    if len(by_kf) < _REVISIT_MIN_VIEWS:
      continue
    # Store pixels; downstream code unprojects through the MEI model itself.
    tracks[tid] = {
      kf: np.asarray(orb[kf].keypoints[kp], dtype=np.float64)
      for kf, kp in by_kf.items()
    }
    tid += 1
  return tracks


def _add_revisit_landmarks(
  front_points: dict[int, Float64Array],
  front_kept: dict[int, dict[int, Float64Array]],
  revisits: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  *,
  min_track_length: int,
  max_landmark_depth: float,
  slot: str,
  anchor_kfs: set[int] | None = None,
) -> None:
  """Add ``revisits`` to ``front_kept`` and their landmarks to ``front_points``."""
  if anchor_kfs:
    tri_tracks = {
      t: {k: v for k, v in obs.items() if k in anchor_kfs}
      for t, obs in revisits.items()
    }
  else:
    tri_tracks = revisits
  revisit_pts = triangulate_lens(
    tri_tracks,
    body_poses,
    lens.body_T_lens,
    lens.params,
    min_track_length,
    max_landmark_depth=max_landmark_depth,
  )
  front_points.update(revisit_pts)
  front_kept.update(revisits)
  print(
    f"  [{slot}] loop closure: +{len(revisit_pts)} landmarks "
    f"from {len(revisits)} revisit tracks"
  )


def _reject_vo_pose_outliers(
  lens_poses: dict[int, gtsam.Pose3],
) -> tuple[dict[int, gtsam.Pose3], list[int]]:
  """Drop incremental-SfM keyframes whose camera center is a local spike."""
  kept = dict(lens_poses)
  removed: list[int] = []
  for _ in range(len(lens_poses)):
    kfs = sorted(kept)
    if len(kfs) < 5:
      break
    centers = {kf: _translation(kept[kf]) for kf in kfs}
    steps = [
      float(np.linalg.norm(centers[kfs[i + 1]] - centers[kfs[i]]))
      / (kfs[i + 1] - kfs[i])
      for i in range(len(kfs) - 1)
    ]
    median_step = float(np.median(steps))
    if median_step < 1e-12:
      break
    worst_kf: int | None = None
    worst_residual = 0.0
    for i in range(1, len(kfs) - 1):
      prev_kf, kf, next_kf = kfs[i - 1], kfs[i], kfs[i + 1]
      frac = (kf - prev_kf) / (next_kf - prev_kf)
      interpolated = (1.0 - frac) * centers[prev_kf] + frac * centers[next_kf]
      residual = float(np.linalg.norm(centers[kf] - interpolated))
      if residual > worst_residual:
        worst_residual, worst_kf = residual, kf
    if worst_kf is not None and worst_residual > _VO_REJECT_FACTOR * median_step:
      del kept[worst_kf]
      removed.append(worst_kf)
    else:
      break
  return kept, sorted(removed)


def _fill_lens_poses(
  lens_poses: dict[int, gtsam.Pose3], n_keyframes: int
) -> list[gtsam.Pose3]:
  """Every keyframe's camera pose; gaps interpolate on SE(3) and hold at the ends."""
  reg = sorted(lens_poses)
  out: list[gtsam.Pose3] = []
  for k in range(n_keyframes):
    after = bisect_left(reg, k)
    lo, hi = reg[max(after - 1, 0)], reg[min(after, len(reg) - 1)]
    pa, pb = lens_poses[lo], lens_poses[hi]
    if hi in (k, lo):
      out.append(pb)
    else:
      f = (k - lo) / (hi - lo)
      out.append(pa.compose(gtsam.Pose3.Expmap(f * gtsam.Pose3.Logmap(pa.between(pb)))))
  return out


def _anchor_pose_seed(
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]],
  body_T_front_lens: gtsam.Pose3,
  n_keyframes: int,
) -> list[gtsam.Pose3] | None:
  """Anchor-PnP body poses, gaps filled; ``None`` when too few keyframes see it."""
  lens_poses = {
    kf: sides["front"].inverse()
    for kf, sides in anchor_obs_per_kf.items()
    if "front" in sides
  }
  if not _enough_registered(len(lens_poses), n_keyframes):
    return None
  return body_poses_from_lens_poses(
    _fill_lens_poses(lens_poses, n_keyframes), body_T_front_lens
  )


def _anchor_metric_scale(
  body_poses_unscaled: list[gtsam.Pose3],
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]],
  front_lens: LensSetup,
  rear_lens: LensSetup,
) -> float | None:
  """Metric scale: anchor-implied over visual body-position spread, or ``None``."""
  vo: list[Float64Array] = []
  anc: list[Float64Array] = []
  for kf, sides in anchor_obs_per_kf.items():
    lens, lens_T_tag = _side_lens_and_pose(sides, front_lens, rear_lens)
    body_in_tag = lens.body_T_lens.compose(lens_T_tag).inverse()
    anc.append(_translation(body_in_tag))
    vo.append(_translation(body_poses_unscaled[kf]))
  if len(vo) < 3:
    return None
  vo_arr = np.asarray(vo, dtype=np.float64)
  anc_arr = np.asarray(anc, dtype=np.float64)
  vo_spread = float(np.sqrt(((vo_arr - vo_arr.mean(axis=0)) ** 2).sum()))
  anc_spread = float(np.sqrt(((anc_arr - anc_arr.mean(axis=0)) ** 2).sum()))
  if vo_spread < 1e-9:
    return None
  return anc_spread / vo_spread


def _side_lens_and_pose(
  sides: dict[str, gtsam.Pose3], front_lens: LensSetup, rear_lens: LensSetup
) -> tuple[LensSetup, gtsam.Pose3]:
  """The anchor observation's lens and ``lens_T_tag``, front lens preferred."""
  if "front" in sides:
    return front_lens, sides["front"]
  return rear_lens, sides["rear"]


def _imu_and_initial_gravity(
  imu: ImuSamples,
  raw_factory_bias: gtsam.imuBias.ConstantBias,
  body_R_imu: Float64Array,
  keyframe_times: Float64Array,
  body_poses_prealign: list[gtsam.Pose3],
) -> tuple[ImuSamples, gtsam.imuBias.ConstantBias, Float64Array | None]:
  """IMU samples, body-frame bias, world gravity or ``None`` if no stationary window."""
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


def _recover_gripper_tags(
  args: CameraArgs,
  rear_tags_per_kf: list[list[TagDetection]],
  rear_lens: LensSetup,
  n_keyframes: int,
  kf_rotations: list[Float64Array],
  rear_tags_per_frame: list[list[TagDetection]] | None = None,
  frame_times: Float64Array | None = None,
) -> tuple[
  gtsam.Pose3 | None,
  dict[str, FingerReproj],
  dict[str, tuple[Float64Array, Float64Array]],
]:
  """Base-tag body pose and, per finger, reprojection inputs and frame angles."""
  base_tag_obs = lens_T_tag_per_kf(
    rear_tags_per_kf,
    tag_id=args.base_tag_id,
    tag_size=args.gripper.tag_size,
    params=rear_lens.params,
    kf_rotations=kf_rotations,
  )
  body_T_base_tag: gtsam.Pose3 | None = None
  if base_tag_obs:
    body_T_base_tag = _mean_pose(
      [rear_lens.body_T_lens.compose(p) for p in base_tag_obs.values()]
    )
  if args.base_tag_id > 0 and body_T_base_tag is None:
    raise ValueError(
      f"base tag {args.base_tag_id} was configured but never detected in the rear "
      f"lens across {n_keyframes} keyframes — its base-tag, finger, and cross-tracker "
      f"factors would all be silently disabled. Wrong id, or the tag out of the "
      f"rear-lens field of view?"
    )

  # URDF joints sit in the base frame; body_T_base_tag carries them to the body.
  joints = read_revolute_joints(args.urdf_path) if args.urdf_path is not None else {}
  finger_reproj: dict[str, FingerReproj] = {}
  frame_angles: dict[str, tuple[Float64Array, Float64Array]] = {}
  for joint_name, finger_id in (
    ("left_finger", args.left_finger_tag_id),
    ("right_finger", args.right_finger_tag_id),
  ):
    if finger_id <= 0 or args.urdf_path is None:
      continue
    joint = joints.get(joint_name)
    if joint is None or joint.joint_T_tag0 is None:
      raise ValueError(
        f"finger tag {finger_id} was configured for '{joint_name}', but "
        f"{args.urdf_path} has no revolute joint '{joint_name}' carrying an "
        f"AprilTag frame (revolute joints present: {sorted(joints)}). Wrong URDF?"
      )
    if body_T_base_tag is None:
      continue
    reproj = build_finger_reproj(
      joint,
      rear_tags_per_kf[:n_keyframes],
      rear_lens,
      finger_id,
      body_T_base_tag,
      args.gripper.tag_size,
    )
    if reproj is None:
      print(
        f"  [warn] finger tag {finger_id} ({joint_name}) never detected in the "
        f"rear keyframes; its angle stays unsolved"
      )
    else:
      finger_reproj[joint_name] = reproj
    if rear_tags_per_frame is None or frame_times is None:
      continue
    framed = build_finger_reproj(
      joint,
      rear_tags_per_frame,
      rear_lens,
      finger_id,
      body_T_base_tag,
      args.gripper.tag_size,
    )
    if framed is not None:
      frame_angles[joint_name] = (
        frame_times[list(framed.observations)],
        np.array([o.theta_seed for o in framed.observations.values()]),
      )
  return body_T_base_tag, finger_reproj, frame_angles


@dataclass
class _LensChannel:
  """One lens's prep state: tag detections in, surviving tracks and landmarks out."""

  setup: LensSetup
  tags: list[list[TagDetection]]
  feature_frames: list[Frame] = field(default_factory=list)
  kept: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  rejected: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  landmarks_prealign: dict[int, Float64Array] = field(default_factory=dict)
  points: dict[int, Float64Array] = field(default_factory=dict)
  tracks: dict[int, dict[int, Float64Array]] = field(default_factory=dict)


def _seed_body_poses(
  slot: str,
  front: _LensChannel,
  rear: _LensChannel,
  n_keyframes: int,
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]],
  cfg: FrontendCfg,
) -> tuple[list[gtsam.Pose3], bool]:
  """The pre-alignment body-pose seed and whether it came from the anchor."""
  sfm_cams, _ = incremental_sfm(
    _bearings_by_track(front.kept, front.setup.params),
    min_shared=cfg.init_min_shared_tracks,
  )
  sfm_cams, vo_outliers = _reject_vo_pose_outliers(sfm_cams)
  if vo_outliers:
    print(f"  [{slot}] visual-odometry rejected {len(vo_outliers)} rotation-spike kfs")
  if _enough_registered(len(sfm_cams), n_keyframes, cfg.vo_min_registered_ratio):
    print(
      f"  [{slot}] visual-odometry init: registered "
      f"{len(sfm_cams)}/{n_keyframes} keyframes"
    )
    filled = _fill_lens_poses(sfm_cams, n_keyframes)
    return body_poses_from_lens_poses(filled, front.setup.body_T_lens), False

  anchor_seed = _anchor_pose_seed(
    anchor_obs_per_kf, front.setup.body_T_lens, n_keyframes
  )
  if anchor_seed is not None:
    n_anchor_kf = sum("front" in sides for sides in anchor_obs_per_kf.values())
    print(
      f"  [{slot}] visual-odometry starved ({len(sfm_cams)}/{n_keyframes}); "
      f"anchor-pose init from {n_anchor_kf} anchor keyframes"
    )
    return anchor_seed, True
  print(
    f"  [{slot}] visual-odometry init: only {len(sfm_cams)}/{n_keyframes} "
    f"keyframes registered — using straight-line init"
  )
  init_ch, (init_first, init_last, init_shared) = max(
    (
      (ch, pick_init_pair(ch.kept, n_keyframes, cfg.init_min_shared_tracks))
      for ch in (front, rear)
    ),
    key=lambda candidate: candidate[1][2],
  )
  if init_shared < 8:
    raise RuntimeError(f"{slot}: no lens has ≥8 shared tracks (best={init_shared})")
  if init_first:
    print(f"  [{slot}] init pair based at keyframe {init_first}", flush=True)
  shared = [
    obs for obs in init_ch.kept.values() if init_first in obs and init_last in obs
  ]
  pix_a = np.asarray([obs[init_first] for obs in shared], dtype=np.float64)
  pix_b = np.asarray([obs[init_last] for obs in shared], dtype=np.float64)
  r_cw_last, t_cw_last, _ = two_view_essential(
    mei_unproject(pix_a, init_ch.setup.params),
    mei_unproject(pix_b, init_ch.setup.params),
  )
  lens_poses = interpolate_camera_poses(
    r_cw_last, t_cw_last, n_keyframes, init_first=init_first, init_last=init_last
  )
  return body_poses_from_lens_poses(lens_poses, init_ch.setup.body_T_lens), False


def _align_visual_inertial(
  slot: str,
  imu: ImuSamples,
  keyframe_times: Float64Array,
  bias0: gtsam.imuBias.ConstantBias,
  body_poses_prealign: list[gtsam.Pose3],
  gravity_world_initial: Float64Array | None,
  fixed_scale: float | None,
) -> tuple[
  list[gtsam.Pose3],
  list[Float64Array],
  list[gtsam.PreintegratedCombinedMeasurements],
  Float64Array,
]:
  """Gravity-aligned poses, velocities, preintegration and the prep-frame 3x3 map."""

  # Gravity never enters the deltas, so one PIM serves alignment and export.
  pim = preintegrate_between_keyframes(
    make_preintegration_params(GRAVITY_WORLD), imu, keyframe_times, bias0
  )

  def align(poses: list[gtsam.Pose3]) -> Alignment:
    return visual_inertial_align(
      poses, pim, fixed_gravity=gravity_world_initial, fixed_scale=fixed_scale
    )

  alignment = align(body_poses_prealign)
  if alignment.scale < 0.0:
    origin = _translation(body_poses_prealign[0])
    body_poses_prealign = [
      gtsam.Pose3(p.rotation(), gtsam.Point3(*(2.0 * origin - _translation(p))))
      for p in body_poses_prealign
    ]
    alignment = align(body_poses_prealign)
  # Gravity takes the stationary IMU reading exactly; the weighted pin holds it only near.
  if gravity_world_initial is not None:
    alignment = replace(alignment, gravity_world=gravity_world_initial)
  print(f"  [{slot}] VI scale={alignment.scale:.4f}")
  body_poses, world_R_prealign = apply_alignment_to_poses(
    body_poses_prealign, alignment
  )
  world_rot = world_R_prealign.matrix()
  velocities = [world_rot @ v for v in alignment.velocities]
  return body_poses, velocities, pim, alignment.scale * world_rot


def cameras_that_fit(session: SessionRecording, slots: Sequence[str]) -> int:
  """How many cameras may prep at once within the memory on hand."""
  if len(slots) <= 1:
    return len(slots)
  height, width = session.decoded_shape(slots[0])
  per_camera = _PREP_BAND * width * height  # one lens's resident band at a time
  shared = session.path.stat().st_size
  budget = available_ram() * _PREP_MEMORY_SHARE - shared
  affordable = int(budget / per_camera)
  return max(1, min(len(slots), affordable))


def keyframe_grid(
  times: Float64Array,
  covered: npt.NDArray[np.int64],
  fps: float,
  keyframe_every: int,
) -> npt.NDArray[np.int64]:
  """Keyframe rows of the covered span, phased onto the stride lattice when possible."""
  stride = schema.keyframe_stride(fps)
  session_frames = np.round(times * fps).astype(np.int64)
  on_lattice = np.flatnonzero(session_frames[covered] % stride == 0)
  phase = int(on_lattice[0]) if len(on_lattice) else 0
  return covered[phase::keyframe_every]


def finger_stride(keyframe_every: int, fps: float) -> int:
  """The finger stride: the largest divisor of ``keyframe_every`` up to ``fps / 30``."""
  return max(
    (
      d
      for d in range(1, keyframe_every + 1)
      if keyframe_every % d == 0 and d <= fps / 30.0
    ),
    default=1,
  )


def prepare_camera(slot: str, args: CameraArgs) -> PreparedCamera:
  """Every per-camera stage; body poses land in the camera's gravity-aligned world."""
  configure_detector(args.frontend.parallel_tag_detect and args.parallel_siblings)
  session = args.session
  front_video = session.lens_video(args.slot, "front")
  rear_video = session.lens_video(args.slot, "rear")
  fps = fps_from_times(front_video.times)

  keyframe_every = resolve_keyframe_every(args.keyframe_every, fps)
  raw_imu = session.imu(args.slot)
  covered = covered_samples(front_video.times, rear_video.times, raw_imu.t, fps)
  front_samples = keyframe_grid(front_video.times, covered, fps, keyframe_every)
  grid_times = front_video.times[front_samples]
  rear_kf_rows = nearest_samples(rear_video.times, grid_times)

  anchor_tag_ids = set(args.anchor.tag_ids)

  def tag_masks(
    shape: tuple[int, int], tags_per_kf: list[list[TagDetection]]
  ) -> list[list[Rect]]:
    """Per-keyframe rectangles over every non-anchor tag, empty with masking off."""
    if not args.frontend.mask_tags:
      return [[] for _ in tags_per_kf]
    return _build_non_anchor_tag_masks(
      shape, tags_per_kf, anchor_tag_ids, args.frontend.mask_tag_pad
    )

  def walk_front() -> tuple[int, list[list[TagDetection]], list[Frame]]:
    """One banded walk for tags and loop-closure ORB, the ORB empty with closure off."""
    walk = LensFrames(front_video, front_samples, band=_PREP_BAND)
    width = int(walk[0].shape[1])
    if not args.frontend.revisit:
      return width, _per_band(detect_all_tags, walk), []

    def band_fn(
      band: list[np.ndarray],
    ) -> list[tuple[list[TagDetection], Frame]]:
      tags = detect_all_tags(band)
      rects = tag_masks(band[0].shape[:2], tags)
      orb = parallel_map(
        lambda f, m: detect_orb(
          f, args.frontend.orb_n_features, 20, mask_plane(f.shape[:2], m)
        ),
        band,
        rects,
      )
      return list(zip(tags, orb, strict=True))

    fused = _per_band(band_fn, walk)
    return width, [tags for tags, _ in fused], [orb for _, orb in fused]

  decoded_width, front_tags_per_kf, front_orb = walk_front()
  front_lens, rear_lens = session.lens_setups(args.slot, decoded_width)
  release_freed_memory()
  # With fingers configured, the rear walk detects tags at the finger stride.
  wants_fingers = args.urdf_path is not None and (
    args.left_finger_tag_id > 0 or args.right_finger_tag_id > 0
  )
  rear_tags_per_frame: list[list[TagDetection]] | None = None
  rear_frame_times: Float64Array | None = None
  if wants_fingers:
    stride = finger_stride(keyframe_every, fps)
    dense_samples = nearest_samples(
      rear_video.times, front_video.times[covered][::stride]
    )
    rear_tags_per_frame = _per_band(
      detect_all_tags, LensFrames(rear_video, dense_samples, band=_PREP_BAND)
    )
    rear_frame_times = rear_video.times[dense_samples]
    rear_tags_per_kf = rear_tags_per_frame[:: keyframe_every // stride]
  else:
    rear_tags_per_kf = _per_band(
      detect_all_tags, LensFrames(rear_video, rear_kf_rows, band=_PREP_BAND)
    )
  n_keyframes = min(len(front_tags_per_kf), len(rear_tags_per_kf))
  release_freed_memory()
  if n_keyframes < 3:
    raise RuntimeError(f"only {n_keyframes} keyframes decoded — window too short")
  front_tags_per_kf = front_tags_per_kf[:n_keyframes]
  rear_tags_per_kf = rear_tags_per_kf[:n_keyframes]
  keyframe_times = grid_times[:n_keyframes]

  body_R_imu = session.body_R_imu(args.slot)
  # Raw-axis samples rotated into the body frame; gyro is already rad/s.
  imu = ImuSamples(
    t=raw_imu.t,
    gyro=raw_imu.gyro @ body_R_imu.T,
    accel=raw_imu.accel @ body_R_imu.T,
  )
  body_deltas = keyframe_delta_rotations(imu, keyframe_times)
  del raw_imu

  def lens_kf_rotations(lens: LensSetup) -> list[Float64Array]:
    r_bl = lens.body_T_lens.rotation().matrix()
    return [r_bl.T @ d.T @ r_bl for d in body_deltas]

  front_kf_rotations = lens_kf_rotations(front_lens)
  rear_kf_rotations = lens_kf_rotations(rear_lens)

  lens_sides = (
    ("front", front_lens, front_tags_per_kf, front_kf_rotations),
    ("rear", rear_lens, rear_tags_per_kf, rear_kf_rotations),
  )
  obs_by_tag = {
    tag_id: _detect_anchor_obs(tag_id, args.anchor.tag_size, lens_sides)
    for tag_id in args.anchor.tag_ids
  }
  prep_anchor_tag_id = _choose_prep_anchor(args.anchor.tag_ids, obs_by_tag)
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]] = obs_by_tag.get(
    prep_anchor_tag_id, {}
  )
  if prep_anchor_tag_id != args.anchor.tag_ids[0]:
    n_ref = len(obs_by_tag.get(args.anchor.tag_ids[0], {}))
    print(
      f"  [{slot}] prep reference anchor: tag {prep_anchor_tag_id} "
      f"({len(anchor_obs_per_kf)} sighted keyframes vs {n_ref} for "
      f"tag {args.anchor.tag_ids[0]})"
    )
  for tag_id, obs in obs_by_tag.items():
    _print_anchor_geometry(slot, tag_id, lens_sides, obs)

  front = _LensChannel(setup=front_lens, tags=front_tags_per_kf)
  rear = _LensChannel(setup=rear_lens, tags=rear_tags_per_kf)
  anchor_tag_ids = set(obs_by_tag)
  # The rear frontend applies the rects, so a run without rear landmarks records none.
  static_rects = (
    list(args.frontend.mask_rear_rects) if args.frontend.rear_landmarks else []
  )

  def run_lens(
    ch: _LensChannel,
    video: LensVideo,
    rows: npt.NDArray[np.int64],
    blocked: list[Rect],
  ) -> None:
    """Decode this lens, take what downstream needs from its pixels, then drop them."""
    frames = LensFrames(video, rows, band=_PREP_BAND)
    shape = frames[0].shape[:2]
    masks = (
      _build_non_anchor_tag_masks(
        shape, ch.tags, anchor_tag_ids, args.frontend.mask_tag_pad
      )
      if args.frontend.mask_tags
      else [[] for _ in ch.tags]
    )
    _apply_static_mask_rects(masks, blocked, shape)
    ch.feature_frames, tracks = run_lens_frontend(
      frames, ch.setup.params, args.frontend, masks=masks
    )
    ch.kept = {
      t: o for t, o in tracks.items() if len(o) >= args.frontend.min_track_length
    }
    if args.frontend.min_parallax > 0.0:
      ch.kept, ch.rejected = filter_tracks_by_parallax(
        ch.kept, ch.setup.params, args.frontend.min_parallax
      )
    release_freed_memory()

  run_lens(front, front_video, front_samples, [])
  if args.frontend.rear_landmarks:
    run_lens(rear, rear_video, rear_kf_rows, static_rects)

  body_poses_prealign, anchor_seed_used = _seed_body_poses(
    slot, front, rear, n_keyframes, anchor_obs_per_kf, args.frontend
  )

  imu, bias0, gravity_world_initial = _imu_and_initial_gravity(
    imu,
    session.factory_bias(args.slot),
    body_R_imu,
    keyframe_times,
    body_poses_prealign,
  )

  for ch in (front, rear):
    ch.landmarks_prealign = triangulate_lens(
      ch.kept,
      body_poses_prealign,
      ch.setup.body_T_lens,
      ch.setup.params,
      args.frontend.min_track_length,
      min_landmark_depth=0.01,
      max_landmark_depth=1000.0,
    )

  # Scale: 1.0 for an anchor seed, else from a visible anchor, else floated on the IMU.
  anchor_scale = _anchor_metric_scale(
    body_poses_prealign, anchor_obs_per_kf, front_lens, rear_lens
  )
  body_poses, velocities, pim, prealign_to_world = _align_visual_inertial(
    slot,
    imu,
    keyframe_times,
    bias0,
    body_poses_prealign,
    gravity_world_initial,
    fixed_scale=1.0 if anchor_seed_used else anchor_scale,
  )

  # Each sighted anchor's pose in this camera's local world, for the shared-world BFS.
  anchor_locals_by_tag: dict[int, gtsam.Pose3] = {
    tag_id: _first_sighting_local(obs, front_lens, rear_lens, body_poses)
    for tag_id, obs in obs_by_tag.items()
    if obs
  }

  if anchor_obs_per_kf:
    revisits: dict[int, dict[int, Float64Array]] = {}
    if args.frontend.revisit:
      revisits = _revisit_tracks(
        front_orb[:n_keyframes],
        body_poses,
        front_lens,
        max(front.kept, default=0),
        args.frontend.loop_closure_max_dist,
        args.frontend.loop_closure_min_kf_gap,
      )
    for ch in (front, rear):
      ch.points = triangulate_lens(
        {t: ch.kept[t] for t in ch.kept if t in ch.landmarks_prealign},
        body_poses,
        ch.setup.body_T_lens,
        ch.setup.params,
        args.frontend.min_track_length,
        max_landmark_depth=args.frontend.max_landmark_depth,
      )
      ch.points = _cap_landmarks(
        ch.points, ch.kept, args.frontend.max_landmarks_per_lens
      )
    if revisits:
      # Merge after the cap so the per-lens budget never culls the revisit landmarks.
      _add_revisit_landmarks(
        front.points,
        front.kept,
        revisits,
        body_poses,
        front_lens,
        min_track_length=args.frontend.min_track_length,
        max_landmark_depth=args.frontend.max_landmark_depth,
        slot=slot,
        anchor_kfs=set(anchor_obs_per_kf) if args.frontend.anchored_revisit else None,
      )
  else:
    for ch in (front, rear):
      ch.points = {
        tid: prealign_to_world @ xyz for tid, xyz in ch.landmarks_prealign.items()
      }

  for ch in (front, rear):
    ch.tracks = {
      t: _subsample_obs(ch.kept[t], args.frontend.max_obs_per_track)
      for t in ch.points
      if t in ch.kept
    }

  v_max = float(max(np.linalg.norm(v) for v in velocities))
  print(f"  [{slot}] VIO LSQ |v|max={v_max:.2f} m/s")
  biases = [bias0 for _ in range(n_keyframes)]

  body_T_base_tag, finger_reproj, finger_frame_angles = _recover_gripper_tags(
    args,
    rear_tags_per_kf,
    rear_lens,
    n_keyframes,
    rear_kf_rotations,
    rear_tags_per_frame,
    rear_frame_times,
  )

  return PreparedCamera(
    name=slot,
    body_poses=body_poses,
    velocities=velocities,
    biases=biases,
    preintegrated=pim,
    front_lens=front_lens,
    rear_lens=rear_lens,
    front_tracks=front.tracks,
    rear_tracks=rear.tracks,
    front_points=front.points,
    rear_points=rear.points,
    anchor_obs_by_tag=obs_by_tag,
    anchor_locals_by_tag=anchor_locals_by_tag,
    body_T_base_tag=body_T_base_tag,
    keyframe_times=keyframe_times,
    keyframe_frames=np.round(keyframe_times * fps).astype(np.int64),
    front_keypoints=[f.keypoints for f in front.feature_frames][:n_keyframes],
    rear_keypoints=[f.keypoints for f in rear.feature_frames][:n_keyframes],
    front_tags_per_kf=front_tags_per_kf,
    rear_tags_per_kf=rear_tags_per_kf,
    front_rejected=front.rejected,
    rear_rejected=rear.rejected,
    rear_static_mask_rects=static_rects,
    finger_reproj=finger_reproj,
    left_finger_tag_id=args.left_finger_tag_id,
    right_finger_tag_id=args.right_finger_tag_id,
    imu=imu,
    finger_frame_angles=finger_frame_angles or None,
  )
