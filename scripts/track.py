"""VI-BA driver for audio-synced captures in the reference anchor's world."""

from __future__ import annotations

import itertools
import json
import time
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import gtsam
import numpy as np
import numpy.typing as npt
import tyro
from specialist.vi_ba._mei_factor_cpp import (
  make_mei_base_tag_factor_cpp,
  make_mei_cross_finger_factor_cpp,
  make_mei_cross_tag_factor_cpp,
  make_mei_finger_factor_cpp,
  make_mei_world_point_factor_cpp,
)

from specialist.loader.audio_sync import sync_two_claps
from specialist.loader.mei_model import MeiParams, mei_project, mei_unproject
from specialist.vi_ba import gpu_flow
from specialist.vi_ba.build import (
  LensSetup,
  add_mei_tag_factor,
  build_vi_graph,
  huber_pixel_noise,
)
from specialist.vi_ba.compose import CameraKeys
from specialist.vi_ba.configs import (
  AnchorCfg,
  FrontendCfg,
  GripperCfg,
  ImuCfg,
  SolverCfg,
  WindowCfg,
)
from specialist.vi_ba.features import set_precompute_concurrency
from specialist.vi_ba.landmark_cull import cull_outlier_landmarks
from specialist.vi_ba.optimize import OptimizeResult, optimize_graph
from specialist.vi_ba.pipeline import capture_duration
from specialist.vi_ba.single_camera import (
  CameraArgs,
  PreparedCamera,
  prepare_camera,
)
from specialist.vi_ba.tag_detection import (
  TagDetection,
  estimate_lens_T_tag,
  tag_corners_local,
)
from specialist.vi_ba.viz import CameraVizData, log_multi_ba_run

Float64Array = npt.NDArray[np.float64]

# Anchor re-estimate moves above these flag weak sighting geometry.
_ANCHOR_CONSISTENCY_TRANS = 5e-3
_ANCHOR_CONSISTENCY_ROT = float(np.radians(0.2))


def _lens_bearing(det: TagDetection, lens: LensSetup) -> Float64Array:
  """Lens-frame bearing of a tag's center, the cheirality ray for MEI factors."""
  return mei_unproject(
    np.asarray(det.center, dtype=np.float64).reshape(1, 2), lens.params
  )[0]


@dataclass
class Args:
  input_insvs: tuple[Path, ...]
  window: WindowCfg
  anchor: AnchorCfg
  output_dir: Path = Path("logs")
  slots: tuple[str, ...] = ()  # auto-derive ("tracker_0", "tracker_1", ...)
  urdf_path: Path | None = None
  log_frames_in_rrd: bool = False
  open_viewer: bool = True
  # One id per tracker; cross-tracker disambiguation needs distinct ids.
  base_tag_ids: tuple[int, ...] = ()
  # Per-tracker finger tag ids; 0 disables that tracker's finger.
  left_finger_tag_ids: tuple[int, ...] = ()
  right_finger_tag_ids: tuple[int, ...] = ()
  # Lighter frontend feature budget so the joint graph stays cheap; override per run.
  frontend: FrontendCfg = field(
    default_factory=lambda: FrontendCfg(orb_n_features=500, init_min_shared_tracks=30)
  )
  imu: ImuCfg = field(default_factory=ImuCfg)
  gripper: GripperCfg = field(default_factory=GripperCfg)
  solver: SolverCfg = field(default_factory=SolverCfg)


@dataclass
class Summary:
  sources: tuple[str, ...]
  slots: tuple[str, ...]
  audio_offsets: tuple[float, ...]
  per_tracker_keyframes: tuple[int, ...]
  anchor_obs_per_tracker: tuple[int, ...]
  cross_tracker_obs: int
  initial_graph_error: float
  final_graph_error: float
  solver_iterations: int
  bounding_box_diagonal: tuple[float, ...]
  trajectory_path_length: tuple[float, ...]


def _derive_names(
  input_insvs: tuple[Path, ...], names: tuple[str, ...]
) -> tuple[str, ...]:
  if names:
    if len(names) != len(input_insvs):
      raise ValueError(
        f"slots ({len(names)}) must match input_insvs ({len(input_insvs)})"
      )
    return names
  return tuple(f"tracker_{i}" for i in range(len(input_insvs)))


def _validate_tag_id_disjointness(args: Args) -> None:
  """Anchor, base and nonzero finger tag ids must be pairwise disjoint."""
  groups = (
    ("--anchor.tag-ids", set(args.anchor.tag_ids)),
    ("--base-tag-ids", {int(i) for i in args.base_tag_ids}),
    (
      "--left/right-finger-tag-ids",
      {
        int(i)
        for i in (*args.left_finger_tag_ids, *args.right_finger_tag_ids)
        if int(i) > 0
      },
    ),
  )
  for (name_a, ids_a), (name_b, ids_b) in itertools.combinations(groups, 2):
    overlap = ids_a & ids_b
    if overlap:
      raise ValueError(
        f"tag ids {sorted(overlap)} appear in both {name_a} and "
        f"{name_b}; anchor, base, and finger tags must be disjoint"
      )


def _audio_align_starts(
  input_insvs: tuple[Path, ...], master_start: float
) -> tuple[list[float], list[float]]:
  """Per-clip local-clock starts aligning every clip, and offsets relative to clip 0."""
  offsets = [0.0]
  for path in input_insvs[1:]:
    sync = sync_two_claps(input_insvs[0], path)
    offsets.append(sync.offset)
    print(
      f"audio sync: {input_insvs[0].name} ↔ {path.name}  "
      f"offset={sync.offset:+.3f}s  "
      f"matched_pairs={sync.matched_pairs} "
      f"(claps_a={sync.clap_times_a.size}, claps_b={sync.clap_times_b.size})"
    )
  lead = min(offsets)
  starts = [master_start + off - lead for off in offsets]
  return starts, offsets


def _bfs_world_alignments(
  anchor_locals_by_camera: dict[str, dict[int, gtsam.Pose3]],
  anchor_tag_ids: tuple[int, ...],
) -> tuple[dict[str, gtsam.Pose3], dict[int, gtsam.Pose3]]:
  """Camera alignments and anchor poses by BFS; raise on an unreachable tracker."""
  ref_tag_id = anchor_tag_ids[0]
  ref = next(
    (
      locals_[ref_tag_id]
      for locals_ in anchor_locals_by_camera.values()
      if ref_tag_id in locals_
    ),
    None,
  )
  if ref is None:
    raise RuntimeError("no tracker observes the reference anchor; cannot share a world")
  world_T_anchor: dict[int, gtsam.Pose3] = {ref_tag_id: ref}
  aligns: dict[str, gtsam.Pose3] = {}
  pending = list(anchor_locals_by_camera)
  while pending:
    aligned_now: list[str] = []
    for name in pending:
      locals_ = anchor_locals_by_camera[name]
      via = next(
        (a for a in anchor_tag_ids if a in world_T_anchor and a in locals_), None
      )
      if via is None:
        continue
      align = world_T_anchor[via].compose(locals_[via].inverse())
      aligns[name] = align
      for tag_id, local in locals_.items():
        if tag_id not in world_T_anchor:
          world_T_anchor[tag_id] = align.compose(local)
      aligned_now.append(name)
    if not aligned_now:
      raise RuntimeError(
        f"cameras {pending} sight no anchor with an established world pose "
        f"(established anchors: {sorted(world_T_anchor)}); the tracker-anchor "
        f"co-visibility graph is disconnected"
      )
    pending = [n for n in pending if n not in aligned_now]
  return aligns, world_T_anchor


def _align_to_shared_world(
  cameras: list[PreparedCamera], anchor_tag_ids: tuple[int, ...]
) -> gtsam.Pose3:
  """Move every tracker into the reference anchor's world; return its world pose."""
  locals_by_camera = {t.name: t.anchor_locals_by_tag for t in cameras}
  aligns, world_T_anchor = _bfs_world_alignments(locals_by_camera, anchor_tag_ids)
  for t in cameras:
    align = aligns[t.name]
    align_R = align.rotation().matrix()
    align_t = np.asarray(align.translation(), dtype=np.float64)
    t.body_poses = [align.compose(p) for p in t.body_poses]
    t.velocities = [align_R @ v for v in t.velocities]
    t.front_points = {tid: align_R @ p + align_t for tid, p in t.front_points.items()}
    t.rear_points = {tid: align_R @ p + align_t for tid, p in t.rear_points.items()}
  return world_T_anchor[anchor_tag_ids[0]]


@dataclass(frozen=True)
class _CrossSighting:
  """One observer-lens sighting of another tracker's tag, for a cross factor."""

  observer: str
  owner: str
  keyframe: int
  lens: LensSetup
  corners: Float64Array  # (4, 2) detected corner pixels
  bearing: Float64Array  # (3,) lens-frame tag direction
  finger: int | None  # None = base tag; 0 = left finger, 1 = right finger


def _cross_tracker_observations(
  cameras: list[PreparedCamera],
  base_tag_ids: list[int],
  base_tag_size: float,
) -> list[_CrossSighting]:
  """Every other-tracker sighting of a PnP-passing base tag or a modeled finger tag."""
  sightings: list[_CrossSighting] = []
  for owner_idx, owner in enumerate(cameras):
    # Visible owner tags: the base tag and each finger tag with URDF joint kinematics.
    targets: dict[int, int | None] = {base_tag_ids[owner_idx]: None}
    for idx, name in ((0, "left_finger"), (1, "right_finger")):
      finger_id = owner.left_finger_tag_id if idx == 0 else owner.right_finger_tag_id
      if finger_id > 0 and name in owner.finger_reproj:
        targets[finger_id] = idx
    for observer in cameras:
      if observer is owner:
        continue
      n = min(observer.n_keyframes, owner.n_keyframes)
      for lens, tags_per_kf in (
        (observer.front_lens, observer.front_tags_per_kf),
        (observer.rear_lens, observer.rear_tags_per_kf),
      ):
        for kf in range(n):
          for det in tags_per_kf[kf]:
            if det.tag_id not in targets:
              continue
            finger = targets[det.tag_id]
            if finger is None and (
              estimate_lens_T_tag(det.corners, base_tag_size, lens.params) is None
            ):
              continue  # base-tag PnP gate: drop degenerate / behind-camera
            sightings.append(
              _CrossSighting(
                observer=observer.name,
                owner=owner.name,
                keyframe=kf,
                lens=lens,
                corners=np.asarray(det.corners, dtype=np.float64),
                bearing=_lens_bearing(det, lens),
                finger=finger,
              )
            )
  return sightings


def _finger_symmetry_factor(
  key_l: int, key_r: int, noise: gtsam.noiseModel.Base
) -> gtsam.CustomFactor:
  """Soft URDF mimic prior with residual θ_left + θ_right."""

  def error_func(
    _this: gtsam.CustomFactor, values: gtsam.Values, jac: list[Float64Array] | None
  ) -> Float64Array:
    tl = values.atDouble(key_l)
    tr = values.atDouble(key_r)
    if jac is not None:
      jac[0] = np.array([[1.0]])
      jac[1] = np.array([[1.0]])
    return np.array([tl + tr])

  return gtsam.CustomFactor(noise, [key_l, key_r], error_func)


def _resolve_urdf_path(explicit: Path | None) -> Path | None:
  """The gripper URDF: an explicit path, else the installed asset, else None."""
  if explicit is not None:
    return explicit
  try:
    from specialist_description import get_urdf_path

    return Path(get_urdf_path("specialist_handheld"))
  except ImportError as exc:  # headless env without the gripper-URDF package
    print(f"  gripper URDF unavailable ({exc}); finger joint factors disabled")
    return None


def _tag_corners_world(pose: gtsam.Pose3, tag_size: float) -> Float64Array:
  """A tag's four metric corners transformed into the world frame, ``(4, 3)``."""
  return np.array(
    [
      pose.transformFrom(gtsam.Point3(*corner))
      for corner in tag_corners_local(tag_size)
    ],
    dtype=np.float64,
  )


def _anchor_sightings(
  tracker: PreparedCamera, anchor_tag_id: int
) -> list[tuple[LensSetup, int, TagDetection]]:
  """Per-keyframe anchor-tag detections in both lenses."""
  out: list[tuple[LensSetup, int, TagDetection]] = []
  for lens, tags_per_kf in (
    (tracker.front_lens, tracker.front_tags_per_kf),
    (tracker.rear_lens, tracker.rear_tags_per_kf),
  ):
    sightings = [
      (kf, det)
      for kf, dets in enumerate(tags_per_kf)
      for det in dets
      if det.tag_id == anchor_tag_id
    ]
    out.extend((lens, kf, det) for kf, det in sightings)
  return out


def _sighting_bearing(
  obs: dict[int, dict[str, gtsam.Pose3]],
  kf: int,
  side: str,
  det: TagDetection,
  lens: LensSetup,
) -> Float64Array:
  """Cheirality bearing: the PnP tag position, else the detection-center ray."""
  lens_T_tag = obs.get(kf, {}).get(side)
  if lens_T_tag is None:
    return _lens_bearing(det, lens)
  return np.asarray(lens_T_tag.translation(), dtype=np.float64)


def _anchored_keyframes(tracker: PreparedCamera) -> set[int]:
  """Keyframes with a PnP sighting of any configured anchor."""
  return set().union(*tracker.anchor_obs_by_tag.values())


def _median_pose(poses: list[gtsam.Pose3]) -> gtsam.Pose3:
  """Median translation and median rotation log about the first pose."""
  ref_rot = poses[0].rotation()
  translations = np.stack(
    [np.asarray(p.translation(), dtype=np.float64) for p in poses]
  )
  logs = np.stack([gtsam.Rot3.Logmap(ref_rot.between(p.rotation())) for p in poses])
  delta = gtsam.Rot3.Expmap(np.median(logs, axis=0))
  return gtsam.Pose3(
    ref_rot.compose(delta), gtsam.Point3(*np.median(translations, axis=0))
  )


def _secondary_anchor_seed(
  cameras: list[PreparedCamera],
  tag_id: int,
  body_poses_by_tracker: dict[str, list[gtsam.Pose3]] | None = None,
) -> gtsam.Pose3 | None:
  """Median ``world_T_body(kf) ∘ body_T_lens ∘ lens_T_tag`` over all PnP sightings."""
  candidates: list[gtsam.Pose3] = []
  for t in cameras:
    poses = (
      t.body_poses if body_poses_by_tracker is None else body_poses_by_tracker[t.name]
    )
    obs = t.anchor_obs_by_tag.get(tag_id, {})
    for kf in sorted(obs):
      sides = obs[kf]
      for side, lens in (("front", t.front_lens), ("rear", t.rear_lens)):
        lens_T_tag = sides.get(side)
        if lens_T_tag is None:
          continue
        candidates.append(poses[kf].compose(lens.body_T_lens).compose(lens_T_tag))
  if not candidates:
    return None
  return _median_pose(candidates)


def _add_anchor_corner_factors(
  graph: gtsam.NonlinearFactorGraph,
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  corner_noise: gtsam.noiseModel.Base,
  tag_id: int,
  corners_world: Float64Array,
  pnp_bearings: bool,
) -> int:
  """Add one anchor's corner factors from every sighting; return the sighting count."""
  n_sightings = 0
  for t in cameras:
    obs = t.anchor_obs_by_tag.get(tag_id, {}) if pnp_bearings else {}
    for lens, kf, det in _anchor_sightings(t, tag_id):
      n_sightings += 1
      pose_key = keys.pose_key(t.name, kf)
      side = "front" if lens is t.front_lens else "rear"
      bearing = _sighting_bearing(obs, kf, side, det, lens)
      for corner_xyz, pixel in zip(corners_world, det.corners, strict=True):
        add_mei_tag_factor(
          graph,
          make_mei_world_point_factor_cpp,
          (pose_key,),
          pixel,
          corner_noise,
          lens,
          corner_xyz,
          bearing,
        )
  return n_sightings


def _factor_observation_lines(
  observer: PreparedCamera,
  observer_poses: list[gtsam.Pose3],
  cross_obs_for_observer: list[_CrossSighting],
  anchor_corners_by_tag: dict[int, Float64Array],
  base_tag_corners_world: dict[str, list[Float64Array]],
) -> tuple[list[Float64Array], list[Float64Array]]:
  """Per-keyframe ``(n, 2, 3)`` lens-to-corner lines of anchor and base-tag factors."""
  anchor_by_kf: dict[int, list[tuple[LensSetup, Float64Array]]] = {}
  for tag_id, corners_world in anchor_corners_by_tag.items():
    for lens, kf, _det in _anchor_sightings(observer, tag_id):
      anchor_by_kf.setdefault(kf, []).append((lens, corners_world))
  anchor_lines: list[Float64Array] = []
  cross_lines: list[Float64Array] = []
  for kf in range(observer.n_keyframes):
    body_pose = observer_poses[kf]
    anchor_strips: list[Float64Array] = []
    for lens, corners_world in anchor_by_kf.get(kf, ()):
      lens_xyz = np.asarray(
        body_pose.compose(lens.body_T_lens).translation(), dtype=np.float64
      )
      for corner_xyz in corners_world:
        anchor_strips.append(np.stack([lens_xyz, corner_xyz]))
    anchor_lines.append(
      np.stack(anchor_strips) if anchor_strips else np.empty((0, 2, 3))
    )
    cross_strips: list[Float64Array] = []
    for sighting in cross_obs_for_observer:
      if sighting.finger is not None or sighting.keyframe != kf:
        continue  # base-tag rays only; owner_corners holds the base-tag corners
      owner_corners = base_tag_corners_world.get(sighting.owner)
      if owner_corners is None or kf >= len(owner_corners):
        continue
      lens_xyz = np.asarray(
        body_pose.compose(sighting.lens.body_T_lens).translation(), dtype=np.float64
      )
      for corner_xyz in owner_corners[kf]:
        cross_strips.append(np.stack([lens_xyz, corner_xyz]))
    cross_lines.append(np.stack(cross_strips) if cross_strips else np.empty((0, 2, 3)))
  return anchor_lines, cross_lines


def _per_kf_rejected_keypoints(
  rejected_tracks: dict[int, dict[int, Float64Array]], n_keyframes: int
) -> list[Float64Array]:
  buckets: list[list[Float64Array]] = [[] for _ in range(n_keyframes)]
  for obs in rejected_tracks.values():
    for kf, pix in obs.items():
      if 0 <= kf < n_keyframes:
        buckets[kf].append(np.asarray(pix, dtype=np.float64))
  return [np.stack(b) if b else np.empty((0, 2)) for b in buckets]


def _reproject_landmarks_per_kf(
  tracks: dict[int, dict[int, Float64Array]],
  points: dict[int, Float64Array],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  n_keyframes: int,
) -> list[Float64Array]:
  buckets: list[list[Float64Array]] = [[] for _ in range(n_keyframes)]
  for tid, obs in tracks.items():
    if tid not in points:
      continue
    lm_world = gtsam.Point3(*points[tid])
    for kf in obs:
      if not (0 <= kf < n_keyframes):
        continue
      world_T_lens = body_poses[kf].compose(lens.body_T_lens)
      p_lens = np.asarray(world_T_lens.transformTo(lm_world), dtype=np.float64)
      if p_lens[2] <= 1e-3:
        continue
      pix = mei_project(p_lens[None, :], lens.params)[0]
      if np.all(np.isfinite(pix)):
        buckets[kf].append(pix)
  return [np.stack(b) if b else np.empty((0, 2)) for b in buckets]


def _project_tag_axes_per_kf(
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  tag_size_by_id: dict[int, float],
  n_keyframes: int,
) -> list[Float64Array]:
  out: list[Float64Array] = []
  for kf in range(n_keyframes):
    tags = tags_per_kf[kf] if kf < len(tags_per_kf) else []
    strips: list[Float64Array] = []
    for det in tags:
      size = tag_size_by_id.get(int(det.tag_id))
      if size is None:
        continue
      pose = estimate_lens_T_tag(det.corners, size, lens.params)
      if pose is None:
        continue
      tips = np.array(
        [
          [0.0, 0.0, 0.0],
          [size, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, size, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, size],
        ],
        dtype=np.float64,
      )
      rmat = pose.rotation().matrix()
      tvec = np.asarray(pose.translation(), dtype=np.float64)
      lens_tips = (rmat @ tips.T).T + tvec
      if np.any(lens_tips[:, 2] <= 1e-3):
        continue
      pix = mei_project(lens_tips, lens.params)
      if not np.all(np.isfinite(pix)):
        continue
      strips.append(pix.reshape(3, 2, 2))
    out.append(np.concatenate(strips) if strips else np.empty((0, 2, 2)))
  return out


def _rms(values: list[float]) -> float:
  """Root-mean-square of ``values``; NaN when empty."""
  return float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")


def _per_kf_reproj_rmse_for_viz(
  values: gtsam.Values,
  tracks: dict[int, dict[int, Float64Array]],
  lens: LensSetup,
  landmark_key_fn: Callable[[int], int],
  pose_key_fn: Callable[[int], int],
  n_keyframes: int,
) -> list[float]:
  """Per-keyframe reprojection RMSE for the Rerun overlay and the stdout print only."""
  buckets: list[list[float]] = [[] for _ in range(n_keyframes)]
  for tid, obs in tracks.items():
    lm_key = landmark_key_fn(tid)
    if not values.exists(lm_key):
      continue
    xyz = values.atPoint3(lm_key)
    for kf, pt in obs.items():
      pk = pose_key_fn(kf)
      if not (0 <= kf < n_keyframes) or not values.exists(pk):
        continue
      body_pose = values.atPose3(pk)
      p_lens = body_pose.compose(lens.body_T_lens).transformTo(
        gtsam.Point3(float(xyz[0]), float(xyz[1]), float(xyz[2]))
      )
      pred = mei_project(np.asarray(p_lens)[None, :], lens.params)[0]
      if np.all(np.isfinite(pred)):
        buckets[kf].append(float(np.linalg.norm(pred - pt)))
  return [_rms(errs) for errs in buckets]


def _aggregate_rmse_for_print(per_kf: list[float]) -> float:
  return _rms([v for v in per_kf if np.isfinite(v)])


def _pose_to_dict(p: gtsam.Pose3) -> dict:
  return {
    "R": [list(map(float, row)) for row in p.rotation().matrix()],
    "t": [float(x) for x in p.translation()],
  }


def _mei_to_dict(p: MeiParams) -> dict:
  return {k: float(v) for k, v in asdict(p).items()}


def _bounding_box_diagonal(poses: list[gtsam.Pose3]) -> float:
  if len(poses) < 2:
    return 0.0
  pts = np.array([np.asarray(p.translation(), dtype=np.float64) for p in poses])
  return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def _path_length(poses: list[gtsam.Pose3]) -> float:
  pts = [np.asarray(p.translation(), dtype=np.float64) for p in poses]
  return float(sum(np.linalg.norm(pts[i] - pts[i - 1]) for i in range(1, len(pts))))


def _stage_timer() -> Callable[[str], None]:
  last = time.perf_counter()

  def mark(stage: str) -> None:
    nonlocal last
    now = time.perf_counter()
    print(f"  [{stage}] +{now - last:.1f}s")
    last = now

  return mark


def _assemble_joint_graph(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  args: Args,
  cross_obs: list[_CrossSighting],
  anchor_corners_world: Float64Array,
  secondary_anchor_world: dict[int, gtsam.Pose3],
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """Build the joint factor graph over every tracker and anchor."""
  graph = gtsam.NonlinearFactorGraph()
  values = gtsam.Values()

  for t in cameras:
    build_vi_graph(
      **t.vi_graph_inputs(),
      sigma=args.frontend.reproj_sigma,
      velocity_prior_sigma=args.solver.velocity_prior_sigma,
      graph=graph,
      values=values,
      **keys.vi_graph_kwargs(t.name),
    )

  corner_noise = huber_pixel_noise(args.anchor.corner_reproj_sigma)
  n_ref = _add_anchor_corner_factors(
    graph,
    cameras,
    keys,
    corner_noise,
    args.anchor.tag_ids[0],
    anchor_corners_world,
    pnp_bearings=True,
  )
  print(f"anchor corner reprojection: {n_ref} anchor sightings")
  for tag_id, world_T_tag in secondary_anchor_world.items():
    n_sec = _add_anchor_corner_factors(
      graph,
      cameras,
      keys,
      corner_noise,
      tag_id,
      _tag_corners_world(world_T_tag, args.anchor.tag_size),
      pnp_bearings=False,
    )
    print(f"secondary anchor {tag_id} corner reprojection: {n_sec} sightings")

  # Base and finger tags pin one static body_T_base_tag; each finger adds a joint angle.
  finger_noise = huber_pixel_noise(args.gripper.finger_reproj_sigma)
  base_tag_noise = huber_pixel_noise(args.gripper.base_tag_reproj_sigma)
  base_tag_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.array(
      [args.gripper.base_tag_prior_sigma_rot] * 3
      + [args.gripper.base_tag_prior_sigma_trans] * 3
    )
  )
  sym_noise = gtsam.noiseModel.Isotropic.Sigma(1, args.gripper.finger_symmetry_sigma)
  base_tag_corners = tag_corners_local(args.gripper.tag_size)
  base_tag_id_by_name = {
    t.name: tid for t, tid in zip(cameras, args.base_tag_ids, strict=True)
  }
  finger_factor_count = 0
  base_tag_factor_count = 0
  base_tag_inserted: set[str] = set()
  theta_inserted: set[tuple[str, int, int]] = set()
  for t in cameras:
    reprojs = {
      0: t.finger_reproj.get("left_finger"),
      1: t.finger_reproj.get("right_finger"),
    }
    has_fingers = any(r is not None for r in reprojs.values())
    base_tag_id = base_tag_id_by_name[t.name]
    if t.body_T_base_tag is None or (not has_fingers and base_tag_id <= 0):
      continue
    bt_key = keys.base_tag_key(t.name)
    values.insert(bt_key, t.body_T_base_tag)
    base_tag_inserted.add(t.name)
    graph.add(gtsam.PriorFactorPose3(bt_key, t.body_T_base_tag, base_tag_prior_noise))
    # Base-tag corners reproject through body_T_base_tag into this tracker's rear lens.
    if base_tag_id > 0:
      for dets in t.rear_tags_per_kf:
        det = next((d for d in dets if d.tag_id == base_tag_id), None)
        if det is None:
          continue
        bearing = _lens_bearing(det, t.rear_lens)
        for corner, pixel in zip(base_tag_corners, det.corners, strict=True):
          add_mei_tag_factor(
            graph,
            make_mei_base_tag_factor_cpp,
            (bt_key,),
            pixel,
            base_tag_noise,
            t.rear_lens,
            corner,
            bearing,
          )
        base_tag_factor_count += 4
    both_fingers = reprojs[0] is not None and reprojs[1] is not None
    observed_kf = sorted(
      {kf for r in reprojs.values() if r is not None for kf in r.observations}
    )
    for kf in observed_kf:
      seen_seed: dict[int, float] = {}
      for finger_idx, reproj in reprojs.items():
        obs = None if reproj is None else reproj.observations.get(kf)
        if reproj is None or obs is None:
          continue
        tkey = keys.theta_key(t.name, finger_idx, kf)
        values.insert(tkey, float(obs.theta_seed))
        theta_inserted.add((t.name, finger_idx, kf))
        for corner, pixel in zip(reproj.joint_t_corners, obs.pixels, strict=True):
          add_mei_tag_factor(
            graph,
            make_mei_finger_factor_cpp,
            (tkey, bt_key),
            pixel,
            finger_noise,
            t.rear_lens,
            corner,
            obs.obs_bearing,
            reproj.base_T_joint,
          )
        finger_factor_count += 4
        seen_seed[finger_idx] = float(obs.theta_seed)
      if both_fingers:
        for finger_idx in (0, 1):
          if finger_idx not in seen_seed:  # unseen finger: inferred from the prior only
            values.insert(
              keys.theta_key(t.name, finger_idx, kf), -next(iter(seen_seed.values()))
            )
            theta_inserted.add((t.name, finger_idx, kf))
        graph.add(
          _finger_symmetry_factor(
            keys.theta_key(t.name, 0, kf), keys.theta_key(t.name, 1, kf), sym_noise
          )
        )

  # Cross-tracker corner reprojection through observer, owner and owner base-tag poses.
  camera_by_slot = {t.name: t for t in cameras}
  base_count = finger_count = 0
  for sighting in cross_obs:
    if sighting.owner not in base_tag_inserted:
      continue
    obs_key = keys.pose_key(sighting.observer, sighting.keyframe)
    owner_key = keys.pose_key(sighting.owner, sighting.keyframe)
    owner_bt_key = keys.base_tag_key(sighting.owner)
    if sighting.finger is None:
      for corner, pixel in zip(base_tag_corners, sighting.corners, strict=True):
        add_mei_tag_factor(
          graph,
          make_mei_cross_tag_factor_cpp,
          (obs_key, owner_key, owner_bt_key),
          pixel,
          base_tag_noise,
          sighting.lens,
          corner,
          sighting.bearing,
        )
      base_count += 4
    elif (sighting.owner, sighting.finger, sighting.keyframe) in theta_inserted:
      finger_name = "left_finger" if sighting.finger == 0 else "right_finger"
      reproj = camera_by_slot[sighting.owner].finger_reproj[finger_name]
      theta_key = keys.theta_key(sighting.owner, sighting.finger, sighting.keyframe)
      for corner, pixel in zip(reproj.joint_t_corners, sighting.corners, strict=True):
        add_mei_tag_factor(
          graph,
          make_mei_cross_finger_factor_cpp,
          (obs_key, owner_key, owner_bt_key, theta_key),
          pixel,
          base_tag_noise,
          sighting.lens,
          corner,
          sighting.bearing,
          reproj.base_T_joint,
        )
      finger_count += 4

  # Pose and velocity priors on each unanchored consecutive keyframe pair cap IMU drift.
  smooth_sigmas = np.array([np.deg2rad(2.0)] * 3 + [0.10] * 3, dtype=np.float64)
  smooth_noise = gtsam.noiseModel.Diagonal.Sigmas(smooth_sigmas)
  vel_smooth_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.3, 0.3, 0.3], dtype=np.float64)
  )
  smooth_count = 0
  vel_smooth_count = 0
  zero_t = gtsam.Point3(0.0, 0.0, 0.0)
  for t in cameras:
    anchored_kfs = _anchored_keyframes(t)
    for k in range(1, t.n_keyframes):
      # Both endpoints must be unanchored; an anchored k-1 fights k's visual factors.
      if (k - 1) in anchored_kfs or k in anchored_kfs:
        continue
      # The mean is the prep relative rotation; identity penalises handheld rotation.
      prep_rel = t.body_poses[k - 1].inverse().compose(t.body_poses[k])
      smooth_mean = gtsam.Pose3(prep_rel.rotation(), zero_t)
      graph.add(
        gtsam.BetweenFactorPose3(
          keys.pose_key(t.name, k - 1),
          keys.pose_key(t.name, k),
          smooth_mean,
          smooth_noise,
        )
      )
      smooth_count += 1
      vel_delta = np.asarray(t.velocities[k] - t.velocities[k - 1], dtype=np.float64)
      graph.add(
        gtsam.BetweenFactorVector(
          keys.velocity_key(t.name, k - 1),
          keys.velocity_key(t.name, k),
          vel_delta,
          vel_smooth_noise,
        )
      )
      vel_smooth_count += 1
  print(
    f"reprojection factors: base-tag {base_tag_factor_count}, "
    f"finger {finger_factor_count}, cross base {base_count}, cross finger "
    f"{finger_count}; smoothness priors: pose {smooth_count}, "
    f"velocity {vel_smooth_count}"
  )
  return graph, values


def _points_from_values(
  values: gtsam.Values,
  key_fn: Callable[[str, int], int],
  name: str,
  tids: Iterable[int],
) -> dict[int, Float64Array]:
  """Pull the solved 3-D landmark points out of ``values`` for ``tids``."""
  return {
    tid: np.asarray(values.atPoint3(key_fn(name, tid)), dtype=np.float64)
    for tid in tids
  }


# The first solve only ranks residuals for the cull.
_PRECULL_MAX_ITERATIONS = 8


def _solve_with_cull_reoptimize(
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  max_iterations: int,
  gpu: bool,
) -> tuple[OptimizeResult, dict[str, tuple[dict[int, dict[int, Float64Array]], ...]]]:
  """Re-solved result and each tracker's kept tracks after an outlier-landmark cull."""
  initial_error = float(graph.error(values))
  print(
    f"\njoint graph: {graph.size()} factors, {values.size()} variables, "
    f"initial error={initial_error:.1f}"
  )
  result = optimize_graph(
    graph,
    values,
    max_iterations=min(max_iterations, _PRECULL_MAX_ITERATIONS),
    gpu=gpu,
  )
  print(
    f"  solve 1: error {result.initial_error:.1f} → {result.final_error:.1f} "
    f"({result.iterations} iters)"
  )

  landmark_keep: dict[str, tuple[dict[int, dict[int, Float64Array]], ...]] = {}
  outlier_keys: set[int] = set()
  for t in cameras:
    poses_1 = [
      result.values.atPose3(keys.pose_key(t.name, i)) for i in range(t.n_keyframes)
    ]
    f_pts_1 = _points_from_values(
      result.values, keys.front_landmark_key, t.name, t.front_points
    )
    r_pts_1 = _points_from_values(
      result.values, keys.rear_landmark_key, t.name, t.rear_points
    )
    front_keep = cull_outlier_landmarks(
      f_pts_1, t.front_tracks_tri, poses_1, t.front_lens
    )
    rear_keep = cull_outlier_landmarks(r_pts_1, t.rear_tracks_tri, poses_1, t.rear_lens)
    landmark_keep[t.name] = (front_keep, rear_keep)
    outlier_keys.update(
      keys.front_landmark_key(t.name, tid)
      for tid in t.front_points
      if tid not in front_keep
    )
    outlier_keys.update(
      keys.rear_landmark_key(t.name, tid)
      for tid in t.rear_points
      if tid not in rear_keep
    )
  first = result
  if not outlier_keys and result.iterations >= _PRECULL_MAX_ITERATIONS:
    # Nothing to cull; finish the short ranking solve on the full budget.
    result = optimize_graph(
      graph, result.values, max_iterations=max_iterations, gpu=gpu
    )
  if outlier_keys:
    pruned = gtsam.NonlinearFactorGraph()
    for i in range(graph.size()):
      factor = graph.at(i)
      if not any(k in outlier_keys for k in factor.keys()):
        pruned.push_back(factor)
    kept_values = gtsam.Values(result.values)
    for key in outlier_keys:
      kept_values.erase(key)
    result = optimize_graph(pruned, kept_values, max_iterations=max_iterations, gpu=gpu)
    result = replace(
      result,
      initial_error=first.initial_error,
      iterations=first.iterations + result.iterations,
    )
    print(
      f"  cull-reoptimize: dropped {len(outlier_keys)} outlier landmarks, "
      f"error → {result.final_error:.1f} ({result.iterations} iters)"
    )
  return result, landmark_keep


def _seed_secondary_anchors(
  cameras: list[PreparedCamera], args: Args
) -> dict[int, gtsam.Pose3]:
  """Initial world pose per sighted secondary anchor, from BFS-aligned prep poses."""
  world: dict[int, gtsam.Pose3] = {}
  for tag_id in args.anchor.tag_ids[1:]:
    seed = _secondary_anchor_seed(cameras, tag_id)
    if seed is None:
      print(f"secondary anchor {tag_id}: no PnP sighting; skipped")
      continue
    world[tag_id] = seed
  return world


def _solve_joint_graph(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  args: Args,
  cross_obs: list[_CrossSighting],
  anchor_corners_world: Float64Array,
) -> tuple[
  OptimizeResult,
  dict[str, tuple[dict[int, dict[int, Float64Array]], ...]],
  dict[int, gtsam.Pose3],
  dict[str, list[gtsam.Pose3]],
]:
  """Solve with secondary anchors pinned at their seeds; return their re-estimates."""
  secondary_world = _seed_secondary_anchors(cameras, args)
  graph, values = _assemble_joint_graph(
    cameras, keys, args, cross_obs, anchor_corners_world, secondary_world
  )
  result, landmark_keep = _solve_with_cull_reoptimize(
    graph, values, cameras, keys, args.solver.max_iterations, args.solver.gpu
  )
  solved_poses = {
    t.name: [
      result.values.atPose3(keys.pose_key(t.name, i)) for i in range(t.n_keyframes)
    ]
    for t in cameras
  }
  for tag_id in secondary_world:
    estimate = _secondary_anchor_seed(
      cameras, tag_id, body_poses_by_tracker=solved_poses
    )
    assert estimate is not None  # seeding required a sighting; obs never shrink
    pinned = secondary_world[tag_id]
    dt = float(
      np.linalg.norm(
        np.asarray(estimate.translation()) - np.asarray(pinned.translation())
      )
    )
    dr = float(
      np.linalg.norm(gtsam.Rot3.Logmap(pinned.rotation().between(estimate.rotation())))
    )
    note = (
      "  WARNING: weak sighting geometry this take"
      if (dt > _ANCHOR_CONSISTENCY_TRANS or dr > _ANCHOR_CONSISTENCY_ROT)
      else ""
    )
    print(
      f"  secondary anchor {tag_id}: post-solve re-estimate moved "
      f"{dt * 1000:.1f}mm / {np.degrees(dr):.2f}deg{note}"
    )
    secondary_world[tag_id] = estimate
  return result, landmark_keep, secondary_world, solved_poses


def _lens_side_dict(lens: LensSetup, decoded_width: int) -> dict[str, object]:
  """One lens's body_T_lens, intrinsics, and decoded frame shape for the export."""
  return {
    "body_T_lens": _pose_to_dict(lens.body_T_lens),
    "params": _mei_to_dict(lens.params),
    "frame_shape": [decoded_width, decoded_width],
  }


def _tags_side_list(
  tags_per_kf: list[list[TagDetection]],
) -> list[list[dict[str, object]]]:
  """One lens's per-keyframe AprilTag detections (id + corners) for the export."""
  return [
    [
      {
        "tag_id": int(d.tag_id),
        "corners": [[float(p[0]), float(p[1])] for p in d.corners],
      }
      for d in tags
    ]
    for tags in tags_per_kf
  ]


def _tracks_side_dict(
  tracks: dict[int, dict[int, Float64Array]],
) -> dict[str, dict[str, list[float]]]:
  """One lens's ``{track_id: {keyframe: pixel}}`` observations for the export."""
  return {
    str(tid): {str(kf): [float(p[0]), float(p[1])] for kf, p in obs.items()}
    for tid, obs in tracks.items()
  }


def _build_raw_export(
  cameras: list[PreparedCamera],
  optimized_body_poses: dict[str, list[gtsam.Pose3]],
  culled_points: dict[str, tuple[dict[int, Float64Array], dict[int, Float64Array]]],
  world_T_anchor: dict[int, gtsam.Pose3],
  audio_offset_by_name: dict[str, float],
  args: Args,
) -> dict[str, object]:
  """Raw observations for downstream evaluation, free of derived metrics."""
  raw: dict[str, object] = {}
  raw["body_poses_by_tracker"] = {
    t.name: [_pose_to_dict(p) for p in optimized_body_poses[t.name]] for t in cameras
  }
  raw["ref_anchor_pose_world"] = _pose_to_dict(
    world_T_anchor[int(args.anchor.tag_ids[0])]
  )
  raw["anchor_poses_world"] = {
    str(tag_id): _pose_to_dict(p) for tag_id, p in world_T_anchor.items()
  }
  raw["lens_by_tracker"] = {
    t.name: {
      "front": _lens_side_dict(t.front_lens, t.decoded_width),
      "rear": _lens_side_dict(t.rear_lens, t.decoded_width),
    }
    for t in cameras
  }
  raw["anchor_obs_by_tracker"] = {
    t.name: [
      {"tag_id": tag_id, "kf": int(kf), "side": side, "lens_T_tag": _pose_to_dict(p)}
      for tag_id, obs in t.anchor_obs_by_tag.items()
      for kf, sides in obs.items()
      for side, p in sides.items()
    ]
    for t in cameras
  }
  raw["tag_detections_by_tracker"] = {
    t.name: {
      "front": _tags_side_list(t.front_tags_per_kf),
      "rear": _tags_side_list(t.rear_tags_per_kf),
    }
    for t in cameras
  }
  raw["tracks_by_tracker"] = {
    t.name: {
      "front": _tracks_side_dict(t.front_tracks_tri),
      "rear": _tracks_side_dict(t.rear_tracks_tri),
    }
    for t in cameras
  }
  raw["landmarks_by_tracker"] = {
    t.name: {
      "front": {
        str(tid): [float(x) for x in p] for tid, p in culled_points[t.name][0].items()
      },
      "rear": {
        str(tid): [float(x) for x in p] for tid, p in culled_points[t.name][1].items()
      },
    }
    for t in cameras
  }
  raw["kf_times_by_tracker"] = {
    t.name: [float(x) for x in t.keyframe_times] for t in cameras
  }
  raw["audio_offset_by_tracker"] = audio_offset_by_name
  raw["body_T_base_tag_by_tracker"] = {
    t.name: (_pose_to_dict(t.body_T_base_tag) if t.body_T_base_tag else None)
    for t in cameras
  }
  raw["joint_angles_by_tracker"] = {
    t.name: t.joint_angles for t in cameras if t.joint_angles
  }
  raw["anchor_tag_id"] = int(args.anchor.tag_ids[0])
  raw["anchor_tag_size"] = float(args.anchor.tag_size)
  raw["gripper_keypoint_offset"] = [float(x) for x in args.gripper.tcp_in_base_tag]
  return raw


def _build_tracker_viz_data(
  t: PreparedCamera,
  poses: list[gtsam.Pose3],
  result: OptimizeResult,
  keys: CameraKeys,
  landmark_keep_t: tuple[
    dict[int, dict[int, Float64Array]], dict[int, dict[int, Float64Array]]
  ],
  args: Args,
  tag_size_by_id: dict[int, float],
  cross_obs_for_t: list[_CrossSighting],
  anchor_corners_by_tag: dict[int, Float64Array],
  base_tag_corners_world: dict[str, list[Float64Array]],
  audio_offset: float,
) -> tuple[
  CameraVizData,
  dict[str, object],
  tuple[dict[int, Float64Array], dict[int, Float64Array]],
]:
  """A tracker's ``CameraVizData``, summary and culled ``(front, rear)`` landmarks."""
  f_pts = _points_from_values(
    result.values, keys.front_landmark_key, t.name, landmark_keep_t[0]
  )
  r_pts = _points_from_values(
    result.values, keys.rear_landmark_key, t.name, landmark_keep_t[1]
  )
  # Drop landmarks and observations the solved geometry rejects; poses stay unchanged.
  front_obs = cull_outlier_landmarks(f_pts, t.front_tracks_tri, poses, t.front_lens)
  rear_obs = cull_outlier_landmarks(r_pts, t.rear_tracks_tri, poses, t.rear_lens)
  f_pts = {tid: f_pts[tid] for tid in front_obs}
  r_pts = {tid: r_pts[tid] for tid in rear_obs}
  front_rmse_per_kf = _per_kf_reproj_rmse_for_viz(
    result.values,
    front_obs,
    t.front_lens,
    lambda tid, n=t.name: keys.front_landmark_key(n, tid),
    lambda i, n=t.name: keys.pose_key(n, i),
    t.n_keyframes,
  )
  rear_rmse_per_kf = _per_kf_reproj_rmse_for_viz(
    result.values,
    rear_obs,
    t.rear_lens,
    lambda tid, n=t.name: keys.rear_landmark_key(n, tid),
    lambda i, n=t.name: keys.pose_key(n, i),
    t.n_keyframes,
  )
  f_rmse = _aggregate_rmse_for_print(front_rmse_per_kf)
  r_rmse = _aggregate_rmse_for_print(rear_rmse_per_kf)
  bbox_diagonal = _bounding_box_diagonal(poses)
  per_tracker_result: dict[str, object] = {
    "name": t.name,
    "n_keyframes": t.n_keyframes,
    "anchor_obs": len(_anchored_keyframes(t)),
    "bounding_box_diagonal": bbox_diagonal,
    "path_length": _path_length(poses),
  }
  print(
    f"  {t.name}: kf={t.n_keyframes} bbox={bbox_diagonal:.3f} m "
    f"front_rmse={f_rmse:.2f}px rear_rmse={r_rmse:.2f}px"
  )
  if t.body_T_base_tag is not None:
    base_tag_poses = [p.compose(t.body_T_base_tag) for p in poses]
  else:
    base_tag_poses = poses
  biases_final = [
    result.values.atConstantBias(keys.bias_key(t.name, i)) for i in range(t.n_keyframes)
  ]
  front_tag_axes = _project_tag_axes_per_kf(
    t.front_tags_per_kf,
    t.front_lens,
    tag_size_by_id,
    t.n_keyframes,
  )
  rear_tag_axes = _project_tag_axes_per_kf(
    t.rear_tags_per_kf,
    t.rear_lens,
    tag_size_by_id,
    t.n_keyframes,
  )
  front_reproj = (
    _reproject_landmarks_per_kf(front_obs, f_pts, poses, t.front_lens, t.n_keyframes)
    if args.log_frames_in_rrd
    else None
  )
  rear_reproj = (
    _reproject_landmarks_per_kf(rear_obs, r_pts, poses, t.rear_lens, t.n_keyframes)
    if args.log_frames_in_rrd
    else None
  )
  front_rejected_kp = (
    _per_kf_rejected_keypoints(t.front_rejected, t.n_keyframes)
    if args.log_frames_in_rrd
    else None
  )
  rear_rejected_kp = (
    _per_kf_rejected_keypoints(t.rear_rejected, t.n_keyframes)
    if args.log_frames_in_rrd
    else None
  )
  anchor_lines, cross_lines = _factor_observation_lines(
    t,
    poses,
    cross_obs_for_t,
    anchor_corners_by_tag,
    base_tag_corners_world,
  )
  viz_data = CameraVizData(
    name=t.name,
    body_poses=poses,
    keyframe_times=np.asarray(t.keyframe_times, dtype=np.float64) - audio_offset,
    front_points=f_pts,
    rear_points=r_pts,
    front_frames=t.front_frames if args.log_frames_in_rrd else None,
    rear_frames=t.rear_frames if args.log_frames_in_rrd else None,
    front_keypoints=t.front_keypoints if args.log_frames_in_rrd else None,
    rear_keypoints=t.rear_keypoints if args.log_frames_in_rrd else None,
    front_tag_detections=t.front_tags_per_kf,
    rear_tag_detections=t.rear_tags_per_kf,
    velocities=[
      np.asarray(result.values.atVector(keys.velocity_key(t.name, i)))
      for i in range(t.n_keyframes)
    ],
    gyro_biases=[np.asarray(b.gyroscope(), dtype=np.float64) for b in biases_final],
    accel_biases=[
      np.asarray(b.accelerometer(), dtype=np.float64) for b in biases_final
    ],
    front_rmse_per_kf=front_rmse_per_kf,
    rear_rmse_per_kf=rear_rmse_per_kf,
    front_tag_axes=front_tag_axes,
    rear_tag_axes=rear_tag_axes,
    front_landmark_reprojections=front_reproj,
    rear_landmark_reprojections=rear_reproj,
    front_rejected_keypoints=front_rejected_kp,
    rear_rejected_keypoints=rear_rejected_kp,
    rear_static_mask_rects=t.rear_static_mask_rects,
    anchor_obs_lines=anchor_lines,
    cross_tracker_lines=cross_lines,
    urdf_root_poses=base_tag_poses,
    joint_angles=t.joint_angles or None,
    tcp_in_base_tag=args.gripper.tcp_in_base_tag,
    front_lens_extrinsic=t.front_lens.body_T_lens,
    rear_lens_extrinsic=t.rear_lens.body_T_lens,
  )
  return viz_data, per_tracker_result, (f_pts, r_pts)


def _adopt_ba_refined_estimates(
  cameras: list[PreparedCamera], result: OptimizeResult, keys: CameraKeys
) -> None:
  """Store each camera's solved base-tag pose and joint angles."""
  for t in cameras:
    for finger_idx, joint_name in ((0, "left_finger"), (1, "right_finger")):
      angles = [float("nan")] * t.n_keyframes
      for kf in range(t.n_keyframes):
        tkey = keys.theta_key(t.name, finger_idx, kf)
        if result.values.exists(tkey):
          angles[kf] = float(result.values.atDouble(tkey))
      t.joint_angles[joint_name] = angles
    bt_key = keys.base_tag_key(t.name)
    if result.values.exists(bt_key):
      t.body_T_base_tag = result.values.atPose3(bt_key)


def _print_capture_diagnostics(
  cameras: list[PreparedCamera],
  optimized_body_poses: dict[str, list[gtsam.Pose3]],
) -> None:
  """Print the maximum keyframe-to-keyframe speed."""
  print("\n=== max kf-to-kf speed (real motion if IMU agrees) ===")
  for name, poses in optimized_body_poses.items():
    pts = np.array([np.asarray(p.translation(), dtype=np.float64) for p in poses])
    t_obj = next(t for t in cameras if t.name == name)
    kf_t = np.asarray(t_obj.keyframe_times, dtype=np.float64)
    dt = np.diff(kf_t)
    speed = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.where(dt > 0, dt, 1.0)
    i = int(np.argmax(speed))
    print(f"  [{name}] max={speed[i]:.2f} m/s at kf {i}→{i + 1} (t={kf_t[i]:.2f}s)")


def main(args: Args) -> None:
  if not args.input_insvs:
    raise ValueError("at least one .insv input required")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  names = _derive_names(args.input_insvs, args.slots)
  if not args.base_tag_ids:
    raise ValueError(
      f"--base-tag-ids must be supplied (one per input, got {len(args.input_insvs)} inputs)"
    )
  for field_name, ids in (
    ("base_tag_ids", args.base_tag_ids),
    ("left_finger_tag_ids", args.left_finger_tag_ids),
    ("right_finger_tag_ids", args.right_finger_tag_ids),
  ):
    if ids and len(ids) != len(args.input_insvs):
      raise ValueError(
        f"{field_name} length ({len(ids)}) must match "
        f"input_insvs ({len(args.input_insvs)})"
      )
  if len(set(args.base_tag_ids)) != len(args.base_tag_ids):
    raise ValueError(
      f"--base-tag-ids must be distinct per tracker for cross-tracker "
      f"disambiguation, got {list(args.base_tag_ids)}"
    )
  if not args.anchor.tag_ids:
    raise ValueError("--anchor.tag-ids must name at least the reference anchor id")
  if len(set(args.anchor.tag_ids)) != len(args.anchor.tag_ids):
    raise ValueError(
      f"--anchor.tag-ids must be distinct, got {list(args.anchor.tag_ids)}"
    )
  _validate_tag_id_disjointness(args)
  mark = _stage_timer()

  starts, offsets = _audio_align_starts(args.input_insvs, args.window.master_start)
  mark(f"audio sync ({len(args.input_insvs)} clips)")

  # Cap the window at the overlap every capture covers.
  overlap = min(
    capture_duration(path) - start
    for path, start in zip(args.input_insvs, starts, strict=True)
  )
  duration = (
    overlap if args.window.duration is None else min(args.window.duration, overlap)
  )
  print(f"window: {duration:.2f}s (capture overlap {overlap:.2f}s)")

  left_finger_ids = args.left_finger_tag_ids or (0,) * len(args.input_insvs)
  right_finger_ids = args.right_finger_tag_ids or (0,) * len(args.input_insvs)
  specs = list(
    zip(
      names,
      args.input_insvs,
      starts,
      args.base_tag_ids,
      left_finger_ids,
      right_finger_ids,
      strict=True,
    )
  )

  # Resolve the URDF once: the finger joint-angle factors and the viz overlay share it.
  urdf_path = _resolve_urdf_path(args.urdf_path)

  def _prep_one(spec: tuple[str, Path, float, int, int, int]) -> PreparedCamera:
    name, path, start, base_tag_id, left_id, right_id = spec
    print(f"\n=== prep tracker {name!r} ({path.name}, start={start:.3f}s) ===")
    tracker = prepare_camera(
      name,
      CameraArgs(
        insv_path=path,
        start=start,
        duration=duration,
        keyframe_every=args.window.keyframe_every,
        keep_frames=args.log_frames_in_rrd,
        base_tag_id=base_tag_id,
        left_finger_tag_id=left_id,
        right_finger_tag_id=right_id,
        frontend=args.frontend,
        imu=args.imu,
        anchor=args.anchor,
        gripper=args.gripper,
        urdf_path=urdf_path,
      ),
    )
    print(
      f"  kf={tracker.n_keyframes}  anchor_obs={len(_anchored_keyframes(tracker))}  "
      f"front_pts={len(tracker.front_points)}  rear_pts={len(tracker.rear_points)}  "
      f"body_T_base_tag={'yes' if tracker.body_T_base_tag else 'no'}"
    )
    return tracker

  # The flow precompute shrinks its per-lens pool to fit the shared cv2 thread pool.
  set_precompute_concurrency(len(specs))
  with ThreadPoolExecutor(max_workers=len(specs)) as prep_pool:
    cameras = list(prep_pool.map(_prep_one, specs))
  # Free the GPU flow workers' device memory for the joint bundle adjustment.
  gpu_flow.shutdown()
  mark("cameras prepared")

  ref_anchor = _align_to_shared_world(cameras, args.anchor.tag_ids)
  print(f"\nshared world: anchor t={np.asarray(ref_anchor.translation())}")

  cross_obs = _cross_tracker_observations(
    cameras, list(args.base_tag_ids), args.gripper.tag_size
  )
  print(f"cross-tracker observations: {len(cross_obs)}")
  if cross_obs:
    by_pair = Counter((sighting.observer, sighting.owner) for sighting in cross_obs)
    for (obs_t, own_t), n in sorted(by_pair.items()):
      print(f"  {obs_t} → {own_t}: {n}")
  mark("cross-tracker obs assembled")

  keys = CameraKeys(list(names))
  anchor_tag_id = int(args.anchor.tag_ids[0])
  anchor_corners_world = _tag_corners_world(ref_anchor, args.anchor.tag_size)
  result, landmark_keep, secondary_anchor_world, optimized_body_poses = (
    _solve_joint_graph(cameras, keys, args, cross_obs, anchor_corners_world)
  )
  mark("joint LM (cull-reoptimize)")

  _adopt_ba_refined_estimates(cameras, result, keys)

  stamp = time.strftime("%Y%m%d_%H%M%S")
  rrd_path = args.output_dir / f"track_{stamp}.rrd"

  tag_size_by_id: dict[int, float] = {
    **{tid: float(args.anchor.tag_size) for tid in args.anchor.tag_ids},
    **{int(tid): float(args.gripper.tag_size) for tid in args.base_tag_ids},
    **{
      int(tid): float(args.gripper.tag_size)
      for tid in (*left_finger_ids, *right_finger_ids)
      if int(tid) > 0
    },
  }

  # The reference anchor fixes the gauge; each secondary anchor takes its re-estimate.
  world_T_anchor = {anchor_tag_id: ref_anchor, **secondary_anchor_world}
  anchor_corners_by_tag = {anchor_tag_id: anchor_corners_world} | {
    tag_id: _tag_corners_world(pose, args.anchor.tag_size)
    for tag_id, pose in secondary_anchor_world.items()
  }

  # Each tracker's base-tag corners in the world frame per keyframe, for cross rays.
  base_tag_corners_world: dict[str, list[Float64Array]] = {}
  for t in cameras:
    if t.body_T_base_tag is None:
      continue
    base_tag_corners_world[t.name] = [
      _tag_corners_world(pose.compose(t.body_T_base_tag), args.gripper.tag_size)
      for pose in optimized_body_poses[t.name]
    ]

  # Per-tracker audio offset to convert local keyframe_times → master clock.
  audio_offset_by_name: dict[str, float] = {
    t.name: offsets[i] for i, t in enumerate(cameras)
  }
  cross_obs_by_observer: dict[str, list[_CrossSighting]] = {t.name: [] for t in cameras}
  for sighting in cross_obs:
    cross_obs_by_observer[sighting.observer].append(sighting)

  per_tracker_results: list[dict[str, object]] = []
  viz_trackers: list[CameraVizData] = []
  culled_points: dict[str, tuple[dict[int, Float64Array], dict[int, Float64Array]]] = {}
  for t in cameras:
    viz_data, per_tracker_result, culled_pair = _build_tracker_viz_data(
      t,
      optimized_body_poses[t.name],
      result,
      keys,
      landmark_keep[t.name],
      args,
      tag_size_by_id,
      cross_obs_by_observer[t.name],
      anchor_corners_by_tag,
      base_tag_corners_world,
      audio_offset_by_name[t.name],
    )
    viz_trackers.append(viz_data)
    per_tracker_results.append(per_tracker_result)
    culled_points[t.name] = culled_pair

  bbox_diag = tuple(
    _bounding_box_diagonal(optimized_body_poses[t.name]) for t in cameras
  )
  path_len = tuple(_path_length(optimized_body_poses[t.name]) for t in cameras)
  summary = Summary(
    sources=tuple(str(p) for p in args.input_insvs),
    slots=names,
    audio_offsets=tuple(offsets),
    per_tracker_keyframes=tuple(t.n_keyframes for t in cameras),
    anchor_obs_per_tracker=tuple(len(_anchored_keyframes(t)) for t in cameras),
    cross_tracker_obs=len(cross_obs),
    initial_graph_error=result.initial_error,
    final_graph_error=result.final_error,
    solver_iterations=result.iterations,
    bounding_box_diagonal=bbox_diag,
    trajectory_path_length=path_len,
  )
  _print_capture_diagnostics(cameras, optimized_body_poses)

  raw = _build_raw_export(
    cameras,
    optimized_body_poses,
    culled_points,
    world_T_anchor,
    audio_offset_by_name,
    args,
  )

  summary_dict = asdict(summary)
  summary_dict["per_tracker"] = per_tracker_results
  summary_dict.update(raw)
  json_path = args.output_dir / f"track_{stamp}.json"
  json_path.write_text(json.dumps(summary_dict, indent=2, default=str))
  print(f"\nWrote {json_path}")

  # Best-effort viz after the result JSON is on disk: a viz failure never masks a solve.
  try:
    log_multi_ba_run(
      rrd_path=rrd_path,
      cameras=viz_trackers,
      world_T_anchor=world_T_anchor,
      reference_tag_id=anchor_tag_id,
      anchor_tag_size=args.anchor.tag_size,
      mask_pad_px=args.frontend.mask_tag_pad,
      urdf_path=urdf_path,
      open_viewer=args.open_viewer,
    )
    print(f"Wrote {rrd_path}")
  except Exception as exc:  # noqa: BLE001 - viz is best-effort; JSON already saved
    print(f"  [viz] skipped ({exc})")


if __name__ == "__main__":
  main(tyro.cli(Args))
