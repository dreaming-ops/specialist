"""VI-BA driver for audio-synced captures in the first tracker's anchor world."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gtsam
import numpy as np
import numpy.typing as npt
import tyro

from specialist.graph.factors import make_cross_tracker_factor
from specialist.loader.audio_sync import sync_two_claps
from specialist.loader.mei_model import MeiParams, mei_project
from specialist.synthesis.tag_detection import TagObservation
from specialist.vi_ba.build import (
  LensSetup,
  add_mei_factor,
  build_vi_graph,
  huber_pixel_noise,
)
from specialist.vi_ba.compose import CameraKeys
from specialist.vi_ba.configs import (
  AnchorCfg,
  CrossTrackerCfg,
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
  # Cache the per-tracker prep by its args; incompatible with --log-frames-in-rrd.
  prep_cache: bool = False
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
  cross: CrossTrackerCfg = field(default_factory=CrossTrackerCfg)
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


def _align_to_shared_world(cameras: list[PreparedCamera]) -> gtsam.Pose3:
  """Map anchor-seeing cameras into the first one's world; return its anchor pose."""
  ref = next(
    (t.anchor_local for t in cameras if t.anchor_local is not None),
    None,
  )
  if ref is None:
    raise RuntimeError("no tracker observes the anchor; cannot share a world")
  for t in cameras:
    if t.anchor_local is None:
      continue
    align = ref.compose(t.anchor_local.inverse())
    align_R = align.rotation().matrix()
    align_t = np.asarray(align.translation(), dtype=np.float64)
    t.body_poses = [align.compose(p) for p in t.body_poses]
    t.velocities = [align_R @ v for v in t.velocities]
    t.front_points = {tid: align_R @ p + align_t for tid, p in t.front_points.items()}
    t.rear_points = {tid: align_R @ p + align_t for tid, p in t.rear_points.items()}
  return ref


def _cross_tracker_observations(
  cameras: list[PreparedCamera],
  base_tag_ids: list[int],
  base_tag_size: float,
) -> list[TagObservation]:
  """A ``TagObservation`` per PnP of an owner's base tag in an observer's frames."""
  obs: list[TagObservation] = []
  ids = {t.name: base_tag_ids[i] for i, t in enumerate(cameras)}
  for owner_idx, owner in enumerate(cameras):
    owner_tag_id = ids[owner.name]
    for obs_idx, observer in enumerate(cameras):
      if obs_idx == owner_idx:
        continue
      n = min(observer.n_keyframes, owner.n_keyframes)
      for side, lens, tags_per_kf in (
        ("front", observer.front_lens, observer.front_tags_per_kf),
        ("rear", observer.rear_lens, observer.rear_tags_per_kf),
      ):
        for kf in range(n):
          for det in tags_per_kf[kf]:
            if det.tag_id != owner_tag_id:
              continue
            lens_T_tag = estimate_lens_T_tag(det.corners, base_tag_size, lens.params)
            if lens_T_tag is None:
              continue
            obs.append(
              TagObservation(
                observer_tracker=observer.name,
                fisheye_side="front" if side == "front" else "back",
                keyframe_index=kf,
                tag_id=f"tracker_{owner.name}",
                tag_kind="tracker",
                tag_owner_tracker=owner.name,
                camera_to_tag=lens_T_tag,
              )
            )
            break  # one detection per (kf, side) — owner's tag is unique
  return obs


def _anchor_corners_world(ref_anchor: gtsam.Pose3, tag_size: float) -> Float64Array:
  """The anchor tag's four metric corners in the world frame, ``(4, 3)``."""
  return np.array(
    [
      np.asarray(ref_anchor.transformFrom(gtsam.Point3(*corner)), dtype=np.float64)
      for corner in tag_corners_local(tag_size)
    ]
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


def _factor_observation_lines(
  observer: PreparedCamera,
  observer_poses: list[gtsam.Pose3],
  cross_obs_for_observer: list[TagObservation],
  anchor_sightings: list[tuple[LensSetup, int, TagDetection]],
  anchor_corners_world: Float64Array,
) -> tuple[list[Float64Array], list[Float64Array]]:
  """Anchor-corner and cross-tracker lines, as per-keyframe ``(n, 2, 3)`` endpoints."""
  anchor_by_kf: dict[int, list[LensSetup]] = {}
  for lens, kf, _det in anchor_sightings:
    anchor_by_kf.setdefault(kf, []).append(lens)
  anchor_lines: list[Float64Array] = []
  cross_lines: list[Float64Array] = []
  for kf in range(observer.n_keyframes):
    body_pose = observer_poses[kf]
    anchor_strips: list[Float64Array] = []
    for lens in anchor_by_kf.get(kf, ()):
      lens_xyz = np.asarray(
        body_pose.compose(lens.body_T_lens).translation(), dtype=np.float64
      )
      for corner_xyz in anchor_corners_world:
        anchor_strips.append(np.stack([lens_xyz, corner_xyz]))
    anchor_lines.append(
      np.stack(anchor_strips) if anchor_strips else np.empty((0, 2, 3))
    )
    cross_strips: list[Float64Array] = []
    for o in cross_obs_for_observer:
      if o.keyframe_index != kf:
        continue
      lens = observer.front_lens if o.fisheye_side == "front" else observer.rear_lens
      lens_world = body_pose.compose(lens.body_T_lens)
      lens_xyz = np.asarray(lens_world.translation(), dtype=np.float64)
      tag_xyz = np.asarray(
        lens_world.compose(o.camera_to_tag).translation(), dtype=np.float64
      )
      cross_strips.append(np.stack([lens_xyz, tag_xyz], axis=0))
    cross_lines.append(
      np.stack(cross_strips, axis=0)
      if cross_strips
      else np.empty((0, 2, 3), dtype=np.float64)
    )
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


def _project_tag_axes_per_lens(
  tags_per_kf: list,
  lens: LensSetup,
  tag_size_by_id: dict[int, float],
  n_keyframes: int,
) -> list[Float64Array]:
  out: list[Float64Array] = []
  for kf in range(n_keyframes):
    tags = tags_per_kf[kf] if kf < len(tags_per_kf) else []
    strips: list[np.ndarray] = []
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


def _per_keyframe_reproj_rmse_for_viz(
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
  return [
    float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    for errs in buckets
  ]


def _aggregate_rmse_for_print(per_kf: list[float]) -> float:
  finite = [v for v in per_kf if np.isfinite(v)]
  return float(np.sqrt(np.mean(np.square(finite)))) if finite else float("nan")


def _pose_to_dict(p: gtsam.Pose3) -> dict:
  return {
    "R": [list(map(float, row)) for row in p.rotation().matrix()],
    "t": [float(x) for x in p.translation()],
  }


def _mei_to_dict(p: MeiParams) -> dict:
  return {
    "fx": float(p.fx),
    "fy": float(p.fy),
    "cx": float(p.cx),
    "cy": float(p.cy),
    "k1": float(p.k1),
    "k2": float(p.k2),
    "k3": float(p.k3),
    "p1": float(p.p1),
    "p2": float(p.p2),
    "xi": float(p.xi),
  }


def _prepare_cached(name: str, targs: "CameraArgs", enabled: bool) -> "PreparedCamera":
  """prepare_camera cached on disk; a cached run cannot use --log-frames-in-rrd."""
  if not enabled:
    return prepare_camera(name, targs)
  cache_dir = Path("/tmp/specialist_prep_cache")
  try:
    mt = targs.insv_path.stat().st_mtime
  except OSError:
    mt = 0.0
  key = hashlib.sha256(f"{targs!r}|{mt}".encode()).hexdigest()[:16]
  cpath = cache_dir / f"{name}_{key}.pkl"
  if cpath.exists():
    print(f"  [prep cache HIT] {cpath.name} — skipping decode+frontend")
    return pickle.loads(cpath.read_bytes())
  tracker = prepare_camera(name, targs)
  cache_dir.mkdir(parents=True, exist_ok=True)
  slim = copy.copy(tracker)
  slim.front_frames = []
  slim.rear_frames = []
  try:
    cpath.write_bytes(pickle.dumps(slim))
    print(f"  [prep cache saved] {cpath.name}")
  except Exception as exc:  # noqa: BLE001 - caching is best-effort
    print(f"  [prep cache save failed: {exc}]")
  return tracker


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
  cross_obs: list[TagObservation],
  anchor_tag_id: int,
  anchor_corners_world: Float64Array,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """The joint graph and seed values: VI, anchor-corner, tag and smoothness factors."""
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

  # Reproject the anchor tag's four metric corners through the MEI model.
  corner_keys = [keys.anchor_corner_key(j) for j in range(4)]
  corner_prior = gtsam.noiseModel.Isotropic.Sigma(3, 1.0e-4)
  corner_noise = huber_pixel_noise(args.anchor.corner_reproj_sigma)
  for corner_key, corner_xyz in zip(corner_keys, anchor_corners_world, strict=True):
    point = gtsam.Point3(*corner_xyz)
    values.insert(corner_key, point)
    graph.add(gtsam.PriorFactorPoint3(corner_key, point, corner_prior))

  n_anchor_sightings = 0
  for t in cameras:
    for lens, kf, anchor_det in _anchor_sightings(t, anchor_tag_id):
      n_anchor_sightings += 1
      pose_key = keys.pose_key(t.name, kf)
      # The detected tag direction in the lens frame fixes the omni depth sign.
      side = "front" if lens is t.front_lens else "rear"
      lens_T_tag = t.anchor_obs_per_kf.get(kf, {}).get(side)
      obs_bearing = (
        None
        if lens_T_tag is None
        else np.asarray(lens_T_tag.translation(), dtype=np.float64)
      )
      for corner_key, pixel in zip(corner_keys, anchor_det.corners, strict=True):
        add_mei_factor(
          graph, pose_key, corner_key, pixel, corner_noise, lens, obs_bearing
        )
  print(f"anchor corner reprojection: {n_anchor_sightings} anchor sightings")

  # The owner's body_T_tag comes from its first stationary rear-lens PnP of its tag.
  body_T_tag_by_name = {
    t.name: t.body_T_base_tag for t in cameras if t.body_T_base_tag is not None
  }
  for o in cross_obs:
    owner_offset = body_T_tag_by_name.get(o.tag_owner_tracker or "")
    if owner_offset is None:
      continue
    observer_lens = (
      next(t for t in cameras if t.name == o.observer_tracker).front_lens
      if o.fisheye_side == "front"
      else next(t for t in cameras if t.name == o.observer_tracker).rear_lens
    )
    graph.add(
      make_cross_tracker_factor(
        keys.pose_key(o.observer_tracker, o.keyframe_index),
        keys.pose_key(o.tag_owner_tracker or "", o.keyframe_index),
        o.camera_to_tag,
        observer_lens.body_T_lens,
        owner_offset,
        args.cross.translation_sigma,
        args.cross.rotation_sigma,
      )
    )

  # Finger-tag cross factors use the owner's per-keyframe body_T_finger and a looser σ.
  finger_cross_count = 0
  finger_cross_t_sigma = 0.040  # 40mm
  finger_cross_r_sigma = 0.10  # ~6°
  for owner in cameras:
    for finger_id, btf_per_kf in (
      (owner.left_finger_tag_id, owner.body_T_left_finger_per_kf),
      (owner.right_finger_tag_id, owner.body_T_right_finger_per_kf),
    ):
      if finger_id <= 0 or not btf_per_kf:
        continue
      for observer in cameras:
        if observer.name == owner.name:
          continue
        n = min(observer.n_keyframes, owner.n_keyframes)
        for lens, tags_per_kf in (
          (observer.front_lens, observer.front_tags_per_kf),
          (observer.rear_lens, observer.rear_tags_per_kf),
        ):
          for kf in range(n):
            if kf not in btf_per_kf:
              continue
            for det in tags_per_kf[kf]:
              if det.tag_id != finger_id:
                continue
              lens_T_tag = estimate_lens_T_tag(
                det.corners, args.gripper.tag_size, lens.params
              )
              if lens_T_tag is None:
                continue
              factor = make_cross_tracker_factor(
                keys.pose_key(observer.name, kf),
                keys.pose_key(owner.name, kf),
                lens_T_tag,
                lens.body_T_lens,
                btf_per_kf[kf],
                finger_cross_t_sigma,
                finger_cross_r_sigma,
                huber_k=1.0,
              )
              graph.add(factor)
              finger_cross_count += 1
              break
  if finger_cross_count:
    print(f"finger-tag cross-tracker factors: {finger_cross_count}")

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
    for k in range(1, t.n_keyframes):
      # Both endpoints must be unanchored; an anchored k-1 fights k's visual factors.
      if (k - 1) in t.anchor_obs_per_kf or k in t.anchor_obs_per_kf:
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
  print(f"smoothness priors: pose {smooth_count}, velocity {vel_smooth_count}")
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


def _solve_with_cull_reoptimize(
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  max_iterations: int,
) -> tuple[OptimizeResult, dict[str, tuple[dict[int, dict[int, Float64Array]], ...]]]:
  """Re-solved result and each tracker's kept tracks after an outlier-landmark cull."""
  initial_error = float(graph.error(values))
  print(
    f"\njoint graph: {graph.size()} factors, {values.size()} variables, "
    f"initial error={initial_error:.1f}"
  )
  result = optimize_graph(graph, values, max_iterations=max_iterations)
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
  if outlier_keys:
    pruned = gtsam.NonlinearFactorGraph()
    for i in range(graph.size()):
      factor = graph.at(i)
      if not any(k in outlier_keys for k in factor.keys()):
        pruned.push_back(factor)
    kept_values = gtsam.Values(result.values)
    for key in outlier_keys:
      kept_values.erase(key)
    result = optimize_graph(pruned, kept_values, max_iterations=max_iterations)
    print(
      f"  cull-reoptimize: dropped {len(outlier_keys)} outlier landmarks, "
      f"error → {result.final_error:.1f} ({result.iterations} iters)"
    )
  return result, landmark_keep


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
  ref_anchor: gtsam.Pose3,
  master_offset_by_name: dict[str, float],
  args: Args,
) -> dict[str, object]:
  """Raw observations for downstream evaluation, free of derived metrics."""
  raw: dict[str, object] = {}
  raw["body_poses_by_tracker"] = {
    t.name: [_pose_to_dict(p) for p in optimized_body_poses[t.name]] for t in cameras
  }
  raw["ref_anchor_pose_world"] = _pose_to_dict(ref_anchor)
  raw["lens_by_tracker"] = {
    t.name: {
      "front": _lens_side_dict(t.front_lens, t.decoded_width),
      "rear": _lens_side_dict(t.rear_lens, t.decoded_width),
    }
    for t in cameras
  }
  raw["anchor_obs_by_tracker"] = {
    t.name: [
      {"kf": int(kf), "side": side, "lens_T_tag": _pose_to_dict(lens_T_tag)}
      for kf, sides in t.anchor_obs_per_kf.items()
      for side, lens_T_tag in sides.items()
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
  raw["audio_offset_by_tracker"] = master_offset_by_name
  raw["body_T_base_tag_by_tracker"] = {
    t.name: (_pose_to_dict(t.body_T_base_tag) if t.body_T_base_tag else None)
    for t in cameras
  }
  raw["anchor_tag_id"] = int(args.anchor.tag_id)
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
  cross_obs_for_t: list[TagObservation],
  anchor_tag_id: int,
  anchor_corners_world: Float64Array,
  master_offset: float,
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
  front_rmse_per_kf = _per_keyframe_reproj_rmse_for_viz(
    result.values,
    front_obs,
    t.front_lens,
    lambda tid, n=t.name: keys.front_landmark_key(n, tid),
    lambda i, n=t.name: keys.pose_key(n, i),
    t.n_keyframes,
  )
  rear_rmse_per_kf = _per_keyframe_reproj_rmse_for_viz(
    result.values,
    rear_obs,
    t.rear_lens,
    lambda tid, n=t.name: keys.rear_landmark_key(n, tid),
    lambda i, n=t.name: keys.pose_key(n, i),
    t.n_keyframes,
  )
  f_rmse = _aggregate_rmse_for_print(front_rmse_per_kf)
  r_rmse = _aggregate_rmse_for_print(rear_rmse_per_kf)
  per_tracker_result: dict[str, object] = {
    "name": t.name,
    "n_keyframes": t.n_keyframes,
    "anchor_obs": len(t.anchor_obs_per_kf),
    "bounding_box_diagonal": _bounding_box_diagonal(poses),
    "path_length": _path_length(poses),
  }
  print(
    f"  {t.name}: kf={t.n_keyframes} bbox={_bounding_box_diagonal(poses):.3f} m "
    f"front_rmse={f_rmse:.2f}px rear_rmse={r_rmse:.2f}px"
  )
  if t.body_T_base_tag is not None:
    base_tag_poses = [p.compose(t.body_T_base_tag) for p in poses]
  else:
    base_tag_poses = poses
  biases_final = [
    result.values.atConstantBias(keys.bias_key(t.name, i)) for i in range(t.n_keyframes)
  ]
  front_tag_axes = _project_tag_axes_per_lens(
    t.front_tags_per_kf,
    t.front_lens,
    tag_size_by_id,
    t.n_keyframes,
  )
  rear_tag_axes = _project_tag_axes_per_lens(
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
    _anchor_sightings(t, anchor_tag_id),
    anchor_corners_world,
  )
  viz_data = CameraVizData(
    name=t.name,
    body_poses=poses,
    keyframe_times=np.asarray(t.keyframe_times, dtype=np.float64) - master_offset,
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

  def _prep_one(spec: tuple[str, Path, float, int, int, int]) -> PreparedCamera:
    name, path, start, base_tag_id, left_id, right_id = spec
    print(f"\n=== prep tracker {name!r} ({path.name}, start={start:.3f}s) ===")
    tracker = _prepare_cached(
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
      ),
      args.prep_cache,
    )
    print(
      f"  kf={tracker.n_keyframes}  anchor_obs={len(tracker.anchor_obs_per_kf)}  "
      f"front_pts={len(tracker.front_points)}  rear_pts={len(tracker.rear_points)}  "
      f"body_T_base_tag={'yes' if tracker.body_T_base_tag else 'no'}"
    )
    return tracker

  # The flow precompute shrinks its per-lens pool to fit the shared cv2 thread pool.
  set_precompute_concurrency(len(specs))
  with ThreadPoolExecutor(max_workers=len(specs)) as prep_pool:
    cameras = list(prep_pool.map(_prep_one, specs))
  mark("cameras prepared")

  ref_anchor = _align_to_shared_world(cameras)
  print(f"\nshared world: anchor t={np.asarray(ref_anchor.translation())}")

  cross_obs = _cross_tracker_observations(
    cameras, list(args.base_tag_ids), args.gripper.tag_size
  )
  print(f"cross-tracker observations: {len(cross_obs)}")
  if cross_obs:
    by_pair: dict[tuple[str, str], int] = {}
    for o in cross_obs:
      key = (o.observer_tracker, o.tag_owner_tracker or "?")
      by_pair[key] = by_pair.get(key, 0) + 1
    for (obs_t, own_t), n in sorted(by_pair.items()):
      print(f"  {obs_t} → {own_t}: {n}")
  mark("cross-tracker obs assembled")

  keys = CameraKeys(list(names))
  anchor_tag_id = int(args.anchor.tag_id)
  anchor_corners_world = _anchor_corners_world(ref_anchor, args.anchor.tag_size)
  graph, values = _assemble_joint_graph(
    cameras, keys, args, cross_obs, anchor_tag_id, anchor_corners_world
  )

  result, landmark_keep = _solve_with_cull_reoptimize(
    graph, values, cameras, keys, args.solver.max_iterations
  )
  mark("joint LM (cull-reoptimize)")

  stamp = time.strftime("%Y%m%d_%H%M%S")
  rrd_path = args.output_dir / f"track_{stamp}.rrd"

  urdf_path = args.urdf_path
  if urdf_path is None:
    try:
      from specialist_description import get_urdf_path

      urdf_path = Path(get_urdf_path("specialist_handheld"))
    except ImportError as exc:  # headless env without the gripper-URDF package
      print(f"  [viz] gripper URDF unavailable ({exc}); skipping URDF overlay")
      urdf_path = None

  tag_size_by_id: dict[int, float] = {
    int(args.anchor.tag_id): float(args.anchor.tag_size),
    **{int(tid): float(args.gripper.tag_size) for tid in args.base_tag_ids},
    **{
      int(tid): float(args.gripper.tag_size)
      for tid in (*left_finger_ids, *right_finger_ids)
      if int(tid) > 0
    },
  }

  optimized_body_poses: dict[str, list[gtsam.Pose3]] = {
    t.name: [
      result.values.atPose3(keys.pose_key(t.name, i)) for i in range(t.n_keyframes)
    ]
    for t in cameras
  }

  # Per-tracker audio offset to convert local keyframe_times → master clock.
  master_offset_by_name: dict[str, float] = {
    t.name: offsets[i] for i, t in enumerate(cameras)
  }
  cross_obs_by_observer: dict[str, list[TagObservation]] = {t.name: [] for t in cameras}
  for o in cross_obs:
    cross_obs_by_observer[o.observer_tracker].append(o)

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
      anchor_tag_id,
      anchor_corners_world,
      master_offset_by_name[t.name],
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
    anchor_obs_per_tracker=tuple(len(t.anchor_obs_per_kf) for t in cameras),
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
    ref_anchor,
    master_offset_by_name,
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
      world_T_anchor=ref_anchor,
      anchor_tag_size=args.anchor.tag_size,
      anchor_tag_id=args.anchor.tag_id,
      mask_pad_px=args.frontend.mask_tag_pad,
      urdf_path=urdf_path,
      open_viewer=args.open_viewer,
    )
    print(f"Wrote {rrd_path}")
  except Exception as exc:  # noqa: BLE001 - viz is best-effort; JSON already saved
    print(f"  [viz] skipped ({exc})")


if __name__ == "__main__":
  main(tyro.cli(Args))
