"""VI-BA driver for audio-synced captures in the first tracker's anchor world."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gtsam
import numpy as np
import numpy.typing as npt
import tyro

from specialist.graph.factors import (
  make_anchor_observation_factor,
  make_anchor_prior,
  make_cross_tracker_factor,
)
from specialist.loader.audio_sync import sync_two_claps
from specialist.loader.mei_model import MeiParams, mei_project
from specialist.synthesis.tag_detection import TagObservation
from specialist.vi_ba.build import LensSetup, build_vi_graph
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
from specialist.vi_ba.optimize import optimize_graph
from specialist.vi_ba.pipeline import capture_duration
from specialist.vi_ba.single_camera import (
  CameraArgs,
  PreparedCamera,
  prepare_camera,
)
from specialist.vi_ba.tag_detection import (
  TagDetection,
  estimate_lens_T_tag,
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


def _factor_observation_lines(
  observer: PreparedCamera,
  observer_poses: list[gtsam.Pose3],
  cross_obs_for_observer: list[TagObservation],
) -> tuple[list[Float64Array], list[Float64Array]]:
  """Per-keyframe ``(n, 2, 3)`` lens-to-tag lines: ``(anchor_lines, cross_lines)``."""
  anchor_lines: list[Float64Array] = []
  cross_lines: list[Float64Array] = []
  for kf in range(observer.n_keyframes):
    body_pose = observer_poses[kf]
    anchor_strips: list[Float64Array] = []
    for side, lens_T_tag in observer.anchor_obs_per_kf.get(kf, {}).items():
      lens = observer.front_lens if side == "front" else observer.rear_lens
      lens_world = body_pose.compose(lens.body_T_lens)
      lens_xyz = np.asarray(lens_world.translation(), dtype=np.float64)
      tag_xyz = np.asarray(
        lens_world.compose(lens_T_tag).translation(), dtype=np.float64
      )
      anchor_strips.append(np.stack([lens_xyz, tag_xyz], axis=0))
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


# Past this radius fraction the anchor σ gets 7× more.
_EXTREME_RADIUS_FRAC = 0.65


def _radius_frac(pixel_center: Float64Array, frame_shape: tuple[int, ...]) -> float:
  """Distance from the image center, ``1.0`` at the lens optical-circle edge."""
  h, w = frame_shape[:2]
  cx, cy = float(w) / 2.0, float(h) / 2.0
  return float(
    np.sqrt((pixel_center[0] - cx) ** 2 + (pixel_center[1] - cy) ** 2)
    / (min(w, h) / 2.0)
  )


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


def _scaled_sigma(base: float, rf: float, pivot: float, scale: float) -> float:
  if rf <= pivot:
    return base
  return base * (1.0 + (rf - pivot) / scale) ** 2


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
  slim.front_frames = tracker.front_frames[:1]
  slim.rear_frames = tracker.rear_frames[:1]
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

  cameras: list[PreparedCamera] = []
  left_finger_ids = args.left_finger_tag_ids or (0,) * len(args.input_insvs)
  right_finger_ids = args.right_finger_tag_ids or (0,) * len(args.input_insvs)
  for name, path, start, base_tag_id, left_id, right_id in zip(
    names,
    args.input_insvs,
    starts,
    args.base_tag_ids,
    left_finger_ids,
    right_finger_ids,
    strict=True,
  ):
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
    cameras.append(tracker)
    mark(f"tracker {name!r} prepared")

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

  # Anchor variable + per-tracker anchor observation factors.
  anchor_key = keys.anchor_key()
  values.insert(anchor_key, ref_anchor)
  graph.add(
    make_anchor_prior(
      anchor_key,
      ref_anchor,
      args.anchor.prior_translation_sigma,
      args.anchor.prior_rotation_sigma,
    )
  )
  pivot = args.anchor.sigma_radius_pivot
  rscale = args.anchor.sigma_radius_scale
  sigma_rs: list[float] = []
  # Extra σ multiplier past _EXTREME_RADIUS_FRAC, where AprilTag PnP fails badly.
  extreme_r_mult = 7.0
  for t in cameras:
    for kf, sides in t.anchor_obs_per_kf.items():
      for side, lens_T_tag in sides.items():
        lens = t.front_lens if side == "front" else t.rear_lens
        tags_per_kf = t.front_tags_per_kf if side == "front" else t.rear_tags_per_kf
        frames = t.front_frames if side == "front" else t.rear_frames
        anchor_det = next(
          (d for d in tags_per_kf[kf] if int(d.tag_id) == int(args.anchor.tag_id)),
          None,
        )
        rf = (
          _radius_frac(anchor_det.center, frames[0].shape)
          if anchor_det and frames
          else pivot
        )
        sigma_t = _scaled_sigma(args.anchor.translation_sigma, rf, pivot, rscale)
        sigma_r = _scaled_sigma(args.anchor.rotation_sigma, rf, pivot, rscale)
        if rf > _EXTREME_RADIUS_FRAC:
          sigma_t *= extreme_r_mult
          sigma_r *= extreme_r_mult
        sigma_rs.append(sigma_r)
        graph.add(
          make_anchor_observation_factor(
            keys.pose_key(t.name, kf),
            anchor_key,
            lens_T_tag,
            lens.body_T_lens,
            sigma_t,
            sigma_r,
            huber_k=args.anchor.huber_k,
          )
        )
  if sigma_rs:
    print(
      f"anchor σ_r min/median/max = "
      f"{min(sigma_rs):.4f}/{float(np.median(sigma_rs)):.4f}/{max(sigma_rs):.4f} rad"
    )

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

  initial_error = float(graph.error(values))
  print(
    f"\njoint graph: {graph.size()} factors, {values.size()} variables, "
    f"initial error={initial_error:.1f}"
  )
  result = optimize_graph(graph, values, max_iterations=args.solver.max_iterations)
  mark(f"joint LM ({result.iterations} iters)")
  print(
    f"  error: {result.initial_error:.1f} → {result.final_error:.1f} "
    f"({result.iterations} iters)"
  )

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
  for t in cameras:
    poses = optimized_body_poses[t.name]
    f_pts = {
      tid: np.asarray(
        result.values.atPoint3(keys.front_landmark_key(t.name, tid)), dtype=np.float64
      )
      for tid in t.front_points
    }
    r_pts = {
      tid: np.asarray(
        result.values.atPoint3(keys.rear_landmark_key(t.name, tid)), dtype=np.float64
      )
      for tid in t.rear_points
    }
    front_rmse_per_kf = _per_keyframe_reproj_rmse_for_viz(
      result.values,
      t.front_tracks_tri,
      t.front_lens,
      lambda tid, n=t.name: keys.front_landmark_key(n, tid),
      lambda i, n=t.name: keys.pose_key(n, i),
      t.n_keyframes,
    )
    rear_rmse_per_kf = _per_keyframe_reproj_rmse_for_viz(
      result.values,
      t.rear_tracks_tri,
      t.rear_lens,
      lambda tid, n=t.name: keys.rear_landmark_key(n, tid),
      lambda i, n=t.name: keys.pose_key(n, i),
      t.n_keyframes,
    )
    f_rmse = _aggregate_rmse_for_print(front_rmse_per_kf)
    r_rmse = _aggregate_rmse_for_print(rear_rmse_per_kf)
    per_tracker_results.append(
      {
        "name": t.name,
        "n_keyframes": t.n_keyframes,
        "anchor_obs": len(t.anchor_obs_per_kf),
        "bounding_box_diagonal": _bounding_box_diagonal(poses),
        "path_length": _path_length(poses),
      }
    )
    print(
      f"  {t.name}: kf={t.n_keyframes} bbox={_bounding_box_diagonal(poses):.3f} m "
      f"front_rmse={f_rmse:.2f}px rear_rmse={r_rmse:.2f}px"
    )
    if t.body_T_base_tag is not None:
      base_tag_poses = [p.compose(t.body_T_base_tag) for p in poses]
    else:
      base_tag_poses = poses
    biases_final = [
      result.values.atConstantBias(keys.bias_key(t.name, i))
      for i in range(t.n_keyframes)
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
      _reproject_landmarks_per_kf(
        t.front_tracks_tri, f_pts, poses, t.front_lens, t.n_keyframes
      )
      if args.log_frames_in_rrd
      else None
    )
    rear_reproj = (
      _reproject_landmarks_per_kf(
        t.rear_tracks_tri, r_pts, poses, t.rear_lens, t.n_keyframes
      )
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
      t, poses, cross_obs_by_observer[t.name]
    )
    viz_trackers.append(
      CameraVizData(
        name=t.name,
        body_poses=poses,
        keyframe_times=np.asarray(t.keyframe_times, dtype=np.float64)
        - master_offset_by_name[t.name],
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
    )

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

  # Raw observations for downstream evaluation, free of derived metrics.
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
        str(tid): [
          float(x) for x in result.values.atPoint3(keys.front_landmark_key(t.name, tid))
        ]
        for tid in t.front_points
      },
      "rear": {
        str(tid): [
          float(x) for x in result.values.atPoint3(keys.rear_landmark_key(t.name, tid))
        ]
        for tid in t.rear_points
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
