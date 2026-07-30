"""VI-BA driver for audio-synced captures in the reference anchor's world."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gtsam
import numpy as np
import numpy.typing as npt
import tyro

from specialist.loader.audio_sync import sync_two_claps
from specialist.vi_ba import gpu_flow
from specialist.vi_ba.compose import (
  CameraKeys,
  anchored_keyframes,
  cross_tracker_observations,
)
from specialist.vi_ba.configs import Args, validate
from specialist.vi_ba.export import Summary, build_raw_export, write_run_json
from specialist.vi_ba.features import set_precompute_concurrency
from specialist.vi_ba.pipeline import capture_duration
from specialist.vi_ba.reference_solve import adopt_ba_refined_estimates, solve_joint
from specialist.vi_ba.single_camera import (
  CameraArgs,
  PreparedCamera,
  prepare_camera,
)
from specialist.vi_ba.viz import log_multi_ba_run
from specialist.vi_ba.viz_build import build_run_outputs

Float64Array = npt.NDArray[np.float64]


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


def _stage_timer() -> Callable[[str], None]:
  last = time.perf_counter()

  def mark(stage: str) -> None:
    nonlocal last
    now = time.perf_counter()
    print(f"  [{stage}] +{now - last:.1f}s")
    last = now

  return mark


def _print_capture_diagnostics(
  cameras: list[PreparedCamera],
  optimized_body_poses: dict[str, list[gtsam.Pose3]],
) -> None:
  """Print the maximum keyframe-to-keyframe speed."""
  print("\n=== max kf-to-kf speed (real motion if IMU agrees) ===")
  for t in cameras:
    poses = optimized_body_poses[t.name]
    pts = np.array([np.asarray(p.translation(), dtype=np.float64) for p in poses])
    kf_t = np.asarray(t.keyframe_times, dtype=np.float64)
    dt = np.diff(kf_t)
    speed = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.where(dt > 0, dt, 1.0)
    i = int(np.argmax(speed))
    print(f"  [{t.name}] max={speed[i]:.2f} m/s at kf {i}→{i + 1} (t={kf_t[i]:.2f}s)")


def main(args: Args) -> None:
  validate(args)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  names = _derive_names(args.input_insvs, args.slots)
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
      f"  kf={tracker.n_keyframes}  anchor_obs={len(anchored_keyframes(tracker))}  "
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

  cross_obs = cross_tracker_observations(
    cameras, list(args.base_tag_ids), args.gripper.tag_size
  )
  print(f"cross-tracker observations: {len(cross_obs)}")
  if cross_obs:
    by_pair = Counter((sighting.observer, sighting.owner) for sighting in cross_obs)
    for (obs_t, own_t), n in sorted(by_pair.items()):
      print(f"  {obs_t} → {own_t}: {n}")
  mark("cross-tracker obs assembled")

  keys = CameraKeys(list(names))
  selected = solve_joint(cameras, keys, args, cross_obs)
  anchor_tag_id = selected.ref_tag_id
  ref_anchor = selected.ref_anchor
  result = selected.result
  landmark_keep = selected.landmark_keep
  secondary_anchor_world = selected.secondary_world
  optimized_body_poses = selected.body_poses
  print(f"\nshared world: anchor t={np.asarray(ref_anchor.translation())}")
  mark("joint LM (cull-reoptimize)")

  adopt_ba_refined_estimates(cameras, result, keys)

  stamp = time.strftime("%Y%m%d_%H%M%S")
  rrd_path = args.output_dir / f"track_{stamp}.rrd"

  outputs = build_run_outputs(
    cameras,
    optimized_body_poses,
    result,
    keys,
    landmark_keep,
    args,
    cross_obs,
    offsets,
    ref_anchor,
    anchor_tag_id,
    secondary_anchor_world,
  )

  summary = Summary(
    sources=tuple(str(p) for p in args.input_insvs),
    slots=names,
    cross_tracker_obs=len(cross_obs),
    initial_graph_error=result.initial_error,
    final_graph_error=result.final_error,
    solver_iterations=result.iterations,
  )
  _print_capture_diagnostics(cameras, optimized_body_poses)

  raw = build_raw_export(
    cameras,
    optimized_body_poses,
    outputs.culled_points,
    outputs.world_T_anchor,
    outputs.audio_offset_by_tracker,
    anchor_tag_id,
    args,
  )

  json_path = args.output_dir / f"track_{stamp}.json"
  write_run_json(json_path, summary, outputs.per_tracker, raw)
  print(f"\nWrote {json_path}")

  # Best-effort viz after the result JSON is on disk: a viz failure never masks a solve.
  try:
    log_multi_ba_run(
      rrd_path=rrd_path,
      cameras=outputs.viz_trackers,
      world_T_anchor=outputs.world_T_anchor,
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
