"""VI-BA driver: per-tracker prep, one joint bundle in the reference anchor's world."""

from __future__ import annotations

import dataclasses
import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gtsam
import numpy as np
import tyro

from specialist.record.layer import write_solution_layers
from specialist.record.session import SessionRecording
from specialist.vi_ba import gpu_flow
from specialist.vi_ba.compose import CameraKeys, anchored_keyframes, cross_sightings
from specialist.vi_ba.configs import Args, CameraArgs, validate
from specialist.vi_ba.features import set_precompute_concurrency
from specialist.vi_ba.gpu_lm import assert_backend
from specialist.vi_ba.reference_solve import adopt_ba_refined_estimates, solve_joint
from specialist.vi_ba.single_camera import (
  PreparedCamera,
  cameras_that_fit,
  prepare_camera,
)
from specialist.vi_ba.summary import Summary
from specialist.vi_ba.viz_build import build_run_outputs


def _resolve_urdf_path(explicit: Path | None) -> Path | None:
  """The gripper URDF: an explicit path, else the installed asset, else None."""
  if explicit is not None:
    return explicit
  try:
    from specialist_description import get_urdf_path

    return Path(get_urdf_path("specialist_handheld"))
  except ImportError as exc:  # headless env without the gripper-URDF package
    print(f"  [warn] [gripper] URDF unavailable ({exc}); finger joint factors disabled")
    return None


def _stage_timer() -> Callable[[str], None]:
  last = time.perf_counter()

  def mark(stage: str) -> None:
    nonlocal last
    now = time.perf_counter()
    print(f"  [timing] {stage} +{now - last:.1f}s")
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


def _non_default_flags(args: object, prefix: str = "") -> list[str]:
  """Every non-default value of a defaulted field, dotted through the nested configs."""
  out: list[str] = []
  for spec in dataclasses.fields(args):  # type: ignore[arg-type]
    value = getattr(args, spec.name)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
      out += _non_default_flags(value, f"{prefix}{spec.name}.")
      continue
    if spec.default is not dataclasses.MISSING:
      default = spec.default
    elif spec.default_factory is not dataclasses.MISSING:
      default = spec.default_factory()
    else:
      continue
    if value != default:
      out.append(f"{prefix}{spec.name}={value}")
  return out


def main(args: Args) -> None:
  # Line buffering shows each line as it prints when stdout goes to a file.
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  session = SessionRecording(args.base)
  names = tuple(session.slots)
  validate(args, len(names))
  if args.solver.gpu == "require":
    # A missing CUDA stack is a hard error in this mode; check it before the prep.
    assert_backend()
  rrd_path = args.output_dir / "track" / f"{session.session_id}.rrd"
  viz_path = args.output_dir / "viz" / f"{session.session_id}.rrd"
  existing = [p for p in (rrd_path, viz_path) if p.exists()]
  if existing and not args.overwrite:
    raise SystemExit(
      f"{', '.join(str(p) for p in existing)} exists; pass --overwrite to "
      f"replace it (a registered layer must also be re-registered, or its "
      f"byte-range index goes stale)"
    )
  mark = _stage_timer()
  print(f"session {session.session_id}: cameras {list(names)}")

  # The per-tracker id lists pair with the recording's names in sorted order.
  left_finger_ids = args.left_finger_tag_ids or (0,) * len(names)
  right_finger_ids = args.right_finger_tag_ids or (0,) * len(names)
  specs = list(
    zip(names, args.base_tag_ids, left_finger_ids, right_finger_ids, strict=True)
  )
  for name, base_tag_id, left_id, right_id in specs:
    print(f"  {name}: base tag {base_tag_id}, finger tags {left_id}/{right_id}")

  # Resolve the URDF once: the finger joint-angle factors and the viz overlay share it.
  urdf_path = _resolve_urdf_path(args.urdf_path)

  # The capture count sizes the flow worker pools and sets whether tag walks overlap.
  concurrent = cameras_that_fit(session, names)
  set_precompute_concurrency(concurrent)

  def _prep_one(spec: tuple[str, int, int, int]) -> PreparedCamera:
    name, base_tag_id, left_id, right_id = spec
    print(f"\n=== prep tracker {name!r} ===")
    try:
      tracker = prepare_camera(
        name,
        CameraArgs(
          session=session,
          slot=name,
          keyframe_every=args.keyframe_every,
          base_tag_id=base_tag_id,
          left_finger_tag_id=left_id,
          right_finger_tag_id=right_id,
          frontend=args.frontend,
          anchor=args.anchor,
          parallel_siblings=concurrent > 1,
          gripper=args.gripper,
          urdf_path=urdf_path,
        ),
      )
    except Exception as exc:
      # Name the failure at once: the traceback waits for sibling decodes to finish.
      print(f"\n=== prep tracker {name!r} FAILED: {exc!r} ===")
      raise
    print(
      f"  [{name}/prep] kf={tracker.n_keyframes}  "
      f"anchor_obs={len(anchored_keyframes(tracker))}  "
      f"front_pts={len(tracker.front.points)}  rear_pts={len(tracker.rear.points)}  "
      f"body_T_base_tag={'yes' if tracker.body_T_base_tag else 'no'}"
    )
    return tracker

  try:
    with ThreadPoolExecutor(max_workers=concurrent) as prep_pool:
      cameras = list(prep_pool.map(_prep_one, specs))
  finally:
    # Free the GPU flow workers' device memory for the joint bundle adjustment.
    gpu_flow.shutdown()
  mark("cameras prepared")

  cross_obs = cross_sightings(cameras, list(args.base_tag_ids), args.gripper.tag_size)
  print(f"[graph] cross-tracker observations: {len(cross_obs)}")
  if cross_obs:
    by_pair = Counter((sighting.observer, sighting.owner) for sighting in cross_obs)
    for (obs_t, own_t), n in sorted(by_pair.items()):
      print(f"  [graph] {obs_t} → {own_t}: {n}")
  mark("cross-tracker obs assembled")

  keys = CameraKeys(list(names))
  solve = solve_joint(cameras, keys, args, cross_obs)
  mark("joint LM (cull-reoptimize)")

  adopt_ba_refined_estimates(cameras, solve.result, keys)

  outputs = build_run_outputs(
    cameras,
    solve.body_poses,
    solve.result,
    keys,
    solve.landmark_keep,
    args,
    cross_obs,
    solve.world_T_ref_anchor,
    solve.ref_tag_id,
    solve.secondary_anchor_world,
  )

  worst_move_translation, worst_move_rotation = solve.worst_secondary_move
  summary = Summary(
    non_default_flags=tuple(_non_default_flags(args)),
    sources=tuple(session.sources),
    slots=names,
    cross_tracker_obs=len(cross_obs),
    initial_graph_error=solve.result.initial_error,
    final_graph_error=solve.result.final_error,
    solver_iterations=solve.result.iterations,
    secondary_anchor_move_translation=worst_move_translation,
    secondary_anchor_move_rotation=worst_move_rotation,
  )
  _print_capture_diagnostics(cameras, solve.body_poses)

  timeline_start = min(
    float(session.lens_times(n, lens)[0]) for n in names for lens in ("front", "rear")
  )
  # The frame period limits each image overlay to the video frame it comes from.
  frame_period = float(np.median(np.diff(session.lens_times(names[0], "front"))))
  write_solution_layers(
    track_path=rrd_path,
    viz_path=viz_path,
    timeline_start=timeline_start,
    frame_period=frame_period,
    outputs=outputs,
    summary=summary,
    reference_tag_id=solve.ref_tag_id,
    anchor_tag_size=args.anchor.tag_size,
    mask_pad_px=args.frontend.mask_tag_pad if args.frontend.mask_tags else -1,
    urdf_path=urdf_path,
    session_id=session.session_id,
    open_viewer=args.open_viewer,
  )
  print(f"Wrote {rrd_path} and {viz_path}")


if __name__ == "__main__":
  main(tyro.cli(Args))
