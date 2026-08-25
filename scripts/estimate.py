"""VI-BA driver: per-camera prep, one joint bundle in the reference anchor's world."""

from __future__ import annotations

import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gtsam
import numpy as np
import tyro

from specialist.record import schema
from specialist.record.layer import write_estimate_layers
from specialist.record.session import SessionRecording
from specialist.vi_ba import gpu_flow
from specialist.vi_ba.compose import CameraKeys, anchored_keyframes, cross_sightings
from specialist.vi_ba.configs import Args, CameraArgs, validate
from specialist.vi_ba.features import set_precompute_concurrency
from specialist.vi_ba.geometry import translation
from specialist.vi_ba.gpu_lm import assert_backend
from specialist.vi_ba.reference_solve import adopt_ba_refined_estimates, solve_joint
from specialist.vi_ba.single_camera import (
  PreparedCamera,
  cameras_that_fit,
  discover_camera_tags,
  prepare_camera,
)
from specialist.vi_ba.summary import Summary
from specialist.vi_ba.tag_discovery import validate_discovered
from specialist.vi_ba.urdf_joints import read_revolute_joints
from specialist.vi_ba.viz_build import build_run_estimate


def _resolve_urdf_path(explicit: Path | None) -> Path:
  """The gripper URDF: an explicit path, else the installed description asset."""
  if explicit is not None:
    return explicit
  try:
    from specialist_description import get_urdf_path

    return Path(get_urdf_path("specialist_handheld"))
  except ImportError as exc:
    raise SystemExit(
      f"the gripper URDF is unavailable ({exc}); pass --urdf-path"
    ) from exc


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
  print("\n=== max kf-to-kf speed (real motion if IMU agrees) ===")
  for t in cameras:
    poses = optimized_body_poses[t.name]
    pts = np.array([translation(p) for p in poses])
    kf_t = np.asarray(t.keyframe_times, dtype=np.float64)
    dt = np.diff(kf_t)
    speed = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.where(dt > 0, dt, 1.0)
    i = int(np.argmax(speed))
    print(f"  [{t.name}] max={speed[i]:.2f} m/s at kf {i}→{i + 1} (t={kf_t[i]:.2f}s)")


def main(args: Args) -> None:
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  validate(args)
  session = SessionRecording(args.base)
  slots = tuple(session.slots)
  if args.solver.gpu == "require":
    assert_backend()
  rrd_path = args.output_dir / "estimate" / f"{session.session_id}.rrd"
  viz_path = args.output_dir / "viz" / f"{session.session_id}.rrd"
  existing = [p for p in (rrd_path, viz_path) if p.exists()]
  if existing and not args.overwrite:
    raise SystemExit(
      f"{', '.join(str(p) for p in existing)} exists; pass --overwrite to "
      f"replace it (a registered layer must also be re-registered, or its "
      f"byte-range index goes stale)"
    )
  mark = _stage_timer()
  print(f"session {session.session_id}: cameras {list(slots)}")

  urdf_path = _resolve_urdf_path(args.urdf_path)

  concurrent = cameras_that_fit(session, slots)
  set_precompute_concurrency(concurrent)

  joints = read_revolute_joints(urdf_path)
  specs = {
    slot: CameraArgs(
      session=session,
      slot=slot,
      keyframe_every=args.keyframe_every,
      frontend=args.frontend,
      anchor=args.anchor,
      parallel_siblings=concurrent > 1,
      gripper=args.gripper,
      joints=joints,
    )
    for slot in slots
  }
  tags = {slot: discover_camera_tags(spec) for slot, spec in specs.items()}
  validate_discovered(tags)
  for slot, found in tags.items():
    print(f"  {slot}: base tag {found.base}, finger tags {found.fingers}")
  mark("tags discovered")

  def _prep_one(slot: str) -> PreparedCamera:
    print(f"\n=== prep camera {slot!r} ===")
    try:
      camera = prepare_camera(specs[slot], tags[slot])
    except Exception as exc:
      print(f"\n=== prep camera {slot!r} FAILED: {exc!r} ===")
      raise
    print(
      f"  [{slot}/prep] kf={camera.n_keyframes}  "
      f"anchor_obs={len(anchored_keyframes(camera))}  "
      f"front_pts={len(camera.front.points)}  rear_pts={len(camera.rear.points)}  "
      f"tags={camera.tags.base}/{camera.tags.fingers[0]}/{camera.tags.fingers[1]}"
    )
    return camera

  try:
    with ThreadPoolExecutor(max_workers=concurrent) as prep_pool:
      cameras = list(prep_pool.map(_prep_one, slots))
  finally:
    # Release the per-stream GPU workers so the linear solver has device memory.
    gpu_flow.shutdown()
  mark("cameras prepared")

  cross_obs = cross_sightings(cameras, args.gripper.tag_size)
  print(f"[graph] cross-camera observations: {len(cross_obs)}")
  if cross_obs:
    by_pair = Counter((sighting.observer, sighting.owner) for sighting in cross_obs)
    for (obs_t, own_t), n in sorted(by_pair.items()):
      print(f"  [graph] {obs_t} → {own_t}: {n}")
  mark("cross-camera obs assembled")

  keys = CameraKeys(list(slots))
  solve = solve_joint(cameras, keys, args, cross_obs)
  mark("joint LM (cull-reoptimize)")

  adopt_ba_refined_estimates(cameras, solve.result, keys)

  estimate = build_run_estimate(cameras, solve, keys, args, cross_obs)

  worst_move_translation, worst_move_rotation = solve.worst_secondary_move
  summary = Summary(
    cross_camera_obs=len(cross_obs),
    initial_graph_error=solve.result.initial_error,
    final_graph_error=solve.result.final_error,
    solver_iterations=solve.result.iterations,
    secondary_anchor_move_translation=worst_move_translation,
    secondary_anchor_move_rotation=worst_move_rotation,
  )
  _print_capture_diagnostics(cameras, solve.body_poses)

  timeline_start = min(
    float(session.lens_times(n, lens)[0]) for n in slots for lens in schema.LENSES
  )
  frame_period = float(np.median(np.diff(session.lens_times(slots[0], "front"))))
  write_estimate_layers(
    estimate_path=rrd_path,
    viz_path=viz_path,
    timeline_start=timeline_start,
    frame_period=frame_period,
    estimate=estimate,
    summary=summary,
    anchor_tag_size=args.anchor.tag_size,
    mask_pad_px=args.frontend.mask_tag_pad if args.frontend.mask_tags else -1,
    urdf_path=urdf_path,
    session_id=session.session_id,
    open_viewer=args.open_viewer,
  )
  print(f"Wrote {rrd_path} and {viz_path}")


if __name__ == "__main__":
  main(tyro.cli(Args))
