"""VI-BA driver: per-camera prep, one joint bundle in the reference anchor's world."""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gtsam
import numpy as np
import tyro
from specialist_dataset import schema
from specialist_dataset.frames import release_videos

from specialist.record.layer import write_estimate_layers
from specialist.record.session import SessionRecording
from specialist.vi_ba import gpu_flow
from specialist.vi_ba.blueprint import blueprint_for_slots
from specialist.vi_ba.compose import CameraKeys, anchored_keyframes, cross_sightings
from specialist.vi_ba.configs import Args, CameraArgs, validate
from specialist.vi_ba.features import set_precompute_concurrency
from specialist.vi_ba.geometry import translation
from specialist.vi_ba.gpu_lm import assert_backend
from specialist.vi_ba.reference_solve import adopt_ba_refined_estimates, solve_joint
from specialist.vi_ba.single_camera import (
  PreparedCamera,
  bound_malloc_arenas,
  cameras_that_fit,
  discover_camera_tags,
  prepare_camera,
  stage_timer,
)
from specialist.vi_ba.summary import Summary
from specialist.vi_ba.tag_discovery import GripperTags, validate_discovered
from specialist.vi_ba.urdf_joints import read_revolute_joints
from specialist.vi_ba.viz_build import build_run_estimate

# Arenas the C allocator may spread the walks' threads over.
_MALLOC_ARENAS = 8


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


def _print_capture_diagnostics(
  cameras: list[PreparedCamera],
  optimized_body_poses: dict[str, list[gtsam.Pose3]],
) -> None:
  print("\n=== max kf-to-kf speed (real motion if IMU agrees) ===")
  for camera in cameras:
    poses = optimized_body_poses[camera.slot]
    pts = np.array([translation(p) for p in poses])
    kf_t = np.asarray(camera.keyframe_times, dtype=np.float64)
    dt = np.diff(kf_t)
    speed = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.where(dt > 0, dt, 1.0)
    i = int(np.argmax(speed))
    print(
      f"  [{camera.slot}] max={speed[i]:.2f} m/s at kf {i}→{i + 1} (t={kf_t[i]:.2f}s)"
    )


def main(args: Args) -> None:
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  validate(args)
  stage = tempfile.TemporaryDirectory(prefix="lenses-")
  session = SessionRecording(args.base, Path(stage.name))
  slots = tuple(session.slots)
  assert_backend()
  rrd_path = schema.layer_rrd(args.output_dir, schema.ESTIMATE, session.session_id)
  viz_path = schema.layer_rrd(args.output_dir, schema.VIZ, session.session_id)
  existing = [p for p in (rrd_path, viz_path) if p.exists()]
  if existing and not args.overwrite:
    raise SystemExit(
      f"{', '.join(str(p) for p in existing)} exists; pass --overwrite to "
      f"replace it (a registered layer must also be re-registered, or its "
      f"byte-range index goes stale)"
    )
  mark = stage_timer()
  bound_malloc_arenas(_MALLOC_ARENAS)
  print(f"session {session.session_id}: cameras {list(slots)}")

  urdf_path = _resolve_urdf_path(args.urdf_path)

  concurrent = cameras_that_fit(session, slots)
  set_precompute_concurrency(concurrent)

  joints = read_revolute_joints(urdf_path)
  specs = {
    slot: CameraArgs(
      session=session,
      slot=slot,
      frontend=args.frontend,
      anchor=args.anchor,
      gripper=args.gripper,
      joints=joints,
    )
    for slot in slots
  }

  def _discover_one(slot: str) -> GripperTags:
    try:
      return discover_camera_tags(specs[slot])
    except Exception as exc:
      print(f"\n=== discover camera {slot!r} FAILED ===\n{exc}")
      raise

  # The front lenses stage while discovery reads the rear ones.
  with ThreadPoolExecutor(max_workers=len(slots)) as prefetch:
    front_loads = [prefetch.submit(session.lens_video, slot, "front") for slot in slots]
    with ThreadPoolExecutor(max_workers=concurrent) as discovery_pool:
      tags = dict(zip(slots, discovery_pool.map(_discover_one, slots), strict=True))
    for load in front_loads:
      load.result()
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
    front_obs = sum(len(obs) for obs in camera.front.tracks.values())
    rejected_obs = sum(len(obs) for obs in camera.front.rejected.values())
    print(
      f"  [{slot}/prep] kf={camera.n_keyframes}  "
      f"anchor_obs={len(anchored_keyframes(camera))}  "
      f"front_pts={len(camera.front.points)}  front_obs={front_obs}  "
      f"rejected_obs={rejected_obs}  "
      f"tags={camera.tags.base}/{camera.tags.fingers[0]}/{camera.tags.fingers[1]}"
    )
    return camera

  try:
    with ThreadPoolExecutor(max_workers=concurrent) as prep_pool:
      cameras = list(prep_pool.map(_prep_one, slots))
  finally:
    gpu_flow.shutdown()
    release_videos()
    stage.cleanup()
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

  adopt_ba_refined_estimates(cameras, solve.result, keys, joints)

  estimate = build_run_estimate(cameras, solve, keys, args, cross_obs)
  mark("estimate built")

  worst_move_translation, worst_move_rotation = solve.worst_anchor_move
  summary = Summary(
    cross_camera_obs=len(cross_obs),
    initial_graph_error=solve.result.initial_error,
    final_graph_error=solve.result.final_error,
    solver_iterations=solve.result.iterations,
    anchor_move_translation=worst_move_translation,
    anchor_move_rotation=worst_move_rotation,
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
    urdf_path=urdf_path,
    session_id=session.session_id,
  )
  # The dashboard for sessions opened from their files; each solve rewrites it.
  blueprints_dir = args.output_dir / "blueprints"
  blueprints_dir.mkdir(parents=True, exist_ok=True)
  rbl = blueprints_dir / "specialist.rbl"
  blueprint_for_slots(
    [c.slot for c in estimate.cameras],
    anchor_tag_ids=sorted(estimate.world_T_anchor),
    anchor_tag_size=args.anchor.tag_size,
  ).save(schema.APPLICATION_ID, str(rbl))
  mark("layers written")
  print(f"Wrote {rrd_path}, {viz_path} and {rbl}")


if __name__ == "__main__":
  main(tyro.cli(Args))
