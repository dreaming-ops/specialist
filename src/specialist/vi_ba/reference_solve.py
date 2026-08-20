"""Joint VI-BA solve with a landmark cull, retried per candidate reference anchor."""

from __future__ import annotations

from dataclasses import dataclass, replace

import gtsam
import numpy as np

from specialist.vi_ba.compose import (
  CameraKeys,
  CrossSighting,
  LandmarkKeep,
  align_to_shared_world,
  assemble_joint_graph,
  points_from_values,
)
from specialist.vi_ba.configs import Args, GpuMode
from specialist.vi_ba.geometry import average_pose
from specialist.vi_ba.landmark_cull import cull_outlier_landmarks
from specialist.vi_ba.optimize import OptimizeResult, optimize_graph
from specialist.vi_ba.single_camera import PreparedCamera
from specialist.vi_ba.tag_detection import tag_corners_in_frame
from specialist.vi_ba.urdf_joints import FINGER_JOINTS

__all__ = [
  "ReferenceSolve",
  "adopt_ba_refined_estimates",
  "solve_joint",
]

# Anchor re-estimate moves above these flag weak sighting geometry.
_ANCHOR_CONSISTENCY_TRANS = 5e-3
_ANCHOR_CONSISTENCY_ROT = float(np.radians(0.2))

# The first solve only ranks residuals for the cull.
_PRECULL_MAX_ITERATIONS = 8


def _prune_factor_graph(
  graph: gtsam.NonlinearFactorGraph, drop_keys: set[int]
) -> gtsam.NonlinearFactorGraph:
  """A copy of ``graph`` without the factors that touch a key in ``drop_keys``."""
  pruned = gtsam.NonlinearFactorGraph()
  for i in range(graph.size()):
    factor = graph.at(i)
    if not any(k in drop_keys for k in factor.keys()):
      pruned.push_back(factor)
  return pruned


def _outlier_landmark_keys(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  landmark_keep: LandmarkKeep,
) -> set[int]:
  """The landmark keys the cull dropped, across both lenses of every camera."""
  drop: set[int] = set()
  for t in cameras:
    front_keep, rear_keep = landmark_keep[t.name]
    drop.update(
      keys.front_landmark_key(t.name, tid)
      for tid in t.front.points
      if tid not in front_keep
    )
    drop.update(
      keys.rear_landmark_key(t.name, tid)
      for tid in t.rear.points
      if tid not in rear_keep
    )
  return drop


def _solve_with_cull_reoptimize(
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  max_iterations: int,
  gpu: GpuMode,
) -> tuple[OptimizeResult, LandmarkKeep]:
  """Re-solved result and each camera's kept tracks after an outlier-landmark cull."""
  print(f"\njoint graph: {graph.size()} factors, {values.size()} variables")
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

  poses_by_camera = _solved_body_poses(cameras, keys, result.values)
  landmark_keep: LandmarkKeep = {}
  for t in cameras:
    poses_1 = poses_by_camera[t.name]
    f_pts_1 = points_from_values(
      result.values, keys.front_landmark_key, t.name, t.front.points
    )
    r_pts_1 = points_from_values(
      result.values, keys.rear_landmark_key, t.name, t.rear.points
    )
    front_keep = cull_outlier_landmarks(f_pts_1, t.front.tracks, poses_1, t.front.setup)
    rear_keep = cull_outlier_landmarks(r_pts_1, t.rear.tracks, poses_1, t.rear.setup)
    landmark_keep[t.name] = (front_keep, rear_keep)
  first = result
  outlier_keys = _outlier_landmark_keys(cameras, keys, landmark_keep)
  if not outlier_keys and result.iterations >= _PRECULL_MAX_ITERATIONS:
    # Nothing to cull; finish the short ranking solve on the full budget.
    result = optimize_graph(
      graph, result.values, max_iterations=max_iterations, gpu=gpu
    )
  if outlier_keys:
    pruned = _prune_factor_graph(graph, outlier_keys)
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


def _secondary_anchor_seed(
  cameras: list[PreparedCamera],
  tag_id: int,
  body_poses_by_tracker: dict[str, list[gtsam.Pose3]],
) -> gtsam.Pose3 | None:
  """Median ``world_T_body(kf) ∘ body_T_lens ∘ lens_T_tag`` over all PnP sightings."""
  candidates: list[gtsam.Pose3] = []
  for t in cameras:
    poses = body_poses_by_tracker[t.name]
    obs = t.anchor_obs_by_tag.get(tag_id, {})
    for kf in sorted(obs):
      sides = obs[kf]
      for ch in t.lenses:
        lens_T_tag = sides.get(ch.lens)
        if lens_T_tag is None:
          continue
        candidates.append(poses[kf].compose(ch.setup.body_T_lens).compose(lens_T_tag))
  if not candidates:
    return None
  return average_pose(candidates, "median")


def _seed_secondary_anchors(
  cameras: list[PreparedCamera], args: Args
) -> dict[int, gtsam.Pose3]:
  """Initial world pose per sighted non-global anchor, from BFS-aligned prep poses."""
  prep_poses = {t.name: t.body_poses for t in cameras}
  world: dict[int, gtsam.Pose3] = {}
  for tag_id in args.anchor.tag_ids[1:]:
    seed = _secondary_anchor_seed(cameras, tag_id, prep_poses)
    if seed is None:
      print(f"anchor {tag_id}: no PnP sighting; skipped")
      continue
    world[tag_id] = seed
  return world


def _solved_body_poses(
  cameras: list[PreparedCamera], keys: CameraKeys, values: gtsam.Values
) -> dict[str, list[gtsam.Pose3]]:
  return {
    t.name: [values.atPose3(keys.pose_key(t.name, i)) for i in range(t.n_keyframes)]
    for t in cameras
  }


def _secondary_pose_move(
  pinned: gtsam.Pose3, estimate: gtsam.Pose3
) -> tuple[float, float]:
  """Translation (m) and rotation (rad) separating ``pinned`` from ``estimate``."""
  dt = float(
    np.linalg.norm(
      np.asarray(estimate.translation()) - np.asarray(pinned.translation())
    )
  )
  dr = float(
    np.linalg.norm(gtsam.Rot3.Logmap(pinned.rotation().between(estimate.rotation())))
  )
  return dt, dr


@dataclass
class ReferenceSolve:
  """Everything the export needs from the joint solve."""

  ref_tag_id: int
  ref_anchor: gtsam.Pose3
  result: OptimizeResult
  landmark_keep: LandmarkKeep
  secondary_world: dict[int, gtsam.Pose3]
  body_poses: dict[str, list[gtsam.Pose3]]
  # Per secondary anchor, how far its re-estimate moved off its seed (m, rad).
  secondary_moves: dict[int, tuple[float, float]]

  @property
  def worst_secondary_move(self) -> tuple[float, float]:
    """Largest secondary-anchor translation and rotation move, or NaN with none."""
    if not self.secondary_moves:
      return float("nan"), float("nan")
    moves = self.secondary_moves.values()
    return max(dt for dt, _ in moves), max(dr for _, dr in moves)


def _solve_joint_graph(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  args: Args,
  cross_obs: list[CrossSighting],
  ref_anchor: gtsam.Pose3,
) -> ReferenceSolve:
  """Pin ``ref_anchor``, solve, then re-estimate secondary anchors from solved poses."""
  seed = _seed_secondary_anchors(cameras, args)
  graph, values = assemble_joint_graph(
    cameras,
    keys,
    args,
    cross_obs,
    tag_corners_in_frame(ref_anchor, args.anchor.tag_size),
    seed,
  )
  result, landmark_keep = _solve_with_cull_reoptimize(
    graph, values, cameras, keys, args.solver.max_iterations, args.solver.gpu
  )

  solved_poses = _solved_body_poses(cameras, keys, result.values)
  resolved: dict[int, gtsam.Pose3] = {}
  moves: dict[int, tuple[float, float]] = {}
  for tag_id in seed:
    estimate = _secondary_anchor_seed(cameras, tag_id, solved_poses)
    assert estimate is not None  # seeding required a sighting; obs never shrink
    dt, dr = _secondary_pose_move(seed[tag_id], estimate)
    weak = dt > _ANCHOR_CONSISTENCY_TRANS or dr > _ANCHOR_CONSISTENCY_ROT
    note = "  WARNING: weak sighting geometry this take" if weak else ""
    print(
      f"  secondary anchor {tag_id}: post-solve re-estimate moved "
      f"{dt * 1000:.1f}mm / {np.degrees(dr):.2f}deg{note}"
    )
    resolved[tag_id] = estimate
    moves[tag_id] = (dt, dr)
  return ReferenceSolve(
    ref_tag_id=args.anchor.tag_ids[0],
    ref_anchor=ref_anchor,
    result=result,
    landmark_keep=landmark_keep,
    secondary_world=resolved,
    body_poses=solved_poses,
    secondary_moves=moves,
  )


def solve_joint(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  args: Args,
  cross_obs: list[CrossSighting],
) -> ReferenceSolve:
  """Solve the joint graph with the first enabled tag pinning the world."""
  ref_anchor = align_to_shared_world(cameras, args.anchor.tag_ids)
  return _solve_joint_graph(cameras, keys, args, cross_obs, ref_anchor)


def adopt_ba_refined_estimates(
  cameras: list[PreparedCamera], result: OptimizeResult, keys: CameraKeys
) -> None:
  """Store each camera's solved base-tag pose and joint angles."""
  for t in cameras:
    for finger_idx, joint_name in enumerate(FINGER_JOINTS):
      angles = [float("nan")] * t.n_keyframes
      for kf in range(t.n_keyframes):
        tkey = keys.theta_key(t.name, finger_idx, kf)
        if result.values.exists(tkey):
          angles[kf] = float(result.values.atDouble(tkey))
      t.joint_angles[joint_name] = angles
    bt_key = keys.base_tag_key(t.name)
    if result.values.exists(bt_key):
      t.body_T_base_tag = result.values.atPose3(bt_key)
