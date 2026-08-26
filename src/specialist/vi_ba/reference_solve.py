"""The joint solve with a landmark cull-reoptimize pass, read back for export."""

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
from specialist.vi_ba.landmark_cull import cull_outlier_landmarks
from specialist.vi_ba.optimize import OptimizeResult, optimize_graph
from specialist.vi_ba.single_camera import PreparedCamera
from specialist.vi_ba.urdf_joints import FINGER_JOINTS

__all__ = [
  "ReferenceSolve",
  "adopt_ba_refined_estimates",
  "solve_joint",
]

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
  """The landmark keys the cull dropped, over every camera's front lens."""
  drop: set[int] = set()
  for t in cameras:
    keep = landmark_keep[t.name]
    drop.update(
      keys.front_landmark_key(t.name, tid) for tid in t.front.points if tid not in keep
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
  """Solve, cull outlier landmarks and re-solve warm; return the kept tracks too."""
  print(f"\n[graph] joint graph: {graph.size()} factors, {values.size()} variables")
  result = optimize_graph(
    graph,
    values,
    max_iterations=min(max_iterations, _PRECULL_MAX_ITERATIONS),
    gpu=gpu,
  )
  poses_by_camera = _solved_body_poses(cameras, keys, result.values)
  landmark_keep: LandmarkKeep = {}
  for t in cameras:
    poses_1 = poses_by_camera[t.name]
    points = points_from_values(result.values, keys, t.name, t.front.points)
    landmark_keep[t.name] = cull_outlier_landmarks(
      points, t.front.tracks, poses_1, t.front.setup
    )
  first = result
  outlier_keys = _outlier_landmark_keys(cameras, keys, landmark_keep)
  if outlier_keys:
    pruned = _prune_factor_graph(graph, outlier_keys)
    kept_values = gtsam.Values(result.values)
    for key in outlier_keys:
      kept_values.erase(key)
    result = optimize_graph(pruned, kept_values, max_iterations=max_iterations, gpu=gpu)
    print(f"  [solve] cull-reoptimize dropped {len(outlier_keys)} outlier landmarks")
  elif result.iterations >= _PRECULL_MAX_ITERATIONS:
    result = optimize_graph(
      graph, result.values, max_iterations=max_iterations, gpu=gpu
    )
  else:
    return result, landmark_keep
  return (
    replace(
      result,
      initial_error=first.initial_error,
      iterations=first.iterations + result.iterations,
    ),
    landmark_keep,
  )


def _solved_body_poses(
  cameras: list[PreparedCamera], keys: CameraKeys, values: gtsam.Values
) -> dict[str, list[gtsam.Pose3]]:
  return {
    t.name: [values.atPose3(keys.pose_key(t.name, i)) for i in range(t.n_keyframes)]
    for t in cameras
  }


def _anchor_pose_move(seed: gtsam.Pose3, estimate: gtsam.Pose3) -> tuple[float, float]:
  """Translation (m) and rotation (rad) separating ``seed`` from ``estimate``."""
  rel = seed.between(estimate)
  dr = gtsam.Rot3.Logmap(rel.rotation())
  return float(np.linalg.norm(rel.translation())), float(np.linalg.norm(dr))


@dataclass
class ReferenceSolve:
  """Everything the export needs from the joint solve."""

  ref_tag_id: int
  result: OptimizeResult
  landmark_keep: LandmarkKeep
  world_T_anchor: dict[int, gtsam.Pose3]
  body_poses: dict[str, list[gtsam.Pose3]]
  # Per anchor, how far the solve moves it off its seed (meters, radians).
  anchor_moves: dict[int, tuple[float, float]]

  @property
  def worst_anchor_move(self) -> tuple[float, float]:
    """Worst translation and rotation off seed over non-reference anchors, or NaN."""
    moves = [m for tag, m in self.anchor_moves.items() if tag != self.ref_tag_id]
    if not moves:
      return float("nan"), float("nan")
    return max(dt for dt, _ in moves), max(dr for _, dr in moves)


def _solve_joint_graph(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  args: Args,
  cross_obs: list[CrossSighting],
  anchor_seed: dict[int, gtsam.Pose3],
) -> ReferenceSolve:
  """Solve the joint graph and read every anchor's world pose back out of it."""
  graph, values = assemble_joint_graph(cameras, keys, args, cross_obs, anchor_seed)
  result, landmark_keep = _solve_with_cull_reoptimize(
    graph, values, cameras, keys, args.solver.max_iterations, args.solver.gpu
  )

  solved: dict[int, gtsam.Pose3] = {}
  moves: dict[int, tuple[float, float]] = {}
  for tag_id, seed in anchor_seed.items():
    estimate = result.values.atPose3(keys.anchor_key(tag_id))
    dt, dr = _anchor_pose_move(seed, estimate)
    print(f"  [anchor] tag {tag_id} moved {dt * 1000:.1f}mm / {np.degrees(dr):.2f}deg")
    solved[tag_id] = estimate
    moves[tag_id] = (dt, dr)
  return ReferenceSolve(
    ref_tag_id=args.anchor.tag_ids[0],
    result=result,
    landmark_keep=landmark_keep,
    world_T_anchor=solved,
    body_poses=_solved_body_poses(cameras, keys, result.values),
    anchor_moves=moves,
  )


def solve_joint(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  args: Args,
  cross_obs: list[CrossSighting],
) -> ReferenceSolve:
  """Seed every camera in the anchors' shared world, then solve the joint graph."""
  anchor_seed = align_to_shared_world(cameras, args.anchor.tag_ids)
  return _solve_joint_graph(cameras, keys, args, cross_obs, anchor_seed)


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
