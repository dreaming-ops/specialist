"""The joint solve with a landmark cull-reoptimize pass, read back for export."""

from __future__ import annotations

from dataclasses import dataclass, replace

import gtsam
import numpy as np

from specialist.vi_ba._mei_factor_cpp import prune_factors
from specialist.vi_ba.compose import (
  CameraKeys,
  CrossSighting,
  LandmarkKeep,
  align_to_shared_world,
  assemble_joint_graph,
  points_from_values,
)
from specialist.vi_ba.configs import Args
from specialist.vi_ba.gpu_lm import OptimizeResult, optimize_graph
from specialist.vi_ba.landmark_cull import cull_outlier_landmarks
from specialist.vi_ba.single_camera import PreparedCamera
from specialist.vi_ba.urdf_joints import FINGER_JOINTS, RevoluteJoint

__all__ = [
  "ReferenceSolve",
  "adopt_ba_refined_estimates",
  "solve_joint",
]

# The first solve only ranks residuals for the cull.
_PRECULL_MAX_ITERATIONS = 8


def _outlier_landmark_keys(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  landmark_keep: LandmarkKeep,
) -> set[int]:
  """The landmark keys the cull dropped, over every camera's front lens."""
  drop: set[int] = set()
  for camera in cameras:
    keep = landmark_keep[camera.slot]
    drop.update(
      keys.front_landmark_key(camera.slot, tid)
      for tid in camera.front.points
      if tid not in keep
    )
  return drop


def _solve_with_cull_reoptimize(
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  max_iterations: int,
) -> tuple[OptimizeResult, LandmarkKeep]:
  """Solve, cull outlier landmarks and re-solve warm; return the kept tracks too."""
  print(f"\n[graph] joint graph: {graph.size()} factors, {values.size()} variables")
  result = optimize_graph(
    graph,
    values,
    max_iterations=min(max_iterations, _PRECULL_MAX_ITERATIONS),
  )
  poses_by_camera = _solved_body_poses(cameras, keys, result.values)
  landmark_keep: LandmarkKeep = {}
  for camera in cameras:
    points = points_from_values(result.values, keys, camera.slot, camera.front.points)
    landmark_keep[camera.slot] = cull_outlier_landmarks(
      points, camera.front.tracks, poses_by_camera[camera.slot], camera.front.setup
    )
  first = result
  outlier_keys = _outlier_landmark_keys(cameras, keys, landmark_keep)
  if outlier_keys:
    kept_values = gtsam.Values(result.values)
    for key in outlier_keys:
      kept_values.erase(key)
    result = optimize_graph(
      prune_factors(graph, np.fromiter(outlier_keys, dtype=np.int64)),
      kept_values,
      max_iterations=max_iterations,
      initial_lambda=first.final_lambda,
    )
    print(f"  [solve] cull-reoptimize dropped {len(outlier_keys)} outlier landmarks")
  elif result.iterations >= _PRECULL_MAX_ITERATIONS:
    result = optimize_graph(
      graph,
      result.values,
      max_iterations=max_iterations,
      initial_lambda=first.final_lambda,
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
    camera.slot: [
      values.atPose3(keys.pose_key(camera.slot, i)) for i in range(camera.n_keyframes)
    ]
    for camera in cameras
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
    graph, values, cameras, keys, args.solver.max_iterations
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
  cameras: list[PreparedCamera],
  result: OptimizeResult,
  keys: CameraKeys,
  joints: dict[str, RevoluteJoint],
) -> None:
  """Store each camera's solved base-tag pose and joint angles; NaN past a limit."""
  for camera in cameras:
    for finger_idx, joint_name in enumerate(FINGER_JOINTS):
      angles = [float("nan")] * camera.n_keyframes
      joint = joints[joint_name]
      for kf in range(camera.n_keyframes):
        tkey = keys.theta_key(camera.slot, finger_idx, kf)
        if result.values.exists(tkey):
          theta = float(result.values.atDouble(tkey))
          angles[kf] = theta if joint.lower <= theta <= joint.upper else float("nan")
      camera.joint_angles[joint_name] = angles
    bt_key = keys.base_tag_key(camera.slot)
    if result.values.exists(bt_key):
      camera.body_T_base_tag = result.values.atPose3(bt_key)
