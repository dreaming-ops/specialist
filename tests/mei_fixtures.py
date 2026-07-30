"""MEI lens regimes, their projection, and a visual-only graph for the vi_ba tests."""

from __future__ import annotations

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import MeiParams, mei_project
from specialist.vi_ba.build import LensSetup, build_vi_graph


def _undistorted(f: float, cx: float, cy: float, xi: float) -> MeiParams:
  return MeiParams(
    fx=f, fy=f, cx=cx, cy=cy, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0, xi=xi
  )


def pinhole_mei_params() -> MeiParams:
  return _undistorted(500.0, 320.0, 240.0, 0.0)


def narrow_mei_params() -> MeiParams:
  """240x240 undistorted MEI with a mild xi."""
  return _undistorted(160.0, 120.0, 120.0, 0.8)


def distorted_mei_params(xi: float) -> MeiParams:
  """1920x1920 lightly distorted MEI; ``xi`` sets its depth into super-fisheye."""
  return MeiParams(
    fx=800.0, fy=800.0, cx=960.0, cy=960.0,
    k1=0.02, k2=-0.01, k3=0.003, p1=0.001, p2=-0.002, xi=xi,
  )  # fmt: skip


def project_point(
  world_T_body: gtsam.Pose3,
  body_T_lens: gtsam.Pose3,
  world_xyz: np.ndarray,
  params: MeiParams,
) -> np.ndarray:
  world_T_lens = world_T_body.compose(body_T_lens)
  p_lens = world_T_lens.transformTo(gtsam.Point3(*world_xyz))
  return mei_project(np.asarray(p_lens)[None, :], params)[0]


def front_only_graph(
  body_poses: list[gtsam.Pose3],
  front: LensSetup,
  front_tracks: dict,
  front_points: dict,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """Visual-only graph (no IMU, no rear lens), gauge-pinned at both ends."""
  return build_vi_graph(
    body_poses=body_poses,
    preintegrated=None,
    front=front,
    front_tracks=front_tracks,
    front_points=front_points,
    rear=front,
    rear_tracks={},
    rear_points={},
    sigma=1.0,
    weak_pose_sigmas=(0.3,) * 6,
    anchor_last_pose=True,
  )


def unit_pixel_noise() -> gtsam.noiseModel.Base:
  # Isotropic σ=1 px, so whitened == unwhitened.
  return gtsam.noiseModel.Isotropic.Sigma(2, 1.0)


def fd_pose_columns(
  residual_at, pose: gtsam.Pose3, h: float = 1e-6
) -> npt.NDArray[np.float64]:
  """Central finite difference of a residual w.r.t. the Pose3 retract tangent."""
  cols = []
  for i in range(6):
    xi = np.zeros(6)
    xi[i] = h
    r_plus = residual_at(pose.retract(xi))
    r_minus = residual_at(pose.retract(-xi))
    cols.append((r_plus - r_minus) / (2.0 * h))
  return np.column_stack(cols)  # 2x6
