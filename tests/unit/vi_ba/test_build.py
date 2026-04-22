"""Dual-lens graph assembly and the LM wrapper over small synthetic scenes."""

from __future__ import annotations

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, V, X

from specialist.loader.mei_model import MeiParams, mei_project
from specialist.vi_ba.build import (
  LensSetup,
  build_vi_graph,
  front_landmark_key,
  rear_landmark_key,
)
from specialist.vi_ba.imu import (
  ImuSamples,
  integrate_samples,
  make_preintegration_params,
)
from specialist.vi_ba.optimize import optimize_graph


def _params() -> MeiParams:
  return MeiParams(
    fx=500.0,
    fy=500.0,
    cx=320.0,
    cy=240.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
    xi=0.0,  # pinhole-like
  )


def _project_point(
  pose_wb: gtsam.Pose3,
  body_T_lens: gtsam.Pose3,
  world_xyz: np.ndarray,
  params: MeiParams,
) -> np.ndarray:
  t_wl = pose_wb.compose(body_T_lens)
  p_lens = t_wl.transformTo(gtsam.Point3(*world_xyz))
  return mei_project(np.asarray(p_lens)[None, :], params)[0]


def test_build_graph_dual_lens_uses_disjoint_landmark_keys() -> None:
  params = _params()
  poses = [gtsam.Pose3(), gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.3, 0.0, 0.0))]
  body_T_lens = gtsam.Pose3()

  xyz_front = np.array([0.1, 0.0, 2.0])
  xyz_rear = np.array([-0.1, 0.0, 2.0])  # behind-world landmark

  front_tracks = {
    0: {
      0: _project_point(poses[0], body_T_lens, xyz_front, params),
      1: _project_point(poses[1], body_T_lens, xyz_front, params),
    }
  }
  rear_tracks = {
    0: {
      0: _project_point(poses[0], body_T_lens, xyz_rear, params),
      1: _project_point(poses[1], body_T_lens, xyz_rear, params),
    }
  }

  lens = LensSetup(body_T_lens=body_T_lens, params=params)
  graph, values = build_vi_graph(
    body_poses=poses,
    preintegrated=None,
    front=lens,
    front_tracks=front_tracks,
    front_points={0: xyz_front},
    rear=lens,
    rear_tracks=rear_tracks,
    rear_points={0: xyz_rear},
    sigma=1.0,
    weak_pose_sigmas=(0.3,) * 6,
    anchor_last_pose=True,
  )
  # Front landmark key != rear landmark key.
  assert front_landmark_key(0) != rear_landmark_key(0)
  assert values.exists(front_landmark_key(0))
  assert values.exists(rear_landmark_key(0))
  # 2 tight priors + 2 front obs + 2 rear obs = 6.
  assert graph.size() == 6


def _stationary_pim(
  duration: float, gravity_world: np.ndarray
) -> gtsam.PreintegratedCombinedMeasurements:
  n = int(duration * 1000)
  t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)
  samples = ImuSamples(
    t=t,
    gyro=np.zeros((n, 3), dtype=np.float64),
    accel=np.tile(-gravity_world, (n, 1)),
  )
  p = make_preintegration_params(
    gyro_noise_sigma=1e-4,
    accel_noise_sigma=1e-3,
    gyro_bias_rw_sigma=1e-5,
    accel_bias_rw_sigma=1e-4,
    gravity_world=(
      float(gravity_world[0]),
      float(gravity_world[1]),
      float(gravity_world[2]),
    ),
  )
  bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
  return integrate_samples(p, samples, bias)


def test_build_vi_graph_adds_velocity_bias_imu_factors() -> None:
  params = _params()
  body_T_lens = gtsam.Pose3()
  gravity = np.array([0.0, 0.0, -9.81])
  poses = [gtsam.Pose3(), gtsam.Pose3()]
  vel = [np.zeros(3), np.zeros(3)]
  bias = [
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
  ]
  pim = [_stationary_pim(1.0, gravity)]

  graph, values = build_vi_graph(
    body_poses=poses,
    velocities=vel,
    biases=bias,
    preintegrated=pim,
    front=LensSetup(body_T_lens=body_T_lens, params=params),
    front_tracks={},
    front_points={},
    rear=None,
    rear_tracks=None,
    rear_points=None,
    sigma=1.0,
  )
  # 3 initial priors (X0, V0, B0) + 1 IMU factor = 4 factors; no projection.
  assert graph.size() == 4
  for i in range(2):
    assert values.exists(X(i))
    assert values.exists(V(i))
    assert values.exists(B(i))


def test_build_vi_graph_stationary_lm_converges() -> None:
  """2 poses, one IMU factor and no visual: LM keeps the state near-stationary."""
  params = _params()
  body_T_lens = gtsam.Pose3()
  gravity = np.array([0.0, 0.0, -9.81])
  poses = [gtsam.Pose3(), gtsam.Pose3()]
  vel = [np.zeros(3), np.array([0.05, 0.0, 0.0])]  # slight init perturbation
  bias = [
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
  ]
  pim = [_stationary_pim(1.0, gravity)]
  graph, values = build_vi_graph(
    body_poses=poses,
    velocities=vel,
    biases=bias,
    preintegrated=pim,
    front=LensSetup(body_T_lens=body_T_lens, params=params),
    front_tracks={},
    front_points={},
    rear=None,
    rear_tracks=None,
    rear_points=None,
    sigma=1.0,
  )
  initial_err = graph.error(values)
  result = optimize_graph(graph, values, max_iterations=20)
  assert result.final_error <= initial_err
  v1 = np.asarray(result.values.atVector(V(1)))
  assert np.linalg.norm(v1) < 0.03
