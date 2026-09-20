"""The VI graph builder and the LM wrapper over small synthetic scenes."""

from __future__ import annotations

import gtsam
import numpy as np
import pytest
from gtsam.symbol_shorthand import B, V, X

from specialist.vi_ba import gpu_lm
from specialist.vi_ba.build import LensChannel, LensSetup, build_vi_graph
from specialist.vi_ba.gpu_lm import optimize_graph
from specialist.vi_ba.imu import (
  ImuSamples,
  integrate_samples,
  make_preintegration_params,
)
from tests.mei_fixtures import (
  WIDE_CALIBRATED_RADIUS,
  pinhole_mei_params,
  pinned_visual_graph,
  single_camera_keys,
)

_GRAVITY = np.array([0.0, 0.0, -9.81])
_LENS = LensSetup(
  body_T_lens=gtsam.Pose3(),
  params=pinhole_mei_params(),
  calibrated_radius=WIDE_CALIBRATED_RADIUS,
)


def _imu_only_graph(
  body_poses: list[gtsam.Pose3],
  velocities: list[np.ndarray],
  biases: list[gtsam.imuBias.ConstantBias],
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements],
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """Visual-inertial graph with no observed landmarks (IMU chain only)."""
  return build_vi_graph(
    body_poses=body_poses,
    velocities=velocities,
    biases=biases,
    preintegrated=preintegrated,
    front=LensChannel(lens="front", setup=_LENS),
    sigma=1.0,
    velocity_prior_sigma=0.1,
    **single_camera_keys(),  # type: ignore[arg-type]
  )


def test_optimize_graph_rejects_non_finite_initial_error_before_solving(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  poses = [gtsam.Pose3(), gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(np.nan, 0.0, 0.0))]
  graph, values = pinned_visual_graph(poses, _LENS, {}, {})
  assert not np.isfinite(graph.error(values))

  def _no_solve(*args: object, **kwargs: object) -> None:
    raise AssertionError("the optimizer must not run on a non-finite graph")

  monkeypatch.setattr(gpu_lm, "_Cudss", _no_solve)
  with pytest.raises(RuntimeError, match="not finite"):
    optimize_graph(graph, values, max_iterations=50)


def _stationary_pim() -> gtsam.PreintegratedCombinedMeasurements:
  """A 1 s stationary IMU preintegration: zero motion, gravity along −z."""
  n = 1000
  t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
  samples = ImuSamples(
    t=t,
    gyro=np.zeros((n, 3), dtype=np.float64),
    accel=np.tile(-_GRAVITY, (n, 1)),
  )
  p = make_preintegration_params(
    gyro_noise_sigma=1e-4,
    accel_noise_sigma=1e-3,
    gyro_bias_rw_sigma=1e-5,
    accel_bias_rw_sigma=1e-4,
    gravity_world=(float(_GRAVITY[0]), float(_GRAVITY[1]), float(_GRAVITY[2])),
  )
  bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
  return integrate_samples(p, samples, bias)


def test_build_vi_graph_adds_velocity_bias_imu_factors() -> None:
  poses = [gtsam.Pose3(), gtsam.Pose3()]
  vel = [np.zeros(3), np.zeros(3)]
  bias = [
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
  ]
  pim = [_stationary_pim()]

  graph, values = _imu_only_graph(poses, vel, bias, pim)
  # 2 initial priors (V0, B0) + 1 IMU factor = 3 factors; no projection.
  assert graph.size() == 3
  for i in range(2):
    assert values.exists(X(i))
    assert values.exists(V(i))
    assert values.exists(B(i))


def test_build_vi_graph_stationary_lm_converges() -> None:
  """2 poses, one IMU factor and no visual: LM keeps the state near-stationary."""
  poses = [gtsam.Pose3(), gtsam.Pose3()]
  vel = [np.zeros(3), np.array([0.05, 0.0, 0.0])]  # slight init perturbation
  bias = [
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
  ]
  pim = [_stationary_pim()]
  graph, values = _imu_only_graph(poses, vel, bias, pim)
  graph.push_back(
    gtsam.PriorFactorPose3(X(0), poses[0], gtsam.noiseModel.Isotropic.Sigma(6, 1e-4))
  )
  initial_err = graph.error(values)
  result = optimize_graph(graph, values, max_iterations=20)
  assert result.final_error <= initial_err
  v1 = np.asarray(result.values.atVector(V(1)))
  assert np.linalg.norm(v1) < 0.03
