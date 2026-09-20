from __future__ import annotations

import gtsam
import numpy as np

from specialist.vi_ba.highrate import propagate_highrate
from specialist.vi_ba.imu import ImuSamples

_GRAVITY = np.array([0.0, 0.0, -9.81])


def _constant_accel_setup(a_world: np.ndarray, hz: float = 1000.0, kf_dt: float = 0.1):
  """Keyframes on a constant-acceleration path and the IMU that produced it."""
  kf_times = np.arange(0.0, 1.0 + 1e-9, kf_dt)
  poses = [
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(*(0.5 * a_world * t * t))) for t in kf_times
  ]
  velocities = [a_world * t for t in kf_times]
  biases = [gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)) for _ in kf_times]
  t = np.arange(0.0, 1.0, 1.0 / hz) + 0.5 / hz
  accel = np.tile(a_world - _GRAVITY, (len(t), 1))  # specific force at R = identity
  imu = ImuSamples(t=t, gyro=np.zeros((len(t), 3)), accel=accel)
  return poses, kf_times, velocities, biases, imu


def test_propagation_reproduces_a_constant_acceleration_path() -> None:
  a = np.array([0.3, -0.2, 0.1])
  poses, kf_times, velocities, biases, imu = _constant_accel_setup(a)
  t, pos, quat, vel = propagate_highrate(poses, kf_times, velocities, biases, imu)
  assert len(t) == len(pos) == len(quat) == len(vel)
  expected = 0.5 * a[None, :] * t[:, None] ** 2
  assert np.abs(pos - expected).max() < 1e-6
  assert np.abs(vel - a[None, :] * t[:, None]).max() < 1e-6
  assert np.allclose(np.linalg.norm(quat, axis=1), 1.0)


def test_a_wrong_bias_is_absorbed_by_the_keyframe_correction() -> None:
  a = np.array([0.2, 0.0, 0.0])
  poses, kf_times, velocities, biases, imu = _constant_accel_setup(a)
  biases = [gtsam.imuBias.ConstantBias(np.full(3, 0.05), np.zeros(3)) for _ in kf_times]
  t, pos, _, _ = propagate_highrate(poses, kf_times, velocities, biases, imu)
  for k in range(len(kf_times) - 1):
    at_kf = pos[np.argmin(np.abs(t - kf_times[k + 1]))]
    truth = 0.5 * a * kf_times[k + 1] ** 2
    assert np.linalg.norm(at_kf - truth) < 1e-3


def test_scan_integration_matches_sample_by_sample() -> None:
  """Under a spinning, accelerating, biased motion they agree to rounding."""
  from specialist.vi_ba.highrate import _integrate

  rng = np.random.default_rng(2)
  n = 400
  dts = np.full(n, 0.001)
  gyro = np.array([0.5, -1.2, 2.0]) + rng.normal(0, 0.01, (n, 3))
  accel = np.array([0.3, 0.1, 9.9]) + rng.normal(0, 0.05, (n, 3))
  R0 = gtsam.Rot3.Ypr(0.3, 0.2, 0.1).matrix()
  v0, p0 = np.array([0.1, 0.2, 0.0]), np.array([1.0, 2.0, 3.0])
  seg_p, seg_R, seg_v = _integrate(R0, v0, p0, dts, gyro, accel)

  rotation, velocity, position = R0.copy(), v0.copy(), p0.copy()
  from scipy.spatial.transform import Rotation

  for j in range(n):
    accel_world = rotation @ accel[j] + _GRAVITY
    position = position + velocity * dts[j] + 0.5 * accel_world * dts[j] ** 2
    velocity = velocity + accel_world * dts[j]
    rotation = rotation @ Rotation.from_rotvec(gyro[j] * dts[j]).as_matrix()
    np.testing.assert_allclose(seg_p[j], position, atol=1e-10)
    np.testing.assert_allclose(seg_R[j], rotation, atol=1e-12)
    np.testing.assert_allclose(seg_v[j], velocity, atol=1e-10)
