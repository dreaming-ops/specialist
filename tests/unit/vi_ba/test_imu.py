"""IMU preintegration between keyframes and the stationary-window finder."""

from __future__ import annotations

import gtsam
import numpy as np

from specialist.vi_ba.imu import (
  ImuSamples,
  factory_imu_bias,
  find_stationary_windows,
  integrate_samples,
  make_preintegration_params,
  preintegrate_between_keyframes,
  slice_samples_at_keyframes,
  stationary_mask,
)

_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)
_ZERO_BIAS = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))


def test_constant_gyro_yields_correct_rotation() -> None:
  # Constant 1 rad/s around body z for 0.5 s → ΔR ≈ Rz(0.5 rad).
  n = 500
  t = np.linspace(0.0, 0.5, n, endpoint=False, dtype=np.float64)
  gyro = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float64)
  # Accel = -gravity in body (body aligned with world).
  accel = np.tile(-_GRAVITY, (n, 1)).astype(np.float64)
  samples = ImuSamples(t=t, gyro=gyro, accel=accel)

  params = make_preintegration_params(
    gyro_noise_sigma=1e-4,
    accel_noise_sigma=1e-4,
    gyro_bias_rw_sigma=1e-5,
    accel_bias_rw_sigma=1e-5,
    gravity_world=tuple(_GRAVITY),
  )
  bias = _ZERO_BIAS
  pim = integrate_samples(params, samples, bias)

  omega = gtsam.Rot3.Logmap(pim.deltaRij())
  # 499 intervals of 1 ms integrate 0.499 s, on top of GTSAM's own integration error.
  assert abs(omega[2] - 0.499) < 5e-4
  assert abs(omega[0]) < 1e-3
  assert abs(omega[1]) < 1e-3


def test_factory_imu_bias_decodes_metadata_layout() -> None:
  """Factory layout: 6 floats, gyro biases (rad/s) first, accel (m/s²) last."""
  numbers = np.array([0.01, -0.02, 0.03, -0.04, 0.05, -0.06])
  bias = factory_imu_bias(numbers)
  assert np.allclose(bias.gyroscope(), [0.01, -0.02, 0.03])
  assert np.allclose(bias.accelerometer(), [-0.04, 0.05, -0.06])


def test_slice_samples_at_keyframes_partitions_evenly() -> None:
  # 1 kHz samples from t = 0 to t = 3 s, keyframes every 1 s.
  n = 3000
  t = np.linspace(0.0, 3.0, n, endpoint=False, dtype=np.float64)
  samples = ImuSamples(t=t, gyro=np.zeros((n, 3)), accel=np.zeros((n, 3)))
  keyframes = np.array([0.0, 1.0, 2.0, 3.0])
  windows = slice_samples_at_keyframes(samples, keyframes)
  assert len(windows) == 3
  for w in windows:
    assert 950 < len(w) < 1050


def test_slice_samples_at_keyframes_empty_window() -> None:
  t = np.array([0.0, 0.5, 1.0])
  samples = ImuSamples(t=t, gyro=np.zeros((3, 3)), accel=np.zeros((3, 3)))
  # Keyframes [2, 3] contain no samples.
  windows = slice_samples_at_keyframes(samples, np.array([2.0, 3.0]))
  assert len(windows) == 1 and len(windows[0]) == 0


def test_preintegrate_between_keyframes_partitions_the_samples() -> None:
  """Window i holds the samples in (t_i, t_{i+1}]; a window past them is empty."""
  n = 3000
  t = np.linspace(0.0, 3.0, n, endpoint=False, dtype=np.float64)
  samples = ImuSamples(t=t, gyro=np.zeros((n, 3)), accel=np.tile(-_GRAVITY, (n, 1)))
  params = make_preintegration_params(gravity_world=tuple(_GRAVITY))
  pims = preintegrate_between_keyframes(
    params, samples, np.array([0.0, 1.0, 2.0, 3.0, 4.0]), _ZERO_BIAS
  )
  np.testing.assert_allclose(
    [pim.deltaTij() for pim in pims], [1.0, 1.0, 1.0, 0.0], atol=3e-3
  )


def test_stationary_family_agree() -> None:
  """The window finder is the mask's predicate expressed as index ranges."""
  rng = np.random.default_rng(20260820)
  n = 512
  gyro = rng.normal(0.0, 0.5, (n, 3))
  gyro[100:250] = rng.normal(0.0, 0.01, (150, 3))
  accel = rng.normal(0.0, 0.4, (n, 3)) + np.array([0.0, 0.0, 9.81])
  accel[100:250] = rng.normal(0.0, 0.005, (150, 3)) + np.array([0.0, 0.0, 9.81])
  samples = ImuSamples(t=np.arange(n) / 1000.0, gyro=gyro, accel=accel)
  bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
  windows = find_stationary_windows(samples, bias)
  mask = stationary_mask(samples, bias)
  assert windows == [(int(i) * 50, int(i) * 50 + 50) for i in np.flatnonzero(mask)]
  assert windows, "the quiet stretch must be found"
