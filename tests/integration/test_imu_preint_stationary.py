"""Preintegrating a real stationary IMU clip stays inside a 0.3° rotation budget."""

import math
from pathlib import Path

import gtsam
import numpy as np

from specialist.vi_ba.imu import (
  ImuSamples,
  integrate_samples,
  make_preintegration_params,
)

_FIXTURE = (
  Path(__file__).resolve().parents[1] / "fixtures" / "vi_ba" / "imu_stationary_30s.npz"
)


def _load_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  data = np.load(_FIXTURE)
  return (
    data["t"].astype(np.float64),
    np.deg2rad(data["gyro_deg"].astype(np.float64)),
    data["accel"].astype(np.float64),
  )


def test_stationary_orientation_drift_within_budget() -> None:
  t, gyro, accel = _load_fixture()
  samples = ImuSamples(t=t, gyro=gyro, accel=accel)
  assert len(samples) > 20_000  # 1 kHz × 30 s minus small head/tail

  # The camera sat with body +y up; the world frame puts gravity along −y.
  g_magnitude = float(np.linalg.norm(accel.mean(axis=0)))
  assert 9.6 < g_magnitude < 10.0
  gravity_world = (0.0, -g_magnitude, 0.0)

  # At rest with body == world: gyro bias = mean(gyro), accel bias = mean(accel) + gravity_world.
  seed_mask = samples.t < (samples.t[0] + 3.0)
  bias = gtsam.imuBias.ConstantBias(
    samples.accel[seed_mask].mean(axis=0) + np.array(gravity_world),
    samples.gyro[seed_mask].mean(axis=0),
  )

  tail_mask = samples.t >= (samples.t[0] + 3.0)
  tail = ImuSamples(
    t=samples.t[tail_mask],
    gyro=samples.gyro[tail_mask],
    accel=samples.accel[tail_mask],
  )

  params = make_preintegration_params(
    gyro_noise_sigma=3e-3,  # rad/√s white-noise density
    accel_noise_sigma=1e-2,
    gyro_bias_rw_sigma=1e-4,
    accel_bias_rw_sigma=1e-3,
    gravity_world=gravity_world,
  )
  pim = integrate_samples(params, tail, bias)

  delta_r = pim.deltaRij()
  drift = float(np.linalg.norm(gtsam.Rot3.Logmap(delta_r)))
  drift_deg = math.degrees(drift)
  assert drift_deg < 0.3, f"stationary orientation drift {drift_deg:.3f}° > 0.3°"
