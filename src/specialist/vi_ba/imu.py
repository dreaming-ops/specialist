"""IMU samples, stationarity detection and GTSAM combined preintegration."""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]
Float64Array = npt.NDArray[np.float64]

GRAVITY = 9.81
GRAVITY_WORLD = (0.0, 0.0, -GRAVITY)

# Samples per stationarity window: 50 ms of the ~1 kHz IMU.
_STATIONARY_WINDOW = 50

__all__ = [
  "GRAVITY",
  "GRAVITY_WORLD",
  "ImuSamples",
  "find_stationary_windows",
  "integrate_samples",
  "make_preintegration_params",
  "preintegrate_between_keyframes",
  "stationary_mask",
]


@dataclass(frozen=True)
class ImuSamples:
  """Aligned IMU rows in SI units: ``t`` (s), ``gyro`` (rad/s), ``accel`` (m/s²)."""

  t: Float64Array
  gyro: Float64Array
  accel: Float64Array

  def __len__(self) -> int:
    return int(self.t.shape[0])


def stationary_mask(
  samples: ImuSamples,
  bias: gtsam.imuBias.ConstantBias | None = None,
  *,
  window: int = _STATIONARY_WINDOW,
  gyro_threshold: float = 0.1,
  accel_var_threshold: float = 0.05,
) -> BoolArray:
  """Per ``window``-sample block, whether the peak gyro norm (rad/s) and the accel
  variance stay under their thresholds."""
  blocks = len(samples) // window
  gyro = samples.gyro[: blocks * window]
  accel = samples.accel[: blocks * window]
  if bias is not None:
    gyro, accel = gyro - bias.gyroscope(), accel - bias.accelerometer()
  peak = np.linalg.norm(gyro, axis=1).reshape(blocks, window).max(axis=1)
  variance = accel.reshape(blocks, window, 3).var(axis=1).max(axis=1)
  quiet: BoolArray = (peak <= gyro_threshold) & (variance <= accel_var_threshold)
  return quiet


def find_stationary_windows(
  samples: ImuSamples,
  bias: gtsam.imuBias.ConstantBias,
) -> list[tuple[int, int]]:
  """Ascending ``(start, end)`` ranges of the blocks ``stationary_mask`` passes."""
  quiet = stationary_mask(samples, bias)
  return [
    (int(i) * _STATIONARY_WINDOW, (int(i) + 1) * _STATIONARY_WINDOW)
    for i in np.flatnonzero(quiet)
  ]


# Continuous-time white-noise densities and bias random walks.
_GYRO_NOISE_SIGMA = 0.0125  # rad/s/sqrt(Hz)
_ACCEL_NOISE_SIGMA = 0.01  # m/s^2/sqrt(Hz)
_GYRO_BIAS_RW_SIGMA = 1.0e-7  # rad/s^2/sqrt(Hz)
_ACCEL_BIAS_RW_SIGMA = 1.0e-7  # m/s^3/sqrt(Hz)


def make_preintegration_params(
  gravity_world: tuple[float, float, float],
  *,
  gyro_noise_sigma: float = _GYRO_NOISE_SIGMA,
  accel_noise_sigma: float = _ACCEL_NOISE_SIGMA,
  gyro_bias_rw_sigma: float = _GYRO_BIAS_RW_SIGMA,
  accel_bias_rw_sigma: float = _ACCEL_BIAS_RW_SIGMA,
) -> gtsam.PreintegrationCombinedParams:
  """Combined preintegration params with isotropic noise and bias-walk covariances."""
  params = gtsam.PreintegrationCombinedParams(
    np.asarray(gravity_world, dtype=np.float64)
  )
  params.setAccelerometerCovariance(np.eye(3) * accel_noise_sigma**2)
  params.setGyroscopeCovariance(np.eye(3) * gyro_noise_sigma**2)
  params.setIntegrationCovariance(np.eye(3) * 1e-10)
  params.setBiasAccCovariance(np.eye(3) * accel_bias_rw_sigma**2)
  params.setBiasOmegaCovariance(np.eye(3) * gyro_bias_rw_sigma**2)
  return params


def integrate_samples(
  params: gtsam.PreintegrationCombinedParams,
  samples: ImuSamples,
  bias: gtsam.imuBias.ConstantBias,
) -> gtsam.PreintegratedCombinedMeasurements:
  """Zero-order-hold PIM of ``samples``, skipping non-positive time steps."""
  pim = gtsam.PreintegratedCombinedMeasurements(params, bias)
  dts = np.diff(samples.t)
  for i in range(len(dts)):
    dt = float(dts[i])
    if dt <= 0.0:
      continue
    pim.integrateMeasurement(
      samples.accel[i].astype(np.float64),
      samples.gyro[i].astype(np.float64),
      dt,
    )
  return pim


def preintegrate_between_keyframes(
  params: gtsam.PreintegrationCombinedParams,
  samples: ImuSamples,
  keyframe_times: Float64Array,
  bias: gtsam.imuBias.ConstantBias,
) -> list[gtsam.PreintegratedCombinedMeasurements]:
  """One ``PreintegratedCombinedMeasurements`` per consecutive keyframe pair."""
  # Window i holds the samples in (t_i, t_{i+1}].
  bounds = np.searchsorted(
    samples.t, np.asarray(keyframe_times, np.float64), side="right"
  )
  return [
    integrate_samples(
      params,
      ImuSamples(t=samples.t[a:b], gyro=samples.gyro[a:b], accel=samples.accel[a:b]),
      bias,
    )
    for a, b in zip(bounds[:-1], bounds[1:], strict=True)
  ]
