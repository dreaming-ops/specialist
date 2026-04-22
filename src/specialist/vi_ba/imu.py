"""IMU samples, stationarity detection and GTSAM combined preintegration."""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]

_INITIAL_BIAS_SIGMA = 1e-3

__all__ = [
  "ImuSamples",
  "factory_imu_bias",
  "find_stationary_windows",
  "integrate_samples",
  "keyframe_delta_rotations",
  "make_preintegration_params",
  "preintegrate_between_keyframes",
  "slice_samples_at_keyframes",
]


@dataclass(frozen=True)
class ImuSamples:
  """Aligned IMU rows in SI units: ``t`` (s), ``gyro`` (rad/s), ``accel`` (m/s²)."""

  t: Float64Array
  gyro: Float64Array
  accel: Float64Array

  def __len__(self) -> int:
    return int(self.t.shape[0])


def find_stationary_windows(
  samples: ImuSamples,
  bias: gtsam.imuBias.ConstantBias,
) -> list[tuple[int, int]]:
  """Stationary ``(start, end)`` ranges from gyro magnitude and accel variance alone."""
  window_size = 50
  gyro_threshold = 0.1
  accel_var_threshold = 0.05
  n = len(samples)
  if n < window_size:
    return []
  accel_corr = samples.accel - bias.accelerometer()
  gyro_corr = samples.gyro - bias.gyroscope()
  windows: list[tuple[int, int]] = []
  for start in range(0, n - window_size + 1, window_size):
    end = start + window_size
    if np.max(np.linalg.norm(gyro_corr[start:end], axis=1)) > gyro_threshold:
      continue
    if float(accel_corr[start:end].var(axis=0).max()) > accel_var_threshold:
      continue
    windows.append((start, end))
  return windows


def make_preintegration_params(
  gyro_noise_sigma: float,
  accel_noise_sigma: float,
  gyro_bias_rw_sigma: float,
  accel_bias_rw_sigma: float,
  gravity_world: tuple[float, float, float],
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
  params.setBiasAccOmegaInit(np.eye(6) * _INITIAL_BIAS_SIGMA**2)
  return params


def factory_imu_bias(
  gyro_calib_numbers: Float64Array,
) -> gtsam.imuBias.ConstantBias:
  """Factory IMU bias from ``gyro_calib.numbers``: gyro rad/s, then accel m/s²."""
  arr = np.asarray(gyro_calib_numbers, dtype=np.float64).reshape(6)
  return gtsam.imuBias.ConstantBias(arr[3:6], arr[:3])


def keyframe_delta_rotations(
  samples: ImuSamples,
  keyframe_times: Float64Array,
) -> list[Float64Array]:
  """Each keyframe's gyro body rotation over ``(t_{k-1}, t_k]``; entry 0 is identity."""
  deltas: list[Float64Array] = [np.eye(3)]
  for window in slice_samples_at_keyframes(samples, keyframe_times):
    delta = np.eye(3)
    for j in range(len(window) - 1):
      dt = float(window.t[j + 1] - window.t[j])
      if dt > 0.0:
        delta = delta @ gtsam.Rot3.Expmap(window.gyro[j] * dt).matrix()
    deltas.append(delta)
  return deltas


def slice_samples_at_keyframes(
  samples: ImuSamples,
  keyframe_times: Float64Array,
) -> list[ImuSamples]:
  """``samples`` split per keyframe pair, window ``i`` over ``(t_i, t_{i+1}]``."""
  t_kf = np.asarray(keyframe_times, dtype=np.float64)
  out: list[ImuSamples] = []
  for i in range(len(t_kf) - 1):
    mask = (samples.t > t_kf[i]) & (samples.t <= t_kf[i + 1])
    out.append(
      ImuSamples(
        t=samples.t[mask],
        gyro=samples.gyro[mask],
        accel=samples.accel[mask],
      )
    )
  return out


def preintegrate_between_keyframes(
  params: gtsam.PreintegrationCombinedParams,
  samples: ImuSamples,
  keyframe_times: Float64Array,
  bias: gtsam.imuBias.ConstantBias,
) -> list[gtsam.PreintegratedCombinedMeasurements]:
  """One ``PreintegratedCombinedMeasurements`` per consecutive keyframe pair."""
  per_window = slice_samples_at_keyframes(samples, keyframe_times)
  return [integrate_samples(params, w, bias) for w in per_window]


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
