"""IMU trajectory between keyframes, each interval's end mismatch spread across it."""

import gtsam
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from specialist.vi_ba.geometry import matrices_to_quats
from specialist.vi_ba.imu import GRAVITY_WORLD, ImuSamples

Float64Array = npt.NDArray[np.float64]

_GRAVITY = np.array(GRAVITY_WORLD)


def _cumulative_products(deltas: Float64Array) -> Float64Array:
  """``out[j] = deltas[0] @ ... @ deltas[j]`` over ``(n, 3, 3)`` rotations."""
  out = deltas.copy()
  step = 1
  while step < len(out):
    out[step:] = out[:-step] @ out[step:]
    step *= 2
  return out


def _integrate(
  rotation: Float64Array,
  velocity: Float64Array,
  position: Float64Array,
  dts: Float64Array,
  gyro: Float64Array,
  accel: Float64Array,
) -> tuple[Float64Array, Float64Array, Float64Array]:
  """Positions, rotations and velocities after each sample, from one start state."""
  steps = Rotation.from_rotvec(gyro * dts[:, None]).as_matrix()
  after = rotation @ _cumulative_products(steps)
  before = np.concatenate([rotation[None], after[:-1]])
  accel_world = np.einsum("nij,nj->ni", before, accel) + _GRAVITY
  dv = accel_world * dts[:, None]
  velocity_before = velocity + np.concatenate(
    [np.zeros((1, 3)), np.cumsum(dv, axis=0)[:-1]]
  )
  positions = position + np.cumsum(
    velocity_before * dts[:, None] + 0.5 * accel_world * dts[:, None] ** 2, axis=0
  )
  return positions, after, velocity_before + dv


def propagate_highrate(
  body_poses: list[gtsam.Pose3],
  keyframe_times: Float64Array,
  velocities: list[Float64Array],
  biases: list[gtsam.imuBias.ConstantBias],
  imu: ImuSamples,
) -> tuple[Float64Array, Float64Array, Float64Array, Float64Array]:
  """``(times, positions, xyzw quaternions, velocities)`` at the IMU timestamps
  between the first and last keyframe, pinned to every keyframe."""
  kf_t = np.asarray(keyframe_times, dtype=np.float64)
  kf_R = np.stack([np.asarray(p.rotation().matrix()) for p in body_poses])
  kf_p = np.stack([np.asarray(p.translation()) for p in body_poses])
  kf_v = np.stack([np.asarray(v, dtype=np.float64) for v in velocities])
  out_t: list[Float64Array] = []
  out_p: list[Float64Array] = []
  out_q: list[Float64Array] = []
  out_v: list[Float64Array] = []
  bounds = np.searchsorted(imu.t, kf_t, side="right")
  for k in range(len(kf_t) - 1):
    inside = np.arange(bounds[k], bounds[k + 1])
    if len(inside) == 0:
      continue
    gyro_bias = np.asarray(biases[k].gyroscope(), dtype=np.float64)
    accel_bias = np.asarray(biases[k].accelerometer(), dtype=np.float64)
    sample_t = imu.t[inside]
    dts = np.diff(np.concatenate(([kf_t[k]], sample_t)))
    seg_p, seg_R, seg_v = _integrate(
      kf_R[k],
      kf_v[k],
      kf_p[k],
      dts,
      imu.gyro[inside] - gyro_bias,
      imu.accel[inside] - accel_bias,
    )
    # The step from the last sample to the next keyframe, from the last state.
    position, rotation, velocity = seg_p[-1], seg_R[-1], seg_v[-1]
    dt = float(kf_t[k + 1] - sample_t[-1])
    accel_world = rotation @ (imu.accel[inside[-1]] - accel_bias) + _GRAVITY
    end_p = position + velocity * dt + 0.5 * accel_world * dt * dt
    end_R = (
      rotation
      @ Rotation.from_rotvec((imu.gyro[inside[-1]] - gyro_bias) * dt).as_matrix()
    )
    gap_p = kf_p[k + 1] - end_p
    gap_w = Rotation.from_matrix(end_R.T @ kf_R[k + 1]).as_rotvec()
    gap_v = kf_v[k + 1] - (velocity + accel_world * dt)
    fraction = ((sample_t - kf_t[k]) / (kf_t[k + 1] - kf_t[k]))[:, None]
    out_t.append(sample_t)
    out_p.append(seg_p + fraction * gap_p)
    out_q.append(
      matrices_to_quats(seg_R @ Rotation.from_rotvec(fraction * gap_w).as_matrix())
    )
    out_v.append(seg_v + fraction * gap_v)
  if not out_t:
    raise ValueError(
      "no IMU samples fall between the keyframes; the streams do not overlap"
    )
  return (
    np.concatenate(out_t),
    np.concatenate(out_p),
    np.concatenate(out_q),
    np.concatenate(out_v),
  )
