"""IMU trajectory between keyframes, each interval's end mismatch spread across it."""

from __future__ import annotations

import gtsam
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from specialist.vi_ba.geometry import matrices_to_quats
from specialist.vi_ba.imu import GRAVITY_WORLD, ImuSamples

Float64Array = npt.NDArray[np.float64]

_GRAVITY = np.array(GRAVITY_WORLD)


def _exp_so3(w: Float64Array) -> Float64Array:
  angle = float(np.linalg.norm(w))
  if angle < 1e-12:
    return np.eye(3)
  axis = w / angle
  skew = np.array(
    [
      [0.0, -axis[2], axis[1]],
      [axis[2], 0.0, -axis[0]],
      [-axis[1], axis[0], 0.0],
    ]
  )
  out: Float64Array = (
    np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew
  )
  return out


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
    rotation = kf_R[k].copy()
    velocity = np.asarray(velocities[k], dtype=np.float64).copy()
    position = kf_p[k].copy()
    gyro_bias = np.asarray(biases[k].gyroscope(), dtype=np.float64)
    accel_bias = np.asarray(biases[k].accelerometer(), dtype=np.float64)
    prev_t = kf_t[k]
    seg_p = np.empty((len(inside), 3))
    seg_R = np.empty((len(inside), 3, 3))
    seg_v = np.empty((len(inside), 3))
    for row, j in enumerate(inside):
      dt = float(imu.t[j] - prev_t)
      prev_t = float(imu.t[j])
      accel_world = rotation @ (imu.accel[j] - accel_bias) + _GRAVITY
      position = position + velocity * dt + 0.5 * accel_world * dt * dt
      velocity = velocity + accel_world * dt
      rotation = rotation @ _exp_so3((imu.gyro[j] - gyro_bias) * dt)
      seg_p[row] = position
      seg_R[row] = rotation
      seg_v[row] = velocity
    dt = float(kf_t[k + 1] - prev_t)
    accel_world = rotation @ (imu.accel[inside[-1]] - accel_bias) + _GRAVITY
    end_p = position + velocity * dt + 0.5 * accel_world * dt * dt
    end_R = rotation @ _exp_so3((imu.gyro[inside[-1]] - gyro_bias) * dt)
    gap_p = kf_p[k + 1] - end_p
    gap_w = Rotation.from_matrix(end_R.T @ kf_R[k + 1]).as_rotvec()
    gap_v = kf_v[k + 1] - (velocity + accel_world * dt)
    fraction = ((imu.t[inside] - kf_t[k]) / (kf_t[k + 1] - kf_t[k]))[:, None]
    out_t.append(imu.t[inside])
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
