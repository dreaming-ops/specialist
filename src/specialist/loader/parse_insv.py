"""Parse an Insta360 ``.insv`` file via ``telemetry-parser``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import telemetry_parser


@dataclass(frozen=True)
class LensIntrinsics:
  """Factory MEI intrinsics of one fisheye, distortion ``[k1, k2, k3, p1, p2, xi]``."""

  camera_matrix: npt.NDArray[np.float64]
  distortion_coeffs: npt.NDArray[np.float64]
  calib_width: int
  calib_height: int
  distortion_model: str

  @property
  def xi(self) -> float:
    return float(self.distortion_coeffs[5])


@dataclass(frozen=True)
class InsvData:
  """Parsed ``.insv`` data; times in video-clock seconds, ``gyro`` in rad/s."""

  camera: str
  model: str
  imu_timestamps: npt.NDArray[np.float64]
  gyro: npt.NDArray[np.float64]
  accel: npt.NDArray[np.float64]
  intrinsics: LensIntrinsics
  rolling_shutter_readout: float
  offset_v3: npt.NDArray[np.float64]
  gyro_calib_numbers: npt.NDArray[np.float64]


def load_insv(path: Path) -> InsvData:
  """Parse a ``.insv`` file and return the typed view used by the probes."""
  if not path.is_file():
    raise FileNotFoundError(f"insv not found: {path}")

  parser = telemetry_parser.Parser(str(path))
  raw = parser.telemetry()
  if not isinstance(raw, list) or not raw:
    raise ValueError(f"telemetry-parser returned empty telemetry for {path}")
  meta = raw[0]

  imu_samples = parser.normalized_imu()
  n = len(imu_samples)
  if n == 0:
    raise ValueError(f"no IMU samples in {path}")
  ts = np.empty(n)
  gyro = np.empty((n, 3))
  accel = np.empty((n, 3))
  for i, s in enumerate(imu_samples):
    ts[i] = float(s["timestamp_ms"]) * 1e-3
    gx, gy, gz = s["gyro"]
    ax, ay, az = s["accl"]
    gyro[i, 0], gyro[i, 1], gyro[i, 2] = float(gx), float(gy), float(gz)
    accel[i, 0], accel[i, 1], accel[i, 2] = float(ax), float(ay), float(az)

  lens = meta["Lens"]["Data"]
  fisheye = lens["fisheye_params"]
  intrinsics = LensIntrinsics(
    camera_matrix=np.asarray(fisheye["camera_matrix"], dtype=np.float64),
    distortion_coeffs=np.asarray(fisheye["distortion_coeffs"], dtype=np.float64),
    calib_width=int(lens["calib_dimension"]["w"]),
    calib_height=int(lens["calib_dimension"]["h"]),
    distortion_model=str(lens["distortion_model"]),
  )

  default_meta = meta["Default"]["Metadata"]
  return InsvData(
    camera=str(parser.camera),
    model=str(parser.model),
    imu_timestamps=ts,
    gyro=np.radians(gyro),
    accel=accel,
    intrinsics=intrinsics,
    rolling_shutter_readout=float(lens["frame_readout_time"]) * 1e-3,
    offset_v3=np.asarray(default_meta["offset_v3"], dtype=np.float64),
    gyro_calib_numbers=np.asarray(
      default_meta.get("gyro_calib", {}).get("numbers", [0.0] * 6), dtype=np.float64
    ),
  )
