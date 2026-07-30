"""Parse an Insta360 ``.insv`` file via ``telemetry-parser``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import telemetry_parser

__all__ = ["InsvData", "LensIntrinsics", "load_insv"]


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
  """Parsed ``.insv`` data in SI units on the video clock."""

  imu_timestamps: npt.NDArray[np.float64]
  gyro: npt.NDArray[np.float64]
  accel: npt.NDArray[np.float64]
  intrinsics: LensIntrinsics
  rolling_shutter_readout: float
  offset_v3: npt.NDArray[np.float64]
  gyro_calib_numbers: npt.NDArray[np.float64]


def load_insv(path: Path) -> InsvData:
  if not path.is_file():
    raise FileNotFoundError(f"insv not found: {path}")

  parser = telemetry_parser.Parser(str(path))
  raw = parser.telemetry()
  if not isinstance(raw, list) or not raw:
    raise ValueError(f"telemetry-parser returned empty telemetry for {path}")
  meta = raw[0]

  imu_samples = parser.normalized_imu()
  if not imu_samples:
    raise ValueError(f"no IMU samples in {path}")
  ts = np.array([s["timestamp_ms"] for s in imu_samples], dtype=np.float64) * 1e-3
  gyro = np.array([s["gyro"] for s in imu_samples], dtype=np.float64)
  accel = np.array([s["accl"] for s in imu_samples], dtype=np.float64)

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
