"""Parse an Insta360 ``.insv`` file via ``telemetry-parser``."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import telemetry_parser

__all__ = ["InsvData", "load_insv"]


@dataclass(frozen=True)
class InsvData:
  """Parsed ``.insv`` data in SI units on the video clock."""

  imu_timestamps: npt.NDArray[np.float64]
  gyro: npt.NDArray[np.float64]
  accel: npt.NDArray[np.float64]
  rolling_shutter_readout: float
  offset_v3: npt.NDArray[np.float64]
  gyro_calib_numbers: npt.NDArray[np.float64]
  # Every other metadata field the recording keeps.
  capture: dict[str, Any]
  exposure_timestamps: npt.NDArray[np.float64]
  exposure_durations: npt.NDArray[np.float64]
  ae_timestamps: npt.NDArray[np.float64]
  ae_stats: dict[str, npt.NDArray[np.float64]]


_PARSED_FIELDS = frozenset({"offset_v3", "gyro_calib"})
_SUPERSEDED = frozenset(
  {
    "offset",
    "offset_v2",
    "original_offset",
    "original_offset_v2",
    "original_offset_v3",
    "original_offset_3d",
    "rolling_shutter_time",
  }
)
_CONTAINER_DERIVED = frozenset(
  {
    "dimension",
    "resolution_size",
    "frame_rate",
    "total_time",
    "total_frames",
    "file_size",
  }
)
_ZERO_MEANS_UNSET = frozenset({"creation_time", "export_time"})


def capture_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
  """The kept metadata, flattened to scalars, strings and float arrays."""
  kept: dict[str, Any] = {}

  def put(key: str, value: Any) -> None:
    if value is None or value == "" or value == [] or value == {}:
      return
    if isinstance(value, dict):
      for sub, nested in value.items():
        put(f"{key}_{sub}", nested)
      return
    if isinstance(value, (list, tuple)):
      arr = np.asarray(value)
      if arr.dtype.kind in "if":
        kept[key] = arr.astype(np.float64)
      else:
        kept[key] = str(value)
      return
    kept[key] = value if isinstance(value, (bool, int, float)) else str(value)

  for key, value in metadata.items():
    if key in _PARSED_FIELDS or key in _SUPERSEDED or key in _CONTAINER_DERIVED:
      continue
    if key in _ZERO_MEANS_UNSET and not value:
      continue
    put(key, value)
  return kept


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
  default_meta = meta["Default"]["Metadata"]
  exposure = meta.get("Exposure", {}).get("Data", [])
  expo_t = np.array([e["t"] for e in exposure], dtype=np.float64)
  expo_v = np.array([e["v"] for e in exposure], dtype=np.float64)
  expo_order = np.argsort(expo_t, kind="stable")

  # The first frame's boot time moves AE reports from the boot to the video clock.
  reports = meta["Default"].get("AAAData", [])
  first_frame = float(default_meta.get("first_frame_timestamp") or 0.0) * 1e-6
  ae_t = np.empty(0)
  ae_stats: dict[str, npt.NDArray[np.float64]] = {}
  if reports and first_frame > 0.0:
    ae_t = np.array([r["t"] for r in reports], dtype=np.float64) * 1e-3 - first_frame
    ae_order = np.argsort(ae_t, kind="stable")
    ae_t = ae_t[ae_order]
    ae_stats = {
      name: np.array([float(r["v"][name]) for r in reports], dtype=np.float64)[ae_order]
      for name in sorted(reports[0]["v"])
    }

  return InsvData(
    imu_timestamps=ts,
    gyro=np.radians(gyro),
    accel=accel,
    rolling_shutter_readout=float(lens["frame_readout_time"]) * 1e-3,
    offset_v3=np.asarray(default_meta["offset_v3"], dtype=np.float64),
    gyro_calib_numbers=np.asarray(
      default_meta.get("gyro_calib", {}).get("numbers", [0.0] * 6), dtype=np.float64
    ),
    capture=capture_metadata(default_meta),
    exposure_timestamps=expo_t[expo_order],
    exposure_durations=expo_v[expo_order],
    ae_timestamps=ae_t,
    ae_stats=ae_stats,
  )
