"""Base-recording chunk builders round-trip through a written rrd."""

from pathlib import Path
from typing import Any

import numpy as np
from rerun.experimental import Chunk, RrdReader

from specialist.loader.parse_insv import InsvData, LensIntrinsics
from specialist.record.base import imu_chunks, write_base_recording


def _synthetic_insv() -> InsvData:
  block = [
    1.9,  # xi
    5000.0,
    5001.0,
    2970.0,
    2971.0,  # fx fy cx cy
    0.5,
    -0.3,
    90.0,  # yaw pitch roll deg
    0.01,
    0.02,
    0.03,  # t
    0.1,
    -0.05,
    0.01,
    0.001,
    -0.002,  # k1..k3 p1 p2
    11904.0,
    5952.0,
    0.0,  # frame w h unknown
  ]
  offset_v3 = np.array([2.0, *block, *block, 0.0])
  return InsvData(
    imu_timestamps=np.array([-0.5, 0.0, 0.001, 0.002]),
    gyro=np.radians([[90.0, 0.0, 0.0]] * 4),
    accel=np.array([[0.0, 0.0, 9.81]] * 4),
    intrinsics=LensIntrinsics(
      camera_matrix=np.eye(3),
      distortion_coeffs=np.zeros(6),
      calib_width=11904,
      calib_height=5952,
      distortion_model="mei",
    ),
    rolling_shutter_readout=0.0157,
    offset_v3=offset_v3,
    gyro_calib_numbers=np.array([0.001, 0.002, 0.003, 0.01, 0.02, 0.03]),
    capture={},
    exposure_timestamps=np.empty(0),
    exposure_durations=np.empty(0),
    ae_timestamps=np.empty(0),
    ae_stats={},
  )


def _write(tmp_path: Path, chunks: list[Chunk]) -> RrdReader:
  out = tmp_path / "base.rrd"
  write_base_recording(out, "sess", [], chunks, {})
  return RrdReader(str(out))


def _column(reader: RrdReader, entity: str, component: str) -> tuple[Any, Any]:
  """All (time, value) rows of one component, times None for static."""
  times: list[Any] = []
  values: list[Any] = []
  for chunk in reader.store().stream().to_chunks():
    if chunk.entity_path != entity:
      continue
    batch = chunk.to_record_batch()
    names = batch.schema.names
    if component not in names:
      continue
    col = batch.column(names.index(component))
    if chunk.is_static:
      values += col.to_pylist()
      times += [None] * len(col)
    else:
      timeline = chunk.timeline_names[0]
      times += [v.value for v in batch.column(names.index(timeline))]
      values += col.to_pylist()
  return times, values


def test_imu_chunks_units_and_shift(tmp_path: Path) -> None:
  reader = _write(tmp_path, imu_chunks("tracker_0", _synthetic_insv(), 0.25))
  times, gyro = _column(reader, "/imu/tracker_0/gyro", "Scalars:scalars")
  assert times[0] == round((-0.5 + 0.25) * 1e9)
  np.testing.assert_allclose(gyro[0], [np.pi / 2, 0.0, 0.0])
  _, accel = _column(reader, "/imu/tracker_0/accel", "Scalars:scalars")
  np.testing.assert_allclose(accel[0], [0.0, 0.0, 9.81])


def test_imu_window_trims_rows(tmp_path: Path) -> None:
  reader = _write(
    tmp_path, imu_chunks("tracker_0", _synthetic_insv(), 0.0, window=(0.0, 10.0))
  )
  times, _ = _column(reader, "/imu/tracker_0/gyro", "Scalars:scalars")
  assert times and min(times) >= 0
