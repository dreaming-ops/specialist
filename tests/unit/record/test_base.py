"""Base-recording chunk builders round-trip through a written rrd."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rerun.chunk import Chunk, RrdReader
from specialist_dataset.read import Recording

from specialist.record.base import imu_chunks, write_base_recording
from tests.record_fixtures import synthetic_insv

# Wide enough to keep every synthetic IMU sample.
_WHOLE_SESSION = (-10.0, 10.0)


def _write(tmp_path: Path, chunks: list[Chunk]) -> Recording:
  out = tmp_path / "base.rrd"
  write_base_recording(out, "sess", [], chunks)
  return Recording(out)


def test_imu_chunks_units_and_shift(tmp_path: Path) -> None:
  recording = _write(
    tmp_path, imu_chunks("cam00", synthetic_insv(), 0.25, _WHOLE_SESSION)
  )
  times, gyro = recording.scalars("camera/cam00/imu/gyro", 3)
  assert times[0] == -0.5 + 0.25
  np.testing.assert_allclose(gyro[0], [np.pi / 2, 0.0, 0.0])
  _, accel = recording.scalars("camera/cam00/imu/accel", 3)
  np.testing.assert_allclose(accel[0], [0.0, 0.0, 9.81])


def test_imu_window_trims_rows(tmp_path: Path) -> None:
  recording = _write(
    tmp_path, imu_chunks("cam00", synthetic_insv(), 0.0, window=(0.0, 10.0))
  )
  times, _ = recording.scalars("camera/cam00/imu/gyro", 3)
  assert times.size and times.min() >= 0


def test_write_base_recording_failure_leaves_no_final_file(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  out = tmp_path / "base.rrd"

  def explode(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("mid-write failure")

  monkeypatch.setattr("rerun.send_chunks", explode)
  chunks = imu_chunks("cam00", synthetic_insv(), 0.0, _WHOLE_SESSION)
  with pytest.raises(RuntimeError, match="mid-write failure"):
    write_base_recording(out, "s", [], chunks)
  assert not out.exists()


def test_write_base_recording_replaces_a_stale_partial(tmp_path: Path) -> None:
  out = tmp_path / "base.rrd"
  stale = tmp_path / "base.rrd.partial"
  stale.write_bytes(b"leftover from a crashed run")
  chunks = imu_chunks("cam00", synthetic_insv(), 0.0, _WHOLE_SESSION)
  write_base_recording(out, "s", [], chunks)
  assert not stale.exists()
  assert RrdReader(str(out)).recordings()[0].recording_id == "s"
