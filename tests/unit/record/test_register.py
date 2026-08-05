"""Registration guards: recording ids and layer attachment."""

from pathlib import Path

import pytest
import rerun as rr
from scripts.register import check_layers_attach, recording_id


def _write_rrd(path: Path, rec_id: str) -> Path:
  stream = rr.RecordingStream("test", recording_id=rec_id, send_properties=False)
  stream.save(str(path))
  stream.log("x", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
  stream.flush()
  return path


def test_recording_id_roundtrip(tmp_path: Path) -> None:
  assert recording_id(_write_rrd(tmp_path / "a.rrd", "sess_a")) == "sess_a"


def test_layers_must_attach(tmp_path: Path) -> None:
  base = _write_rrd(tmp_path / "base.rrd", "sess_a")
  good = _write_rrd(tmp_path / "good.rrd", "sess_a")
  orphan = _write_rrd(tmp_path / "orphan.rrd", "sess_b")
  bases = {base: recording_id(base)}
  check_layers_attach(bases, {good: recording_id(good)})
  with pytest.raises(ValueError, match="orphan.rrd"):
    check_layers_attach(bases, {orphan: recording_id(orphan)})
