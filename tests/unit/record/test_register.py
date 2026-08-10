"""Registration guards: recording ids and layer attachment."""

from pathlib import Path

import pytest
import rerun as rr
from scripts.register import check_layers_attach, dataset_layers, recording_id


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
  base_ids = {recording_id(base)}
  check_layers_attach(base_ids, {good: recording_id(good)})
  with pytest.raises(ValueError, match="orphan.rrd"):
    check_layers_attach(base_ids, {orphan: recording_id(orphan)})


def test_dataset_layers_discovers_by_directory(tmp_path: Path) -> None:
  _write_rrd((tmp_path / "base").mkdir() or tmp_path / "base" / "s1.rrd", "s1")
  _write_rrd((tmp_path / "track").mkdir() or tmp_path / "track" / "s1.rrd", "s1")
  _write_rrd((tmp_path / "mocap").mkdir() or tmp_path / "mocap" / "s1.rrd", "s1")
  (tmp_path / "blueprints").mkdir()  # no rrds: not a layer
  bases, layers = dataset_layers(tmp_path)
  assert [p.name for p in bases] == ["s1.rrd"]
  assert sorted(layers) == ["mocap", "track"]


def test_dataset_layers_requires_a_base(tmp_path: Path) -> None:
  (tmp_path / "track").mkdir()
  _write_rrd(tmp_path / "track" / "s1.rrd", "s1")
  with pytest.raises(SystemExit, match="base layer"):
    dataset_layers(tmp_path)
