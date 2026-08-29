"""Registration selects exactly one state configuration."""

from pathlib import Path

import pytest
import rerun as rr
from scripts.catalog import state_files


def _write_rrd(path: Path, rec_id: str) -> Path:
  stream = rr.RecordingStream("test", recording_id=rec_id, send_properties=False)
  stream.save(str(path))
  stream.log("x", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
  stream.flush()
  return path


def test_state_files_selects_the_one_configuration_named(tmp_path: Path) -> None:
  """Naming an absent configuration reports the ones present."""
  for name in ("state_s12_r100", "state_s8_r100", "estimate"):
    _write_rrd((tmp_path / name).mkdir() or tmp_path / name / "s1.rrd", "s1")
  (tmp_path / "state_empty").mkdir()
  files = state_files(tmp_path / "state_s12_r100")
  assert [p.name for p in files] == ["s1.rrd"]
  with pytest.raises(SystemExit, match=r"state_s12_r100.*state_s8_r100"):
    state_files((tmp_path / "state_s4_r100").mkdir() or tmp_path / "state_s4_r100")
  with pytest.raises(SystemExit, match="not a state layer"):
    state_files(tmp_path / "estimate")
