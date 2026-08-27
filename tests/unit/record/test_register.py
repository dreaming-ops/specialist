"""Registration guards: recording ids, layer attachment, and name collisions."""

from pathlib import Path

import pytest
import rerun as rr
from rerun.experimental import Chunk, send_chunks
from scripts.register import (
  check_layers_attach,
  check_no_collisions,
  dataset_layers,
  recording_id,
)


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
  _write_rrd((tmp_path / "estimate").mkdir() or tmp_path / "estimate" / "s1.rrd", "s1")
  _write_rrd((tmp_path / "mocap").mkdir() or tmp_path / "mocap" / "s1.rrd", "s1")
  (tmp_path / "blueprints").mkdir()  # no rrds: not a layer
  bases, layers = dataset_layers(tmp_path)
  assert [p.name for p in bases] == ["s1.rrd"]
  assert sorted(layers) == ["estimate", "mocap"]


def test_dataset_layers_registers_the_one_state_layer_named(
  tmp_path: Path,
) -> None:
  """Only the named state layer registers, none when unnamed; a missing one exits."""
  for name in ("base", "estimate", "state_s12", "state_s8"):
    _write_rrd((tmp_path / name).mkdir() or tmp_path / name / "s1.rrd", "s1")
  _, layers = dataset_layers(tmp_path, state="state_s12")
  assert sorted(layers) == ["estimate", "state_s12"]
  _, layers = dataset_layers(tmp_path)
  assert sorted(layers) == ["estimate"]
  with pytest.raises(SystemExit, match="state_s4"):
    dataset_layers(tmp_path, state="state_s4")


def test_a_state_layer_alone_stands_as_the_segment(tmp_path: Path) -> None:
  _write_rrd(
    (tmp_path / "state_s12").mkdir() or tmp_path / "state_s12" / "s1.rrd", "s1"
  )
  bases, layers = dataset_layers(tmp_path, state="state_s12")
  assert bases == [] and sorted(layers) == ["state_s12"]


def test_dataset_layers_requires_a_base(tmp_path: Path) -> None:
  (tmp_path / "estimate").mkdir()
  _write_rrd(tmp_path / "estimate" / "s1.rrd", "s1")
  with pytest.raises(SystemExit, match="base layer"):
    dataset_layers(tmp_path)


def _layer(path: Path, entity: str, component: str, value: float) -> Path:
  """One rrd holding a single static component."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with rr.RecordingStream("test", recording_id="s", send_properties=False) as stream:
    stream.save(str(path))
    send_chunks(
      [
        Chunk.from_columns(
          entity, indexes=[], columns=rr.AnyValues.columns(**{component: [value]})
        )
      ],
      recording=stream,
    )
  return path


def _property_layer(path: Path, group: str, key: str) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  with rr.RecordingStream("test", recording_id="s", send_properties=False) as stream:
    stream.save(str(path))
    stream.send_property(group, rr.AnyValues(**{key: 1.0}))
  return path


def test_two_layers_writing_one_entity_component_are_refused(tmp_path: Path) -> None:
  """Registration refuses two layers writing one ``(entity, component)``."""
  files = [
    _layer(tmp_path / "estimate" / "s.rrd", "camera/cam00/fit", "front_rmse", 1.0),
    _layer(tmp_path / "viz" / "s.rrd", "camera/cam00/fit", "front_rmse", 2.0),
  ]
  with pytest.raises(SystemExit, match="camera/cam00/fit:front_rmse"):
    check_no_collisions({"s": files})


def test_two_layers_writing_one_property_key_are_refused(tmp_path: Path) -> None:
  """Keys resolve in registration order, so a property key in two layers is refused."""
  files = [
    _property_layer(tmp_path / "estimate" / "s.rrd", "estimate", "worst_rmse"),
    _property_layer(tmp_path / "viz" / "s.rrd", "estimate", "worst_rmse"),
  ]
  with pytest.raises(SystemExit, match="worst_rmse"):
    check_no_collisions({"s": files})


def test_layers_that_share_no_name_pass(tmp_path: Path) -> None:
  """Layers of one session may share an entity, never a component of one."""
  files = [
    _layer(tmp_path / "estimate" / "s.rrd", "camera/cam00/fit", "front_rmse", 1.0),
    _layer(tmp_path / "viz" / "s.rrd", "camera/cam00/fit", "n_keypoints", 2.0),
  ]
  check_no_collisions({"s": files})
