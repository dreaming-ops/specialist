"""The decomposed viz seams are drivable one at a time."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gtsam
import numpy as np
import pytest
from rerun.chunk import Chunk

from specialist.vi_ba.viz import (
  CameraEstimate,
  Layers,
  LensEstimate,
  _urdf_model,
)
from tests.record_fixtures import urdf_file


def _data() -> CameraEstimate:
  times = np.linspace(0.0, 0.2, 3)
  poses = [gtsam.Pose3(gtsam.Rot3(), np.array([0.1 * i, 0.0, 0.0])) for i in range(3)]
  hr = np.linspace(0.0, 0.2, 5)
  data = CameraEstimate(
    slot="cam00",
    body_poses=poses,
    keyframe_times=times,
    body_T_base_tag=gtsam.Pose3(),
    body_T_rear_lens=gtsam.Pose3(),
    base_tag_id=2000,
    keyframe_frames=np.arange(len(times), dtype=np.int64) * 3,
    highrate_times=hr,
    highrate_positions=np.stack([hr, np.zeros(5), np.zeros(5)], axis=1),
    highrate_quats=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (5, 1)),
    highrate_vels=np.zeros((5, 3)),
  )
  data.lenses = {"front": LensEstimate()}
  data.tcp_in_base_tag = (0.0, 0.0, 0.05)
  return data


def test_a_urdf_that_cannot_be_read_stops_the_run(tmp_path: Path) -> None:
  """A URDF that fails to parse raises."""
  broken = tmp_path / "broken.urdf"
  broken.write_text("<robot name=")
  chunks: list[Chunk] = []
  with pytest.raises(RuntimeError):
    _urdf_model(Layers(chunks.append, chunks.append), _data(), broken)


def test_a_missing_mesh_costs_pictures_not_answers(tmp_path: Path) -> None:
  """An unreadable mesh still yields the tree and root frame and writes ``urdf_sha``."""
  path = tmp_path / "robot.urdf"
  path.write_text(
    '<?xml version="1.0"?><robot name="gripper">'
    '<link name="base"><visual><geometry>'
    '<mesh filename="nowhere.stl"/></geometry></visual></link>'
    "</robot>"
  )
  chunks: list[Chunk] = []
  tree, root_frame = _urdf_model(Layers(chunks.append, chunks.append), _data(), path)
  assert tree.name == "gripper"
  assert root_frame == "cam00_base"
  written = {
    name
    for chunk in chunks
    for name in chunk.to_record_batch().schema.names
    if not name.startswith("rerun.controls.")
  }
  assert "urdf_sha" in written, "the model's provenance did not survive"


def _child_frames(chunks: list[Chunk]) -> set[str]:
  """Every frame the chunks define an edge onto."""
  frames = set()
  for chunk in chunks:
    batch = chunk.to_record_batch()
    if "Transform3D:child_frame" in batch.schema.names:
      column = batch.column("Transform3D:child_frame").to_pylist()
      frames |= {f for row in column if row for f in row}
  return frames


def test_geometry_that_cannot_be_read_does_not_cost_the_fixed_joint_edges(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The fixed-joint edge to the base tag survives geometry the reader cannot open."""
  path = urdf_file(tmp_path)
  from rerun.urdf import UrdfTree

  def boom(self: Any, link: Any) -> list[str]:
    raise RuntimeError("mesh unreadable")

  monkeypatch.setattr(UrdfTree, "get_collision_geometry_paths", boom)
  chunks: list[Chunk] = []
  _urdf_model(Layers(chunks.append, chunks.append), _data(), path)
  frames = _child_frames(chunks)
  assert "cam00_base_tag" in frames, f"the fixed joint edge is gone: {sorted(frames)}"


def test_a_missing_mesh_does_not_cost_the_fixed_joint_edges(tmp_path: Path) -> None:
  """The fixed-joint tag frame the solve reprojects onto survives an unreadable mesh."""
  path = tmp_path / "robot.urdf"
  path.write_text(
    '<?xml version="1.0"?><robot name="gripper">'
    '<link name="base"><visual><geometry>'
    '<mesh filename="nowhere.stl"/></geometry></visual></link>'
    '<link name="base_tag"/>'
    '<joint name="base_tag" type="fixed"><parent link="base"/>'
    '<child link="base_tag"/><origin xyz="0 0.01 0.02" rpy="0 0 0"/></joint>'
    "</robot>"
  )
  chunks: list[Chunk] = []
  _urdf_model(Layers(chunks.append, chunks.append), _data(), path)
  frames = set()
  for chunk in chunks:
    batch = chunk.to_record_batch()
    if "Transform3D:child_frame" in batch.schema.names:
      column = batch.column("Transform3D:child_frame").to_pylist()
      frames |= {f for row in column if row for f in row}
  assert "cam00_base_tag" in frames, f"the fixed joint edge is gone: {sorted(frames)}"
