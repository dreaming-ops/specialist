"""The decomposed viz seams are drivable one at a time."""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
from rerun.experimental import Chunk

from specialist.vi_ba.viz import (
  CameraVizData,
  LensVizData,
  _tracker_indexes,
  _trajectory_streams,
  _urdf_model,
)


def _data() -> CameraVizData:
  times = np.linspace(0.0, 0.2, 3)
  poses = [gtsam.Pose3(gtsam.Rot3(), np.array([0.1 * i, 0.0, 0.0])) for i in range(3)]
  data = CameraVizData(name="t0", body_poses=poses, keyframe_times=times)
  data.lenses = {"front": LensVizData()}
  data.urdf_root_poses = list(poses)
  data.tcp_in_base_tag = (0.0, 0.0, 0.05)
  return data


def test_a_urdf_that_cannot_be_read_places_no_model(tmp_path: Path) -> None:
  broken = tmp_path / "broken.urdf"
  broken.write_text("<robot name=")
  chunks: list[Chunk] = []
  assert _urdf_model(chunks.append, chunks.append, _data(), broken) is None
  assert chunks == []


def test_without_an_imu_the_trail_falls_back_to_the_keyframes() -> None:
  data = _data()
  indexes = _tracker_indexes(data, 1.0 / 30.0)
  chunks: list[Chunk] = []
  _trajectory_streams(chunks.append, data, indexes, None, (1, 2, 3))
  strips = chunks[0].to_record_batch().column("LineStrips3D:strips").to_pylist()
  assert len(strips) == len(data.keyframe_times)
