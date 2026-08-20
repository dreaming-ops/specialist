"""The trajectory ribbon carries one step per sample; the window is the view's."""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.experimental import Chunk, RrdReader

from specialist.vi_ba.viz import (
  CameraVizData,
  LensVizData,
  _gripper_streams,
  _tracker_indexes,
  _trail_segments,
  _trajectory_streams,
  blueprint_for_names,
  segment_card_blueprint,
)


def test_each_sample_carries_the_step_that_reached_it() -> None:
  """The first step and a step longer than the window stand still."""
  traj = np.arange(12, dtype=np.float64).reshape(4, 3)
  segments = _trail_segments(traj, np.array([0.0, 0.1, 5.0, 5.1]), 1.0)
  assert np.array_equal(segments, np.stack([traj[[0, 0, 2, 2]], traj], axis=1))


def test_the_trail_ends_at_the_tcp_on_the_base_tag() -> None:
  """Each step ends where the body pose carries the base tag's TCP offset."""
  times = np.linspace(0.0, 0.2, 3)
  poses = [gtsam.Pose3(gtsam.Rot3(), np.array([0.1 * i, 0.0, 0.0])) for i in range(3)]
  hr = np.linspace(0.0, 0.2, 5)
  data = CameraVizData(name="t0", body_poses=poses, keyframe_times=times)
  data.lenses = {"front": LensVizData()}
  data.urdf_root_poses = list(poses)
  data.tcp_in_base_tag = (0.0, 0.0, 0.05)
  data.highrate_times = hr
  data.highrate_positions = np.stack([hr, np.zeros(5), np.zeros(5)], axis=1)
  data.highrate_quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (5, 1))
  indexes = _tracker_indexes(data, 1.0 / 30.0)
  world_T_base_tag = _gripper_streams(lambda _: None, data, indexes, "t0_base")
  assert world_T_base_tag is not None
  translations, rotations = world_T_base_tag
  chunks: list[Chunk] = []
  _trajectory_streams(chunks.append, data, indexes, world_T_base_tag, (1, 2, 3))
  expected = translations + rotations @ np.asarray(data.tcp_in_base_tag)
  strips = chunks[0].to_record_batch().column("LineStrips3D:strips").to_pylist()
  assert np.allclose([s[0][-1] for s in strips], expected)


def _ranged(blueprint: rrb.Blueprint, rrd: Path) -> set[str]:
  """The entity paths the blueprint gives a visible time range."""
  with rr.RecordingStream("app", recording_id="bp", send_properties=False) as stream:
    stream.save(str(rrd))
    stream.send_blueprint(blueprint, make_active=True, make_default=True)
  reader = RrdReader(str(rrd))
  columns = reader.store(store=reader.blueprints()[0]).schema().component_columns()
  return {c.entity_path for c in columns if c.component == "VisibleTimeRanges:ranges"}


def test_every_view_that_draws_the_ribbon_sets_the_window(tmp_path: Path) -> None:
  """The samples carry one step each, so the view sets the visible time range."""
  views = {
    "dashboard": blueprint_for_names(["t0"]),
    "card": segment_card_blueprint(["t0"]),
  }
  for name, blueprint in views.items():
    ranged = _ranged(blueprint, tmp_path / f"{name}.rrd")
    assert any(p.endswith("overrides/t0/trajectory/visualizers") for p in ranged), (
      f"{name}: {ranged}"
    )
