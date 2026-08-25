"""The trajectory ribbon carries one step per sample; the window is the view's."""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.experimental import Chunk, RrdReader

from specialist.vi_ba.blueprint import blueprint_for_slots, segment_card_blueprint
from specialist.vi_ba.viz import (
  Layers,
  _camera_indexes,
  _gripper_streams,
  _trail_segments,
  _trajectory_streams,
  run_blueprint,
)
from tests.record_fixtures import camera_estimate


def test_each_sample_carries_the_step_that_reached_it() -> None:
  """The first step and a step longer than the window stand still."""
  traj = np.arange(12, dtype=np.float64).reshape(4, 3)
  segments = _trail_segments(traj, np.array([0.0, 0.1, 5.0, 5.1]), 1.0)
  assert np.array_equal(segments, np.stack([traj[[0, 0, 2, 2]], traj], axis=1))


def test_the_trail_ends_at_the_tcp_on_the_base_tag() -> None:
  """Each step ends where the body pose carries the base tag's TCP offset."""
  data = camera_estimate("cam00")
  chunks: list[Chunk] = []
  layers = Layers(chunks.append, chunks.append)
  indexes = _camera_indexes(data, 1.0 / 30.0, timeline_start=0.0)
  world_T_base_tag = _gripper_streams(layers, data, indexes, "cam00_base")
  _trajectory_streams(layers, data, world_T_base_tag, (1, 2, 3))
  tcp = np.asarray(data.tcp_in_base_tag)
  expected = [
    gtsam.Pose3(gtsam.Rot3.Quaternion(w, x, y, z), t)
    .compose(data.body_T_base_tag)
    .transformFrom(tcp)
    for (x, y, z, w), t in zip(
      data.highrate_quats, data.highrate_positions, strict=True
    )
  ]
  batch = next(
    c.to_record_batch()
    for c in chunks
    if "LineStrips3D:strips" in c.to_record_batch().schema.names
  )
  strips = batch.column("LineStrips3D:strips").to_pylist()
  assert np.allclose([s[0][-1] for s in strips], expected)


def _ranged(blueprint: rrb.Blueprint, rrd: Path) -> set[str]:
  """The entity paths the blueprint gives a visible time range."""
  with rr.RecordingStream("app", recording_id="bp", send_properties=False) as stream:
    stream.save(str(rrd))
    stream.send_blueprint(blueprint, make_active=True, make_default=True)
  reader = RrdReader(str(rrd))
  columns = reader.store(store=reader.blueprints()[0]).schema().component_columns()
  return {c.entity_path for c in columns if c.component == "VisibleTimeRanges:ranges"}


def _visualized(blueprint: rrb.Blueprint, rrd: Path) -> set[str]:
  """The entity paths the blueprint puts an explicit visualizer set on."""
  with rr.RecordingStream("app", recording_id="bp", send_properties=False) as stream:
    stream.save(str(rrd))
    stream.send_blueprint(blueprint, make_active=True, make_default=True)
  reader = RrdReader(str(rrd))
  columns = reader.store(store=reader.blueprints()[0]).schema().component_columns()
  return {
    c.entity_path for c in columns if c.component == "ActiveVisualizers:instruction_ids"
  }


def test_the_written_blueprint_carries_an_override_for_every_anchor(
  tmp_path: Path,
) -> None:
  """An anchor's pose renders nothing on its own."""
  drawn = _visualized(
    blueprint_for_slots(["cam00"], anchor_tag_ids=[7]), tmp_path / "with.rrd"
  )
  assert any(p.endswith("world/anchor/7/visualizers") for p in drawn), sorted(drawn)
  bare = _visualized(blueprint_for_slots(["cam00"]), tmp_path / "without.rrd")
  assert not any("anchor" in p for p in bare)


def test_the_live_dashboard_draws_the_anchors_the_run_solved(tmp_path: Path) -> None:
  """The post-solve blueprint comes from the run, so it carries the solved anchors."""
  drawn = _visualized(
    run_blueprint([camera_estimate("cam00")], {7: gtsam.Pose3()}.keys(), 0.1),
    tmp_path / "live.rrd",
  )
  assert any(p.endswith("world/anchor/7/visualizers") for p in drawn), sorted(drawn)


def test_every_view_that_draws_the_ribbon_sets_the_window(tmp_path: Path) -> None:
  """The samples carry one step each, so the view sets the visible time range."""
  views = {
    "dashboard": blueprint_for_slots(["cam00"]),
    "card": segment_card_blueprint(["cam00"]),
  }
  for name, blueprint in views.items():
    ranged = _ranged(blueprint, tmp_path / f"{name}.rrd")
    assert any(
      p.endswith("overrides/camera/cam00/trajectory/visualizers") for p in ranged
    ), f"{name}: {ranged}"
