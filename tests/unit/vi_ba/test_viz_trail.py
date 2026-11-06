"""The trajectory ribbon carries one step per sample; the window is the view's."""

from pathlib import Path

import gtsam
import numpy as np
import rerun.blueprint as rrb
from rerun.chunk import Chunk, LazyStore, RrdReader

from specialist.vi_ba.blueprint import (
  blueprint_for_slots,
  state_blueprint,
  state_card_blueprint,
)
from specialist.vi_ba.viz import (
  Layers,
  _camera_indexes,
  _gripper_streams,
  _trail_segments,
  _trajectory_streams,
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
  layers = Layers(chunks, chunks)
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


def _store(blueprint: rrb.Blueprint | rrb.TableBlueprint, rrd: Path) -> LazyStore:
  blueprint.save("app", str(rrd))
  reader = RrdReader(str(rrd))
  return reader.store(store=reader.blueprints()[0])


def _carrying(blueprint: rrb.Blueprint, rrd: Path, component: str) -> set[str]:
  """The blueprint entities that carry one component."""
  columns = _store(blueprint, rrd).schema().component_columns()
  return {c.entity_path for c in columns if c.component == component}


def _views(blueprint: rrb.Blueprint | rrb.TableBlueprint, rrd: Path) -> dict[str, str]:
  """Each view's name and the entity path it takes as its origin."""
  views: dict[str, str] = {}
  for chunk in _store(blueprint, rrd).stream().to_chunks():
    batch = chunk.to_record_batch()
    if "ViewBlueprint:space_origin" not in batch.schema.names:
      continue
    names = batch.column("ViewBlueprint:display_name").to_pylist()
    origins = batch.column("ViewBlueprint:space_origin").to_pylist()
    for name, origin in zip(names, origins, strict=True):
      if name and origin:
        views[name[0]] = origin[0]
  return views


def _overrides(blueprint: rrb.Blueprint, rrd: Path, view: str) -> set[tuple[str, str]]:
  """Each entity the named view overrides, with the component overridden."""
  store = _store(blueprint, rrd)
  names: dict[str, str] = {}
  for chunk in store.stream().to_chunks():
    batch = chunk.to_record_batch()
    if "ViewBlueprint:display_name" in batch.schema.names:
      for name in batch.column("ViewBlueprint:display_name").to_pylist():
        if name:
          names[str(chunk.entity_path)] = name[0]
  out = set()
  for column in store.schema().component_columns():
    root, marker, entity = column.entity_path.partition("/ViewContents/overrides/")
    if marker and names.get(root) == view:
      out.add((entity.split("/visualizers")[0], column.component))
  return out


def test_the_state_dashboard_shows_each_hand_by_name(tmp_path: Path) -> None:
  views = _views(state_blueprint(), tmp_path / "o.rrd")
  assert views["left"] == "/state/left/image"
  assert views["right"] == "/state/right/image"
  assert not any(o.endswith(("/front", "/rear")) for o in views.values())
  plain = _views(blueprint_for_slots(["cam00"]), tmp_path / "p.rrd")
  assert plain["cam00/front"] == "/camera/cam00/front"
  assert "left" not in plain


def test_the_state_card_is_one_pane_over_both_hands(tmp_path: Path) -> None:
  """The viewer shows cards only from a table blueprint's card layout."""
  views = _views(state_card_blueprint(240), tmp_path / "c.rrd")
  assert views == {"state": "/state"}
  paths = _store(state_card_blueprint(240), tmp_path / "c.rrd").schema().entity_paths()
  assert "/table/layouts/cards" in {str(p) for p in paths}


def test_the_state_dashboard_puts_a_triad_on_each_hand(tmp_path: Path) -> None:
  """A bare ``Transform3D`` draws nothing, so the triad is an explicit override."""
  overridden = _overrides(state_blueprint(), tmp_path / "s.rrd", "3D")
  for hand in ("left", "right"):
    assert ("state/" + hand + "/pose", "TransformAxes3D:axis_length") in overridden


def test_the_written_blueprint_carries_an_override_for_every_anchor(
  tmp_path: Path,
) -> None:
  """An anchor's pose renders nothing on its own."""
  drawn = _carrying(
    blueprint_for_slots(["cam00"], anchor_tag_ids=[7]),
    tmp_path / "with.rrd",
    "ActiveVisualizers:instruction_ids",
  )
  assert any(p.endswith("world/anchor/7/visualizers") for p in drawn), sorted(drawn)
  bare = _carrying(
    blueprint_for_slots(["cam00"]),
    tmp_path / "without.rrd",
    "ActiveVisualizers:instruction_ids",
  )
  assert not any("anchor" in p for p in bare)


def test_every_view_that_draws_the_ribbon_sets_the_window(tmp_path: Path) -> None:
  """The samples carry one step each, so the view sets the visible time range."""
  ranged = _carrying(
    blueprint_for_slots(["cam00"]),
    tmp_path / "dashboard.rrd",
    "VisibleTimeRanges:ranges",
  )
  assert any(
    p.endswith("overrides/camera/cam00/trajectory/visualizers") for p in ranged
  ), ranged
