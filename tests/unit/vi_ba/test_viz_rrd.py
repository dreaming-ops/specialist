"""The rrd written by the layer writer round-trips through the reader."""

import hashlib
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import gtsam
import numpy as np
import pytest
from rerun.experimental import Chunk, ChunkStore, RrdReader
from specialist_dataset import schema

from specialist.vi_ba.blueprint import (
  _hidden_overlay_overrides,
  _triad_overrides,
  _without_pixels,
)
from specialist.vi_ba.summary import CameraSummary, Summary
from specialist.vi_ba.viz import (
  CameraEstimate,
  Layers,
  estimate_chunks,
)
from tests.record_fixtures import camera_estimate, urdf_file, write_run

# Every entity a solve writes, and the layer that must carry it.
_ESTIMATE = (
  "/world",
  "/world/anchor/0",
  "/camera/cam00",
  "/camera/cam00/base_tag",
  "/camera/cam00/rear_seat",
  "/camera/cam00/state/pose",
  "/camera/cam00/state/lin_vel",
  "/camera/cam00/lin_vel",
  "/camera/cam00/imu/bias/gyro",
  "/camera/cam00/imu/bias/accel",
  "/camera/cam00/gripper",
  "/camera/cam00/gripper/tf_static",
  "/camera/cam00/front/detection/keypoints",
  "/camera/cam00/front/detection/keypoints_rejected",
  "/camera/cam00/front/detection/tags",
  "/camera/cam00/front/detection/tags/center",
  "/camera/cam01",
)
_VIZ = (
  "/camera/cam00/gripper/visual_geometries/base/visual_0",
  "/world/anchor/0",
  "/camera/cam00/trajectory",
  "/camera/cam00/marker",
  "/camera/cam00/landmark/front",
  "/camera/cam00/front/overlay/tag_axes",
  "/camera/cam00/front/overlay/landmark_reprojection",
  "/camera/cam01/trajectory",
)


def _two_cameras() -> list[CameraEstimate]:
  return [camera_estimate("cam00"), camera_estimate("cam01", offset=1.0)]


@pytest.fixture(scope="module")
def estimated_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
  """One two-camera run with a URDF model."""
  tmp = tmp_path_factory.mktemp("solved")
  # A frame clock, so the review timeline exists.
  return write_run(
    tmp,
    "sess_route",
    _two_cameras(),
    urdf_path=urdf_file(tmp),
    frame_period=1.0 / 30.0,
  )


def _chunks(path: Path) -> list[Chunk]:
  return list(RrdReader(str(path)).store().stream().to_chunks())


def _names(path: Path) -> set[tuple[str, str]]:
  """Every ``(entity, component)`` one layer writes."""
  return {
    (chunk.entity_path, name)
    for chunk in _chunks(path)
    for name in chunk.to_record_batch().schema.names
    if not name.startswith("rerun.controls.") and name not in chunk.timeline_names
  }


def _column(path: Path, entity: str, component: str) -> list[list[object]]:
  """Every row one component holds at one entity, empty rows included."""
  rows = []
  for chunk in _chunks(path):
    if chunk.entity_path != entity:
      continue
    batch = chunk.to_record_batch()
    if component in batch.schema.names:
      rows.extend(batch.to_pydict()[component])
  return rows


def test_the_fixture_carries_one_of_everything_a_camera_holds() -> None:
  camera = camera_estimate("cam00")

  def carries_nothing(owner: object, name: str) -> bool:
    """True for an empty field or one holding a negative sentinel id."""
    value = getattr(owner, name)
    if value is None:
      return True
    if isinstance(value, np.ndarray):
      return value.size == 0
    if isinstance(value, (dict, list, str)):
      return not value
    return isinstance(value, int) and not isinstance(value, bool) and value < 0

  empty = [f.name for f in fields(camera) if carries_nothing(camera, f.name)]
  assert camera.lenses, "a camera with no lens exercises no overlay"
  # A field counts when any lens carries it; the rear lens carries only tags.
  empty += [
    f"LensEstimate.{f.name}"
    for f in fields(next(iter(camera.lenses.values())))
    if all(carries_nothing(lens, f.name) for lens in camera.lenses.values())
  ]
  assert not empty, f"a field no test exercises: {empty}"


def test_each_layer_carries_exactly_what_it_owns(
  estimated_run: tuple[Path, Path],
) -> None:
  """No ``(entity, component)`` is in both layers; each has its entities, none stray."""
  estimate_path, viz_path = estimated_run
  estimate, viz = _names(estimate_path), _names(viz_path)
  estimate_entities = {e for e, _ in estimate}
  viz_entities = {e for e, _ in viz}

  assert len(RrdReader(str(estimate_path)).recordings()) == 1
  assert not estimate & viz, f"{sorted(estimate & viz)} written by both layers"
  assert not set(_ESTIMATE) - estimate_entities, (
    f"missing: {sorted(set(_ESTIMATE) - estimate_entities)}"
  )
  assert not set(_VIZ) - viz_entities, f"missing: {sorted(set(_VIZ) - viz_entities)}"

  known = re.compile(
    r"^/(world(/anchor/\d+)?|estimate|__properties(/\w+)?"
    r"|camera/cam\d\d(/(front|rear)(/video|/(overlay|detection)(/[\w.]+)*)?"
    r"|/(state|imu|fit|landmark|factor|gripper)(/[\w./]+)?"
    r"|/(marker|base_tag|lin_vel|trajectory|rear_seat))?)$"
  )
  stray = sorted({e for e, _ in estimate | viz if not known.match(e)})
  assert not stray, f"entities routed by nobody: {stray}"


def test_a_static_component_never_replaces_a_different_temporal_one(
  estimated_run: tuple[Path, Path],
) -> None:
  """A static hides its component's temporal values for all time, in every layer."""
  for path in estimated_run:
    static: dict[tuple[str, str], list[Any]] = {}
    temporal: dict[tuple[str, str], list[Any]] = {}
    for chunk in _chunks(path):
      batch = chunk.to_record_batch()
      side = static if chunk.is_static else temporal
      for name, values in zip(
        batch.schema.names, batch.to_pydict().values(), strict=True
      ):
        if name.startswith("rerun.controls.") or name in chunk.timeline_names:
          continue
        side.setdefault((chunk.entity_path, name), []).extend(values)
    for key, covered in static.items():
      series = temporal.get(key)
      if series is None:
        continue
      assert all(value == covered[0] for value in series), (
        f"{path.parent.name}: static {key[1]} on {key[0]} replaces a series "
        f"that disagrees with it"
      )


# Entities measured from the camera ride its pose; anything else says where it is.
_RIDES_THE_BODY = frozenset(
  {"", "/front", "/rear", "/rear_seat", "/imu", "/base_tag", "/marker"}
)

# What a CoordinateFrame can place; on an entity holding none of these it is inert.
_DRAWABLE = ("Points3D", "LineStrips3D", "Boxes3D", "Asset3D", "TransformAxes3D")


def test_an_anchor_records_the_width_it_was_solved_against(
  estimated_run: tuple[Path, Path],
) -> None:
  estimate, _ = estimated_run
  sizes = set()
  for chunk in _chunks(estimate):
    if schema.anchor_id_of(str(chunk.entity_path)) is None:
      continue
    batch = chunk.to_record_batch()
    if "tag_size" in batch.schema.names:
      sizes |= {v for row in batch.column("tag_size").to_pylist() if row for v in row}
  assert sizes == {0.1}


def test_an_anchor_keeps_both_its_triad_and_its_outline() -> None:
  """An override replaces the visualizer set an entity gets on its own."""
  overrides = _triad_overrides(["cam00"], (7,), 0.1)
  drawn = overrides[f"/{schema.anchor(7)}"]
  assert [type(v).__name__ for v in drawn] == ["TransformAxes3D", "LineStrips3D"]


def test_what_the_camera_pose_carries_is_only_what_is_mounted_on_it(
  estimated_run: tuple[Path, Path],
) -> None:
  """Anything not mounted on the camera names its own frame or both ends of its edge."""
  declared: set[str] = set()
  detached: set[str] = set()
  drawable: set[str] = set()
  path_framed: set[str] = set()
  for path in estimated_run:
    for chunk in _chunks(path):
      entity, names = chunk.entity_path, chunk.to_record_batch().schema.names
      if "CoordinateFrame:frame" in names:
        declared.add(entity)
      if "Transform3D:parent_frame" in names:
        detached.add(entity)
      elif any(n.startswith("Transform3D:") for n in names):
        path_framed.add(entity)
      if any(n.split(":")[0] in _DRAWABLE for n in names):
        drawable.add(entity)

  def mounted(entity: str) -> bool:
    if not entity.startswith("/camera/cam0"):
      return True
    suffix = entity[len("/camera/cam00") :]
    return suffix in _RIDES_THE_BODY or suffix.startswith(("/front/", "/rear/"))

  adrift = sorted(e for e in drawable - declared if not mounted(e))
  assert not adrift, f"drawn through the camera pose, naming no frame: {adrift}"
  composed = sorted(e for e in path_framed - detached if not mounted(e))
  assert not composed, f"composes with the camera pose unintentionally: {composed}"

  # The blueprint draws on entities the layers only place.
  painted = drawable | set(_triad_overrides(["cam00", "cam01"], (0, 1), 0.1))
  inert = sorted(e for e in declared - painted if e.startswith("/camera/"))
  assert not inert, f"frame declared on an entity holding nothing drawable: {inert}"


def test_every_tag_the_solve_used_is_named_in_the_layer(
  estimated_run: tuple[Path, Path],
) -> None:
  """Each anchor, base tag and finger tag is an entity with its own pose and id."""
  named: dict[str, int] = {}
  for chunk in _chunks(estimated_run[0]):
    batch = chunk.to_record_batch()
    if "tag_id" in batch.schema.names:
      # Clearing rows carry an empty id; tag entities and drawn rows carry a real one.
      ids = [v for row in batch.to_pydict()["tag_id"] or [] for v in (row or [])]
      if ids:
        named[chunk.entity_path] = int(ids[0])
  assert named.get(f"/{schema.base_tag('cam00')}") == 2000
  for joint, tag in (("left_finger", 2001), ("right_finger", 2002)):
    entity = f"/{schema.joint_tag('cam00', joint)}"
    assert named.get(entity) == tag, (entity, named.get(entity), named)

  # Naming a frame nothing defines places the tag nowhere.
  defined, named_frames = set(), {}
  for path in estimated_run:
    for chunk in _chunks(path):
      batch = chunk.to_record_batch()
      row = batch.to_pydict()
      for column, sink in (
        ("Transform3D:child_frame", defined),
        ("CoordinateFrame:frame", None),
      ):
        if column not in batch.schema.names:
          continue
        for value in row[column]:
          frame = value[0] if isinstance(value, list) else value
          if not frame:
            continue
          if sink is None:
            named_frames[chunk.entity_path] = str(frame)
          else:
            sink.add(str(frame))
  dangling = {e: f for e, f in named_frames.items() if f not in defined}
  assert not dangling, f"entities naming a frame nothing defines: {dangling}"


def test_the_model_the_angles_came_from_is_named(
  estimated_run: tuple[Path, Path], tmp_path: Path
) -> None:
  """The layer names the angles' source model and each joint's axis and limits."""
  values: dict[tuple[str, str], Any] = {}
  for chunk in _chunks(estimated_run[0]):
    batch = chunk.to_record_batch()
    for name, column in zip(
      batch.schema.names, batch.to_pydict().values(), strict=True
    ):
      if name in ("robot", "urdf_sha", "axis", "joint_type", "limit_lower"):
        values[chunk.entity_path, name] = column[0]

  def one(entity: str, name: str) -> Any:
    return np.asarray(values[entity, name]).reshape(-1)

  gripper = f"/{schema.gripper('cam00')}"
  assert one(gripper, "robot")[0] == "gripper"
  digest = str(one(gripper, "urdf_sha")[0])
  # The fixture writes the same bytes every time, so a fresh copy has the same digest.
  assert digest == hashlib.sha256(urdf_file(tmp_path).read_bytes()).hexdigest(), (
    "the digest does not track the model it was read from"
  )
  for joint in ("left_finger", "right_finger"):
    entity = f"/{schema.joint_model('cam00', joint)}"
    assert one(entity, "joint_type")[0] == "revolute"
    # The finger factors rotate about +z and do not read the URDF axis.
    assert list(one(entity, "axis")) == [0.0, 0.0, 1.0]
    assert one(entity, "limit_lower")[0] == pytest.approx(-1.0)


def test_no_frame_is_defined_twice(estimated_run: tuple[Path, Path]) -> None:
  """Rerun resolves two definitions of one frame by log order, and a static one hides a temporal one."""
  by_frame: dict[str, set[str]] = {}
  for path in estimated_run:
    for chunk in _chunks(path):
      batch = chunk.to_record_batch()
      if "Transform3D:child_frame" not in batch.schema.names:
        continue
      for value in batch.to_pydict()["Transform3D:child_frame"]:
        name = value[0] if isinstance(value, list) else value
        if name:
          by_frame.setdefault(str(name), set()).add(chunk.entity_path)
  twice = {name: sorted(e) for name, e in by_frame.items() if len(e) > 1}
  assert not twice, f"frames defined by more than one entity: {twice}"


def test_what_the_lens_measured_is_an_answer_not_a_picture(
  estimated_run: tuple[Path, Path],
) -> None:
  """Measured pixels live in the estimate layer; the label string is a rendering."""
  estimate, viz = _names(estimated_run[0]), _names(estimated_run[1])
  lens = "/camera/cam00/front/detection"
  for entity, component in (
    (f"{lens}/keypoints", "Points2D:positions"),
    (f"{lens}/keypoints_rejected", "Points2D:positions"),
    (f"{lens}/tags", "LineStrips2D:strips"),
    (f"{lens}/tags", "tag_id"),
    (f"{lens}/tags/center", "Points2D:positions"),
  ):
    assert (entity, component) in estimate, f"{entity}:{component} is not an answer"
    assert (entity, component) not in viz, f"{entity}:{component} is drawn twice"

  ids = _column(estimated_run[0], f"{lens}/tags", "tag_id")
  assert sorted({int(v) for row in ids for v in row}) == [0, 2000]
  assert (f"{lens}/tags/center", "Points2D:labels") in viz


def test_no_answer_is_split_across_clocks(estimated_run: tuple[Path, Path]) -> None:
  estimate_path, _viz_path = estimated_run
  clocks: dict[tuple[str, str], set[frozenset[str]]] = {}
  for chunk in _chunks(estimate_path):
    if chunk.is_static:
      continue
    timelines = frozenset(chunk.timeline_names)
    for name in chunk.to_record_batch().schema.names:
      if name.startswith("rerun.controls.") or name in timelines:
        continue
      clocks.setdefault((chunk.entity_path, name), set()).add(timelines)
  split = {
    key: sorted(map(sorted, seen)) for key, seen in clocks.items() if len(seen) > 1
  }
  assert not split, f"one name, different clocks: {split}"


def test_the_session_clock_carries_durations_in_every_layer(
  estimated_run: tuple[Path, Path],
) -> None:
  """A reader divides the session clock's durations by a nanosecond."""
  seen = 0
  for path in estimated_run:
    for chunk in _chunks(path):
      columns = chunk.to_record_batch().schema
      if schema.TIMELINE in columns.names:
        dtype = columns.field(schema.TIMELINE).type
        assert "duration" in str(dtype), (path.parent.name, dtype)
        seen += 1
  assert seen, "no chunk carried the session clock"


def test_no_layer_embeds_a_blueprint(
  estimated_run: tuple[Path, Path],
) -> None:
  """An embedded blueprint shadows the dataset .rbl on direct opens."""
  estimate_path, viz_path = estimated_run
  assert not RrdReader(str(viz_path)).blueprints()
  assert not RrdReader(str(estimate_path)).blueprints()


def test_a_model_that_fails_to_write_is_not_mistaken_for_a_missing_model(
  tmp_path: Path,
) -> None:
  """A sink's error on a mesh chunk propagates; the model guard covers only building."""
  data = camera_estimate("cam00")

  def die(chunk: Chunk) -> None:
    if "geometries" in chunk.entity_path:
      raise OSError(28, "No space left on device")

  with pytest.raises(OSError):
    estimate_chunks(
      Layers(die, die),
      [data],
      {0: gtsam.Pose3()},
      0,
      0.1,
      urdf_path=urdf_file(tmp_path),
      timeline_start=0.0,
      frame_period=1.0 / 30.0,
    )


def test_a_viewer_that_cannot_open_does_not_cost_the_run_its_layers(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """With ``rerun.spawn`` patched to fail, the run's layers still get written."""

  def no_display(**kwargs: object) -> None:
    raise RuntimeError("no display")

  monkeypatch.setattr("rerun.spawn", no_display)
  for path in write_run(
    tmp_path,
    "sess_headless",
    _two_cameras(),
    open_viewer=True,
    urdf_path=urdf_file(tmp_path),
    frame_period=1.0 / 30.0,
  ):
    assert path.exists()


def test_every_number_the_summary_carries_reaches_the_layer(
  estimated_run: tuple[Path, Path],
) -> None:
  written = {
    (c.entity_path, c.component)
    for c in RrdReader(str(estimated_run[0])).store().schema().component_columns()
  }
  properties = {c for e, c in written if e.startswith("/__properties")}
  for field in fields(CameraSummary):
    if field.name != "slot":
      assert ("/camera/cam00/fit", field.name) in written, field.name
  # segment_table() lacks projection pushdown, so a query reads every property column.
  assert properties == {f.name for f in fields(Summary)} | {
    "worst_rmse",
    "git_sha",
    "estimated_at",
    "rerun_version",
  }


def test_training_fields_carry_the_keyframe_index(
  estimated_run: tuple[Path, Path],
) -> None:
  """A dataloader joins one training row by the integer keyframe index."""
  timelines: dict[str, set[str]] = {}
  for chunk in _chunks(estimated_run[0]):
    timelines.setdefault(chunk.entity_path, set()).update(chunk.timeline_names)
  pose = "/camera/cam00/state/pose"
  assert {"video_time", "keyframe"} <= timelines[pose], timelines.get(pose)


def test_every_overlay_answers_on_the_keyframe_clock_alone(tmp_path: Path) -> None:
  """An overlay is one keyframe's answer about one frame."""
  estimate, viz = write_run(
    tmp_path,
    "sess_kf",
    _two_cameras(),
    urdf_path=urdf_file(tmp_path),
    frame_period=1.0 / 30.0,
  )
  geometry = ("strips", "positions")
  answered = 0
  for path in (estimate, viz):
    by_entity: dict[str, list[Chunk]] = {}
    for chunk in _chunks(path):
      if not chunk.is_static and (
        "/overlay/" in chunk.entity_path or "/detection/" in chunk.entity_path
      ):
        by_entity.setdefault(chunk.entity_path, []).append(chunk)
    for entity, chunks in by_entity.items():
      assert not any(schema.TIMELINE in c.timeline_names for c in chunks), (
        f"{entity} is written on the video clock, where it would be held over "
        f"the frames that follow the keyframe it was computed from"
      )
      table = (
        ChunkStore.from_chunks(chunks)
        .reader(index=schema.KEYFRAME_TIMELINE)
        .to_arrow_table()
      )
      for name in table.schema.names:
        if name.split(":")[-1] not in geometry:
          continue
        assert any(table.column(name).to_pylist()), (
          f"{entity}:{name} resolves to nothing on the keyframe timeline"
        )
        answered += 1
  assert answered >= 8, f"only {answered} geometry columns were checked"


def test_the_tag_overlay_is_visible_without_a_toggle() -> None:
  """The pixel clouds beside the tags stay a toggle away."""
  hidden = _hidden_overlay_overrides("cam00", "front")
  tags = f"/{schema.detection('cam00', 'front', schema.TAGS_DETECTION)}"
  assert tags not in hidden
  assert f"/{schema.overlay('cam00', 'front', schema.TAG_AXES_OVERLAY)}" not in hidden
  for path, kind in (
    (schema.detection, schema.KEYPOINTS_DETECTION),
    (schema.detection, schema.REJECTED_KEYPOINTS_DETECTION),
    (schema.overlay, schema.LANDMARK_REPROJECTION_OVERLAY),
  ):
    override = hidden[f"/{path('cam00', 'front', kind)}"]
    assert override.visible.as_arrow_array().to_pylist() == [False], kind


def test_a_posed_entity_carries_nothing_wider_than_its_pose(
  estimated_run: tuple[Path, Path],
) -> None:
  """An entity has as many instances as its widest component."""
  posed: set[str] = set()
  widths: dict[str, dict[str, int]] = {}
  for path in estimated_run:
    for chunk in _chunks(path):
      batch = chunk.to_record_batch()
      row = batch.to_pydict()
      for field in batch.schema:
        if (field.metadata or {}).get(b"kind") != b"data":
          continue
        if field.name.startswith("Transform3D:"):
          posed.add(chunk.entity_path)
        width = max((len(v) for v in row[field.name] if v is not None), default=0)
        seen = widths.setdefault(chunk.entity_path, {})
        seen[field.name] = max(seen.get(field.name, 0), width)
  wide = {
    entity: {n: w for n, w in widths[entity].items() if w > 1}
    for entity in posed
    if any(w > 1 for w in widths[entity].values())
  }
  assert not wide, f"posed entities carrying a wider component: {wide}"


def test_the_3d_view_drops_everything_measured_in_pixels(
  estimated_run: tuple[Path, Path],
) -> None:
  """The viewer refuses a 2D archetype without a pinhole above it."""
  flat = ("Points2D", "LineStrips2D", "Boxes2D")
  pixel_entities = {
    chunk.entity_path
    for path in estimated_run
    for chunk in _chunks(path)
    if any(n.split(":")[0] in flat for n in chunk.to_record_batch().schema.names)
  }
  assert pixel_entities, "no 2D entity was written, so nothing is being checked"
  dropped = {
    pattern.removeprefix("- ").removesuffix("/**")
    for pattern in _without_pixels(["cam00", "cam01"])
  }
  stray = sorted(
    entity
    for entity in pixel_entities
    if not any(entity == d or entity.startswith(f"{d}/") for d in dropped)
  )
  assert not stray, f"2D entities the 3D view still pulls in: {stray}"
