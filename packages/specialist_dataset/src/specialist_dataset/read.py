"""Read a dataset layer as numpy through the selectors ``schema`` mints."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import rerun as rr
from specialist_camera import LensName, MeiParams, rescale

from specialist_dataset import schema
from specialist_dataset.poses import quats_to_matrices

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "Anchors",
  "KeyframePoses",
  "Lens",
  "Recording",
  "lens_from_row",
  "read_anchors",
  "read_keyframe_poses",
  "read_lens",
]

_CATALOG_DATASET = "dataset"


def _fields(table: pa.Table, prefix: str) -> dict[str, Any]:
  """Each column under ``prefix``, prefix stripped, to its last stored value."""
  out: dict[str, Any] = {}
  for name in table.schema.names:
    if name.startswith(prefix):
      values = table.column(name).combine_chunks().drop_null()
      if len(values):
        out[name[len(prefix) :]] = values[-1].as_py()
  return out


class Recording:
  """One dataset layer, served on 127.0.0.1 until its with-block ends or it is freed."""

  def __init__(self, rrd_path: Path) -> None:
    self.path = rrd_path
    self._server = rr.server.Server(
      host="127.0.0.1", datasets={_CATALOG_DATASET: [str(rrd_path)]}
    )
    self._entry = rr.catalog.CatalogClient(self._server.url()).get_dataset(
      _CATALOG_DATASET
    )

  def __enter__(self) -> Recording:
    return self

  def __exit__(self, *exception: object) -> None:
    self._server.shutdown()

  def reader(self, contents: str | list[str]) -> Any:
    """A query over the named entities, indexed by the recording's clock."""
    return self._entry.filter_contents(contents).reader(index=schema.TIMELINE)

  def series(self, contents: str | list[str]) -> pa.Table:
    """The named entities over the recording's clock, in index order."""
    table: pa.Table = self.reader(contents).to_arrow_table()
    return table.sort_by(schema.TIMELINE)

  def row(self, entity: str) -> dict[str, Any]:
    """Component name to untyped value for one entity's statics, as stored."""
    contents = self._entry.filter_contents(f"/{entity}")
    return _fields(contents.reader(index=None).to_arrow_table(), f"/{entity}:")

  def property(self, name: str) -> dict[str, Any]:
    """Field name to value for one recording property, as stored."""
    table = self._entry.segment_table().to_arrow_table()
    return _fields(table, f"property:{name}:")

  def values(
    self, entity: str, component: str, width: int | None = None
  ) -> tuple[Float64Array, Float64Array]:
    """One component's ``(times, values)`` in index order, values ``(rows, width)``."""
    column = schema.selector(entity, component)
    table = self.series(f"/{entity}")
    if column not in table.schema.names:
      raise ValueError(f"{self.path} has no {component} at /{entity}")
    array = table.column(column).combine_chunks()
    rows = table.filter(array.is_valid())
    array = array.drop_null()
    while pa.types.is_list(array.type) or pa.types.is_fixed_size_list(array.type):
      array = array.flatten()
    flat = np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)
    return self.seconds(rows), flat.reshape(rows.num_rows, width or -1)

  def scalars(
    self, entity: str, width: int | None = None
  ) -> tuple[Float64Array, Float64Array]:
    """One entity's scalar series: ``(times, values)`` in index order."""
    return self.values(entity, schema.SCALARS, width)

  def entity_paths(self) -> set[str]:
    """Every entity the recording holds, without a leading slash."""
    return {p.strip("/") for p in self._entry.schema().entity_paths()}

  def seconds(self, table: pa.Table) -> Float64Array:
    """The index column of a series, in seconds."""
    return schema.seconds(table.column(schema.TIMELINE))


@dataclass(frozen=True)
class KeyframePoses:
  """Each camera's pose at each keyframe."""

  slots: list[str]
  kf_times_by_slot: dict[str, Float64Array]
  kf_rotations_by_slot: dict[str, Float64Array]  # (n, 3, 3)
  kf_translations_by_slot: dict[str, Float64Array]  # (n, 3)


@dataclass(frozen=True)
class Anchors:
  """The solved anchor tags, and the id of the one the solve held fixed."""

  reference: int
  poses: dict[int, Float64Array]
  """Tag id to its ``(4, 4)`` world pose."""


@dataclass(frozen=True)
class Lens:
  """One lens at the decode resolution."""

  params: MeiParams
  body_T_lens: Float64Array  # (4, 4)


def _scalar(row: dict[str, Any], name: str) -> float:
  return float(np.asarray(row[name], dtype=np.float64).reshape(-1)[0])


def _pose(row: dict[str, Any]) -> Float64Array:
  """The pose one entity's statics carry, as ``(4, 4)``."""
  translation = np.asarray(row[schema.POSE_TRANSLATION], dtype=np.float64)
  pose = np.eye(4)
  pose[:3, :3] = quats_to_matrices(row[schema.POSE_QUATERNION])[0]
  pose[:3, 3] = translation.reshape(-1, 3)[0]
  return pose


def read_keyframe_poses(rrd_path: Path) -> KeyframePoses:
  """Load the estimate layer's keyframe poses and slots."""
  with Recording(rrd_path) as recording:
    paths = recording.entity_paths()
    slots = [s for s in schema.slots_in(paths) if schema.state(s, "pose") in paths]
    if not slots:
      raise ValueError(f"{rrd_path} holds no {schema.state('<slot>', 'pose')} entity")
    times, rotations, translations = {}, {}, {}
    for slot in slots:
      entity = schema.state(slot, "pose")
      times[slot], translations[slot] = recording.values(
        entity, schema.POSE_TRANSLATION, 3
      )
      rows, quaternions = recording.values(entity, schema.POSE_QUATERNION, 4)
      if not np.array_equal(rows, times[slot]):
        raise ValueError(f"{rrd_path}: /{entity} rotations and translations differ")
      rotations[slot] = quats_to_matrices(quaternions)
  return KeyframePoses(
    slots=slots,
    kf_times_by_slot=times,
    kf_rotations_by_slot=rotations,
    kf_translations_by_slot=translations,
  )


def lens_from_row(row: dict[str, Any], decoded_width: int, lens: LensName) -> Lens:
  """One stored lens entity as a lens at the decode resolution."""
  raw = MeiParams(**{f.name: _scalar(row, f.name) for f in fields(MeiParams)})
  return Lens(
    params=rescale(raw, _scalar(row, "frame_width"), decoded_width, lens),
    body_T_lens=_pose(row),
  )


def read_anchors(rrd_path: Path) -> Anchors:
  """The solved anchor poses by tag id, and which one the solve held fixed."""
  poses: dict[int, Float64Array] = {}
  reference: int | None = None
  with Recording(rrd_path) as recording:
    for entity in sorted(recording.entity_paths()):
      if entity.rpartition("/")[0] == f"{schema.WORLD}/anchor":
        row = recording.row(entity)
        if "tag_id" not in row:
          continue
        tag_id = int(_scalar(row, "tag_id"))
        poses[tag_id] = _pose(row)
        if _scalar(row, "is_reference"):
          reference = tag_id
  if reference is None:
    raise ValueError(f"{rrd_path} marks no anchor as the reference")
  return Anchors(reference=reference, poses=poses)


def read_lens(base_rrd: Path, slot: str, lens: LensName, decoded_width: int) -> Lens:
  """One lens of one base layer, at the resolution its frames decode to."""
  with Recording(base_rrd) as recording:
    row = recording.row(schema.lens(slot, lens))
  if schema.POSE_TRANSLATION not in row:
    raise ValueError(f"{base_rrd} holds no /{schema.lens(slot, lens)}")
  return lens_from_row(row, decoded_width, lens)
