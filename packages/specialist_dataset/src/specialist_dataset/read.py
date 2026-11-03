"""Read a dataset layer as numpy through the selectors ``schema`` mints."""

from __future__ import annotations

import re
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
_ANCHOR = re.compile(rf"/{schema.WORLD}/anchor/(\d+):")


def _values(table: pa.Table, column: str, width: int | None = None) -> Float64Array:
  """One column's flattened values; rows without the component drop out."""
  array = table.column(column).combine_chunks()
  array = array.drop_null()
  while pa.types.is_list(array.type) or pa.types.is_fixed_size_list(array.type):
    array = array.flatten()
  flat = np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)
  return flat.reshape(-1, width) if width else flat


def _rows_with(table: pa.Table, column: str) -> pa.Table:
  """The rows where one column carried a value."""
  return table.filter(pa.compute.is_valid(table.column(column).combine_chunks()))


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

  def statics(self, contents: str | list[str]) -> pa.Table:
    """Every static component of the named entities, one row."""
    return self._entry.filter_contents(contents).reader(index=None).to_arrow_table()

  def series(self, contents: str | list[str]) -> pa.Table:
    """The named entities over the recording's clock, in index order."""
    table = (
      self._entry.filter_contents(contents)
      .reader(index=schema.TIMELINE)
      .to_arrow_table()
    )
    return table.sort_by(schema.TIMELINE)

  def row(self, entity: str) -> dict[str, Any]:
    """Component name to untyped value for one entity's statics, as stored."""
    table = self.statics(f"/{entity}")
    prefix = f"/{entity}:"
    out: dict[str, Any] = {}
    for name in table.schema.names:
      if not name.startswith(prefix):
        continue
      values = table.column(name).combine_chunks().drop_null().to_pylist()
      if values:
        out[name[len(prefix) :]] = values[-1]
    return out

  def property(self, name: str) -> dict[str, Any]:
    """Field name to value for one recording property, as stored."""
    table = self._entry.segment_table().to_arrow_table()
    prefix = f"property:{name}:"
    out: dict[str, Any] = {}
    for column in table.schema.names:
      if column.startswith(prefix):
        values = table.column(column).combine_chunks().drop_null().to_pylist()
        if values:
          out[column[len(prefix) :]] = values[0]
    return out

  def values(
    self, entity: str, component: str, width: int | None = None
  ) -> tuple[Float64Array, Float64Array]:
    """One component's ``(times, values)`` in index order, values ``(rows, width)``."""
    column = schema.selector(entity, component)
    table = self.series(f"/{entity}")
    if column not in table.schema.names:
      raise ValueError(f"{self.path} has no {component} at /{entity}")
    rows = _rows_with(table, column)
    flat = _values(rows, column)
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


def _pose_at(table: pa.Table, entity: str) -> Float64Array:
  """The pose one entity carries, as ``(4, 4)``."""
  translation = schema.selector(entity, schema.POSE_TRANSLATION)
  if translation not in table.schema.names:
    raise ValueError(f"no pose at /{entity}")
  pose = np.eye(4)
  pose[:3, :3] = quats_to_matrices(
    _values(table, schema.selector(entity, schema.POSE_QUATERNION), 4)
  )[0]
  pose[:3, 3] = _values(table, translation, 3)[0]
  return pose


def read_keyframe_poses(rrd_path: Path) -> KeyframePoses:
  """Load the estimate layer's keyframe poses and slots."""
  with Recording(rrd_path) as recording:
    slots = sorted(
      slot
      for slot in schema.slots_in(recording.entity_paths())
      if schema.state(slot, "pose") in recording.entity_paths()
    )
    if not slots:
      raise ValueError(f"{rrd_path} holds no {schema.state('<slot>', 'pose')} entity")
    kf_times, kf_rotations, kf_translations = {}, {}, {}
    for slot in slots:
      entity = schema.state(slot, "pose")
      translation = schema.selector(entity, schema.POSE_TRANSLATION)
      table = _rows_with(recording.series(f"/{entity}"), translation)
      kf_times[slot] = recording.seconds(table)
      kf_translations[slot] = _values(table, translation, 3)
      kf_rotations[slot] = quats_to_matrices(
        _values(table, schema.selector(entity, schema.POSE_QUATERNION), 4)
      )
  return KeyframePoses(
    slots=slots,
    kf_times_by_slot=kf_times,
    kf_rotations_by_slot=kf_rotations,
    kf_translations_by_slot=kf_translations,
  )


def lens_from_row(row: dict[str, Any], decoded_width: int, lens: LensName) -> Lens:
  """One stored lens entity as a lens at the decode resolution."""

  def scalar(name: str) -> float:
    return float(np.asarray(row[name], dtype=np.float64).reshape(-1)[0])

  raw = MeiParams(**{f.name: scalar(f.name) for f in fields(MeiParams)})
  translation = np.asarray(row[schema.POSE_TRANSLATION], dtype=np.float64)
  body_T_lens = np.eye(4)
  body_T_lens[:3, :3] = quats_to_matrices(row[schema.POSE_QUATERNION])[0]
  body_T_lens[:3, 3] = translation.reshape(-1, 3)[0]
  return Lens(
    params=rescale(raw, scalar("frame_width"), decoded_width, lens),
    body_T_lens=body_T_lens,
  )


def read_anchors(rrd_path: Path) -> Anchors:
  """The solved anchor poses by tag id, and which one the solve held fixed."""
  poses: dict[int, Float64Array] = {}
  reference: int | None = None
  with Recording(rrd_path) as recording:
    table = recording.statics(f"/{schema.WORLD}/anchor/**")
    entities = {
      name.split(":", 1)[0].strip("/")
      for name in table.schema.names
      if _ANCHOR.match(name)
    }
    for entity in sorted(entities):
      tag_id = int(_values(table, schema.selector(entity, "tag_id"))[0])
      poses[tag_id] = _pose_at(table, entity)
      if bool(_values(table, schema.selector(entity, "is_reference"))[0]):
        reference = tag_id
  if reference is None:
    raise ValueError(f"{rrd_path} marks no anchor as the reference")
  return Anchors(reference=reference, poses=poses)


def read_lens(base_rrd: Path, slot: str, lens: LensName, decoded_width: int) -> Lens:
  """One lens of one base layer, at the resolution its frames decode to."""
  entity = schema.lens(slot, lens)
  with Recording(base_rrd) as recording:
    table = recording.statics(f"/{entity}")
    if schema.selector(entity, schema.POSE_TRANSLATION) not in table.schema.names:
      raise ValueError(f"{base_rrd} holds no /{entity}")
    row = {
      name.split(":", 1)[1]: _values(table, name)
      for name in table.schema.names
      if name.startswith(f"/{entity}:")
    }
  return lens_from_row(row, decoded_width, lens)
