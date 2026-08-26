"""Read an estimate layer into numpy, refusing a layer of another schema version."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from rerun.experimental import Chunk, RrdReader
from specialist_camera import LensName, MeiParams, rescale

from specialist_dataset import schema
from specialist_dataset.poses import quats_to_matrices

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "KeyframePoses",
  "Lens",
  "check_frame_alignment",
  "lens_from_row",
  "newest_values",
  "read_keyframe_poses",
  "read_lens",
]

# Past this, a keyframe pairs its overlays with the wrong video frame.
_ALIGNMENT_TOLERANCE = 1e-3


def _state_pose_paths(paths: list[str]) -> dict[str, str]:
  """The recording's own spelling of each camera's keyframe-pose entity."""
  found = {}
  for path in paths:
    slot = schema.slot_of(path)
    if slot is not None and path.strip("/") == schema.state(slot, "pose"):
      found[slot] = path
  return found


@dataclass(frozen=True)
class KeyframePoses:
  """Each camera's pose at each keyframe."""

  slots: list[str]
  kf_times_by_slot: dict[str, Float64Array]
  kf_rotations_by_slot: dict[str, Float64Array]  # (n, 3, 3)
  kf_translations_by_slot: dict[str, Float64Array]  # (n, 3)


def _pose_rows(chunk: Chunk) -> tuple[Float64Array, Float64Array, Float64Array]:
  """(times, translations, quats_xyzw) of one temporal Transform3D chunk."""
  batch = chunk.to_record_batch()
  names = batch.schema.names

  def vectors(component: str, width: int) -> Float64Array:
    # The writer stores one instance per row, so both nesting levels flatten away.
    values = batch.column(names.index(component))
    flat = values.flatten().flatten().to_numpy(zero_copy_only=False)
    out: Float64Array = flat.astype(np.float64).reshape(len(values), width)
    return out

  return (
    schema.seconds(batch.column(names.index(schema.TIMELINE))),
    vectors(schema.POSE_TRANSLATION, 3),
    vectors(schema.POSE_QUATERNION, 4),
  )


def check_frame_alignment(
  frame_times: dict[str, Float64Array], layer_rrd: Path
) -> None:
  """Assert each camera's keyframe timestamps land on its frame times."""
  run = read_keyframe_poses(layer_rrd)
  for slot, kf_times in run.kf_times_by_slot.items():
    if slot not in frame_times:
      raise ValueError(f"{slot} has no video frame times")
    times = frame_times[slot]
    positions = np.searchsorted(times, kf_times)
    lo = times[np.clip(positions - 1, 0, len(times) - 1)]
    hi = times[np.clip(positions, 0, len(times) - 1)]
    nearest = np.minimum(np.abs(kf_times - lo), np.abs(kf_times - hi))
    worst = float(nearest.max())
    if worst > _ALIGNMENT_TOLERANCE:
      raise ValueError(
        f"{slot}: keyframe times miss the video frames by up to "
        f"{worst * 1e3:.2f} ms (tolerance {_ALIGNMENT_TOLERANCE * 1e3:.2f} ms)"
      )


def read_keyframe_poses(rrd_path: Path) -> KeyframePoses:
  """Load the estimate layer's keyframe poses and slots."""
  store = RrdReader(str(rrd_path)).store()
  paths = store.schema().entity_paths()
  state_paths = _state_pose_paths(paths)
  if not state_paths:
    raise ValueError(f"{rrd_path} holds no {schema.state('<slot>', 'pose')} entity")
  rows_by_slot: dict[str, list[tuple[Float64Array, ...]]] = {}
  keyframe_stream = store.stream().filter(
    content=list(state_paths.values()),
    components=[schema.POSE_TRANSLATION, schema.POSE_QUATERNION],
  )
  by_path = {path: slot for slot, path in state_paths.items()}
  for chunk in keyframe_stream.to_chunks():
    rows_by_slot.setdefault(by_path[chunk.entity_path], []).append(_pose_rows(chunk))
  kf_times = {}
  kf_rotations = {}
  kf_translations = {}
  for slot, rows in rows_by_slot.items():
    times, translations, quats = (np.concatenate(c) for c in zip(*rows, strict=True))
    order = np.argsort(times, kind="stable")
    kf_times[slot] = times[order]
    kf_translations[slot] = translations[order]
    kf_rotations[slot] = quats_to_matrices(quats[order])
  return KeyframePoses(
    slots=sorted(kf_times),
    kf_times_by_slot=kf_times,
    kf_rotations_by_slot=kf_rotations,
    kf_translations_by_slot=kf_translations,
  )


def newest_values(chunks: Iterable[Chunk]) -> dict[str, Any]:
  """Component name -> the value with the highest row id over ``chunks``."""
  out: dict[str, Any] = {}
  newest: dict[str, bytes] = {}
  for chunk in chunks:
    batch = chunk.to_record_batch()
    names = batch.schema.names
    row_ids = batch.column(names.index("rerun.controls.RowId")).to_pylist()
    for name, column in zip(names, batch.columns, strict=True):
      if name.startswith("rerun.controls.") or name == schema.TIMELINE:
        continue
      for i, value in enumerate(column.to_pylist()):
        if value is not None and row_ids[i] > newest.get(name, b""):
          newest[name] = row_ids[i]
          out[name] = value
  return out


@dataclass(frozen=True)
class Lens:
  """One lens at the decode resolution."""

  params: MeiParams
  body_T_lens: Float64Array  # (4, 4)


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


def read_lens(base_rrd: Path, slot: str, lens: LensName, decoded_width: int) -> Lens:
  """One lens of one base layer, at the resolution its frames decode to."""
  entity = schema.lens(slot, lens)
  store = RrdReader(str(base_rrd)).store()
  row = newest_values(store.stream().filter(content=f"/{entity}").to_chunks())
  if not row:
    raise ValueError(f"{base_rrd} holds no /{entity}")
  return lens_from_row(row, decoded_width, lens)
