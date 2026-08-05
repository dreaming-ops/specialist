"""Read a track layer back into numpy: the scorers' view of a run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from rerun.experimental import Chunk, RrdReader

from specialist.record import schema
from specialist.vi_ba.geometry import quats_to_matrices

Float64Array = npt.NDArray[np.float64]

__all__ = ["TrackRun", "read_track"]

_NS = 1e-9


@dataclass(frozen=True)
class TrackRun:
  """One run's track as the scorers consume it."""

  slots: list[str]
  kf_times_by_tracker: dict[str, Float64Array]
  kf_rotations_by_tracker: dict[str, Float64Array]  # (n, 3, 3)
  kf_translations_by_tracker: dict[str, Float64Array]  # (n, 3)
  body_T_base_tag_by_tracker: dict[str, Float64Array]  # (4, 4), identity if absent


def _column_values(chunk: Chunk, component: str) -> list[Any] | None:
  batch = chunk.to_record_batch()
  names = batch.schema.names
  if component not in names:
    return None
  return list(batch.column(names.index(component)))


def _pose_rows(chunk: Chunk) -> tuple[Float64Array, Float64Array, Float64Array]:
  """(times, translations, quats_xyzw) of one temporal Transform3D chunk."""
  batch = chunk.to_record_batch()
  names = batch.schema.names
  times = np.array(
    [v.value * _NS for v in batch.column(names.index(schema.TIMELINE))],
    dtype=np.float64,
  )
  translations = np.array(
    [
      np.asarray(v, dtype=np.float64).reshape(3)
      for v in batch.column(names.index("Transform3D:translation")).to_pylist()
    ]
  )
  quats = np.array(
    [
      np.asarray(v, dtype=np.float64).reshape(4)
      for v in batch.column(names.index("Transform3D:quaternion")).to_pylist()
    ]
  )
  return times, translations, quats


def read_track(rrd_path: Path) -> TrackRun:
  """Load the track layer's keyframe poses, names, and base-tag frames."""
  reader = RrdReader(str(rrd_path))
  times_by_tracker: dict[str, list[Float64Array]] = {}
  translations_by_tracker: dict[str, list[Float64Array]] = {}
  quats_by_tracker: dict[str, list[Float64Array]] = {}
  base_tag_by_tracker: dict[str, Float64Array] = {}
  slots: list[str] = []

  for chunk in reader.store().stream().to_chunks():
    path = chunk.entity_path
    if path == "/__properties/run":
      names = _column_values(chunk, "slots")
      if names:
        first = names[0]
        slots = [str(n) for n in first.as_py()]
      continue
    if path.endswith("/body_keyframes"):
      tracker = path.strip("/").removesuffix("/body_keyframes")
      t, p, q = _pose_rows(chunk)
      times_by_tracker.setdefault(tracker, []).append(t)
      translations_by_tracker.setdefault(tracker, []).append(p)
      quats_by_tracker.setdefault(tracker, []).append(q)
      continue
    if path.endswith("/body/base_tag") and chunk.is_static:
      tracker = path.strip("/").removesuffix("/body/base_tag")
      batch = chunk.to_record_batch()
      names_ = batch.schema.names
      if "Transform3D:translation" not in names_:
        continue
      t_col = batch.column(names_.index("Transform3D:translation")).to_pylist()
      q_col = batch.column(names_.index("Transform3D:quaternion")).to_pylist()
      translation = np.asarray(t_col[0], dtype=np.float64).reshape(3)
      quat = np.asarray(q_col[0], dtype=np.float64).reshape(1, 4)
      pose = np.eye(4)
      pose[:3, :3] = quats_to_matrices(quat)[0]
      pose[:3, 3] = translation
      base_tag_by_tracker[tracker] = pose

  kf_times = {}
  kf_rotations = {}
  kf_translations = {}
  for tracker, chunks_times in times_by_tracker.items():
    times = np.concatenate(chunks_times)
    order = np.argsort(times, kind="stable")
    kf_times[tracker] = times[order]
    kf_translations[tracker] = np.concatenate(translations_by_tracker[tracker])[order]
    quats = np.concatenate(quats_by_tracker[tracker])[order]
    kf_rotations[tracker] = quats_to_matrices(quats)
  if not slots:
    slots = sorted(kf_times)
  return TrackRun(
    slots=slots,
    kf_times_by_tracker=kf_times,
    kf_rotations_by_tracker=kf_rotations,
    kf_translations_by_tracker=kf_translations,
    body_T_base_tag_by_tracker={
      name: base_tag_by_tracker.get(name, np.eye(4)) for name in kf_times
    },
  )
