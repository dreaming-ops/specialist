"""Read a track layer back into numpy: the scorers' view of a run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from rerun.experimental import Chunk, RrdReader

from specialist.record import schema
from specialist.record.session import newest_values
from specialist.vi_ba.geometry import quats_to_matrices

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "TrackRun",
  "check_frame_alignment",
  "read_track",
]

# Past this, a keyframe pairs its overlays with the wrong video frame.
_ALIGNMENT_TOLERANCE = 1e-3

_KEYFRAMES_SUFFIX = f"/{schema.BODY_KEYFRAMES}"
_BASE_TAG_SUFFIX = f"/{schema.BASE_TAG}"


@dataclass(frozen=True)
class TrackRun:
  """One run's track as the scorers consume it."""

  slots: list[str]
  kf_times_by_tracker: dict[str, Float64Array]
  kf_rotations_by_tracker: dict[str, Float64Array]  # (n, 3, 3)
  kf_translations_by_tracker: dict[str, Float64Array]  # (n, 3)
  body_T_base_tag_by_tracker: dict[str, Float64Array]  # (4, 4), identity if absent


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
    vectors("Transform3D:translation", 3),
    vectors("Transform3D:quaternion", 4),
  )


def _pose_matrix(row: dict[str, Any]) -> Float64Array:
  """One static Transform3D row as a 4x4 homogeneous pose."""
  quat = np.asarray(row["Transform3D:quaternion"], dtype=np.float64).reshape(1, 4)
  pose = np.eye(4)
  pose[:3, :3] = quats_to_matrices(quat)[0]
  pose[:3, 3] = np.asarray(row["Transform3D:translation"], dtype=np.float64).reshape(3)
  return pose


def check_frame_alignment(
  frame_times: dict[str, Float64Array], layer_rrd: Path
) -> None:
  """Raises ValueError when a tracker's keyframe times miss its video frame times."""
  run = read_track(layer_rrd)
  for tracker, kf_times in run.kf_times_by_tracker.items():
    if tracker not in frame_times:
      raise ValueError(f"{tracker} has no video frame times")
    times = frame_times[tracker]
    positions = np.searchsorted(times, kf_times)
    lo = times[np.clip(positions - 1, 0, len(times) - 1)]
    hi = times[np.clip(positions, 0, len(times) - 1)]
    nearest = np.minimum(np.abs(kf_times - lo), np.abs(kf_times - hi))
    worst = float(nearest.max())
    if worst > _ALIGNMENT_TOLERANCE:
      raise ValueError(
        f"{tracker}: keyframe times miss the video frames by up to "
        f"{worst * 1e3:.2f} ms (tolerance {_ALIGNMENT_TOLERANCE * 1e3:.2f} ms)"
      )


def read_track(rrd_path: Path) -> TrackRun:
  """Load the track layer's keyframe poses, names, and base-tag frames."""
  store = RrdReader(str(rrd_path)).store()
  paths = store.schema().entity_paths()
  kf_paths = [p for p in paths if p.endswith(_KEYFRAMES_SUFFIX)]
  rows_by_tracker: dict[str, list[tuple[Float64Array, ...]]] = {}
  keyframe_stream = store.stream().filter(
    content=kf_paths,
    components=["Transform3D:translation", "Transform3D:quaternion"],
  )
  for chunk in keyframe_stream.to_chunks() if kf_paths else ():
    tracker = chunk.entity_path.strip("/").removesuffix(_KEYFRAMES_SUFFIX)
    rows_by_tracker.setdefault(tracker, []).append(_pose_rows(chunk))
  base_tag_by_tracker: dict[str, Float64Array] = {}
  for path in (p for p in paths if p.endswith(_BASE_TAG_SUFFIX)):
    # No component projection: it can drop the chunk holding the winning row.
    stream = store.stream().filter(content=path, is_static=True)
    row = newest_values(stream.to_chunks())
    if "Transform3D:translation" in row:
      tracker = path.strip("/").removesuffix(_BASE_TAG_SUFFIX)
      base_tag_by_tracker[tracker] = _pose_matrix(row)

  kf_times = {}
  kf_rotations = {}
  kf_translations = {}
  for tracker, rows in rows_by_tracker.items():
    times, translations, quats = (np.concatenate(c) for c in zip(*rows, strict=True))
    order = np.argsort(times, kind="stable")
    kf_times[tracker] = times[order]
    kf_translations[tracker] = translations[order]
    kf_rotations[tracker] = quats_to_matrices(quats[order])
  return TrackRun(
    slots=sorted(kf_times),
    kf_times_by_tracker=kf_times,
    kf_rotations_by_tracker=kf_rotations,
    kf_translations_by_tracker=kf_translations,
    body_T_base_tag_by_tracker={
      name: base_tag_by_tracker.get(name, np.eye(4)) for name in kf_times
    },
  )
