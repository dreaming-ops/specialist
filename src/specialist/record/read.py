"""Read an estimate layer back into numpy: the scorers' view of a run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from rerun.experimental import Chunk, RrdReader

from specialist.record import schema
from specialist.vi_ba.geometry import quats_to_matrices

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "KeyframePoses",
  "check_frame_alignment",
  "read_keyframe_poses",
]

# Past this, a keyframe pairs its overlays with the wrong video frame.
_ALIGNMENT_TOLERANCE = 1e-3


def _paths_by_slot(paths: list[str], mint: Callable[[str], str]) -> dict[str, str]:
  """The recording's own spelling of one per-camera entity, per slot."""
  found = {}
  for path in paths:
    slot = schema.slot_of(path)
    if slot is not None and path.strip("/") == mint(slot):
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
    vectors("Transform3D:translation", 3),
    vectors("Transform3D:quaternion", 4),
  )


def check_frame_alignment(
  frame_times: dict[str, Float64Array], layer_rrd: Path
) -> None:
  """Raise ``ValueError`` unless every keyframe time lands on a frame of its camera."""
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
  state_paths = _paths_by_slot(paths, lambda slot: schema.state(slot, "pose"))
  rows_by_slot: dict[str, list[tuple[Float64Array, ...]]] = {}
  if state_paths:
    keyframe_stream = store.stream().filter(
      content=list(state_paths.values()),
      components=["Transform3D:translation", "Transform3D:quaternion"],
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
