"""Conservative IMU triple-tap candidates and reviewable episode boundaries."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TapConfig:
  # Physical acceleration excess and robust z-score must BOTH pass.
  min_excess: float = 8.0
  z_threshold: float = 12.0
  debounce: float = 0.12
  max_gap: float = 0.65
  camera_tolerance: float = 0.15
  # Ignore small contacts relative to the strongest impact in each burst.
  relative_strength: float = 0.25


@dataclass(frozen=True)
class Marker:
  first: float
  last: float


@dataclass(frozen=True)
class Episode:
  episode_id: str
  capture_start: float
  task_end: float
  capture_end: float
  task: str = ""


def triple_taps(t: Array, accel: Array, config: TapConfig) -> list[Marker]:
  """Exactly three debounced impacts per isolated burst; flat signals are valid."""
  if (
    len(t) < 2
    or accel.shape != (len(t), 3)
    or not np.isfinite(t).all()
    or not np.isfinite(accel).all()
    or np.any(np.diff(t) <= 0)
  ):
    raise ValueError("IMU needs finite, increasing times and Nx3 acceleration")
  if not (
    0 < config.debounce < config.max_gap
    and config.min_excess > 0
    and config.z_threshold > 0
    and config.camera_tolerance > 0
    and 0 <= config.relative_strength < 1
  ):
    raise ValueError("invalid tap thresholds")
  norm = np.linalg.norm(accel, axis=1)
  center = float(np.median(norm))
  spread = 1.4826 * float(np.median(np.abs(norm - center)))
  threshold = max(config.min_excess, config.z_threshold * spread)
  edges = np.flatnonzero(np.diff(np.r_[False, norm - center > threshold, False]))
  hits: list[tuple[float, float]] = []
  for lo, hi in zip(edges[::2], edges[1::2], strict=True):
    index = lo + np.argmax(norm[lo:hi])
    hits.append((float(t[index]), float(norm[index] - center)))
  bursts: list[list[tuple[float, float]]] = []
  for hit in hits:
    if not bursts or hit[0] - bursts[-1][-1][0] > config.max_gap:
      bursts.append([])
    bursts[-1].append(hit)
  markers = []
  for burst in bursts:
    strongest = max(strength for _, strength in burst)
    kept: list[float] = []
    # Select the strongest impact first, rather than an earlier weak precursor.
    for at, strength in sorted(burst, key=lambda hit: (-hit[1], hit[0])):
      if strength >= strongest * config.relative_strength and all(
        abs(at - previous) >= config.debounce for previous in kept
      ):
        kept.append(at)
    kept.sort()
    if len(kept) == 3 and max(np.diff(kept)) <= config.max_gap:
      markers.append(Marker(kept[0], kept[-1]))
  return markers


def detect(
  times: dict[str, Array],
  accelerations: dict[str, Array],
  config: TapConfig | None = None,
) -> tuple[list[Marker], list[str]]:
  """Match isolated triples across every camera; report unmatched candidates."""
  if not times or set(times) != set(accelerations):
    raise ValueError("one acceleration stream is required for each camera")
  config = config or TapConfig()
  candidates = {s: triple_taps(t, accelerations[s], config) for s, t in times.items()}
  slots = list(times)
  used: dict[str, set[int]] = {s: set() for s in slots}
  shared: list[Marker] = []
  for i, marker in enumerate(candidates[slots[0]]):
    matches = {slots[0]: i}
    for slot in slots[1:]:
      options = [
        j
        for j, other in enumerate(candidates[slot])
        if j not in used[slot]
        and abs(other.first - marker.first) <= config.camera_tolerance
        and abs(other.last - marker.last) <= config.camera_tolerance
      ]
      if len(options) == 1:
        matches[slot] = options[0]
    if len(matches) == len(slots):
      group = [candidates[s][j] for s, j in matches.items()]
      shared.append(Marker(min(m.first for m in group), max(m.last for m in group)))
      for s, j in matches.items():
        used[s].add(j)
  warnings = [
    f"{s}: unmatched triple at {m.first:.3f}s"
    for s in slots
    for j, m in enumerate(candidates[s])
    if j not in used[s]
  ]
  return shared, warnings


def propose(markers: list[Marker], window: tuple[float, float]) -> list[Episode]:
  """Pair start/end markers. An unclosed start is never silently discarded."""
  if not markers or len(markers) % 2:
    raise ValueError(f"expected a nonzero even marker count, found {len(markers)}")
  episodes = [
    Episode(
      f"ep{i // 2 + 1:04d}", start.last + 0.2, end.first, min(end.last + 0.2, window[1])
    )
    for i in range(0, len(markers), 2)
    for start, end in [(markers[i], markers[i + 1])]
  ]
  validate(episodes, window)
  return episodes


def validate(episodes: list[Episode], window: tuple[float, float]) -> None:
  import re

  previous = window[0]
  ids: set[str] = set()
  if not episodes:
    raise ValueError("no episodes selected")
  for e in episodes:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", e.episode_id) or e.episode_id in ids:
      raise ValueError("episode IDs must be unique, safe filename components")
    if (
      not all(math.isfinite(v) for v in (e.capture_start, e.task_end, e.capture_end))
      or not previous <= e.capture_start < e.task_end < e.capture_end <= window[1]
      or e.task_end - e.capture_start < 2.0
    ):
      raise ValueError(f"{e.episode_id}: invalid, overlapping or too short boundaries")
    previous = e.capture_end
    ids.add(e.episode_id)


def read_manifest(
  path: Path, session_id: str, window: tuple[float, float]
) -> list[Episode]:
  data = json.loads(path.read_text(encoding="utf-8"))
  if data.get("version") != 1 or data.get("session_id") != session_id:
    raise ValueError("manifest belongs to a different session or format version")
  if data.get("reviewed") is not True:
    raise ValueError("review the candidate boundaries, then set reviewed to true")
  episodes = [Episode(**e) for e in data["episodes"]]
  validate(episodes, window)
  return episodes


def manifest(
  session_id: str, markers: list[Marker], episodes: list[Episode], warnings: list[str]
) -> dict[str, object]:
  return dict(
    version=1,
    session_id=session_id,
    reviewed=False,
    markers=[asdict(m) for m in markers],
    episodes=[asdict(e) for e in episodes],
    warnings=warnings,
  )
