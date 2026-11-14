"""Conservative clock alignment from matching triple-tap IMU markers."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from specialist_dataset.episodes import Marker, TapConfig, triple_taps

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ImuSyncResult:
  offset: float  # t_B = t_A + offset
  confidence: float  # matched marker count / strongest competing alignment count
  matches: int
  residual_seconds: float


def match_markers(
  a: list[Marker],
  b: list[Marker],
  duration: float,
  max_offset: float = 30.0,
) -> ImuSyncResult:
  """Require a unique consensus, coverage across the capture and low jitter.

  This does not silently accept an arbitrary best correlation. At least three
  complete triples spanning half the capture must agree, with no near-equal
  competing lag and no endpoint residual over 40 ms.
  """
  if len(a) < 3 or len(b) < 3 or duration <= 0:
    raise ValueError("IMU alignment needs at least three shared triple-tap groups")
  first = np.array([[y.first - x.first for y in b] for x in a])
  last = np.array([[y.last - x.last for y in b] for x in a])
  centers = (first + last) / 2
  candidates = centers[(np.abs(first - last) <= 0.08) & (np.abs(centers) <= max_offset)]

  def pairs(offset: float) -> list[tuple[int, int]]:
    mask = (np.abs(first - offset) <= 0.06) & (np.abs(last - offset) <= 0.06)
    # An event must have exactly one partner in both directions.
    unique = mask & (mask.sum(axis=1)[:, None] == 1) & (mask.sum(axis=0)[None, :] == 1)
    return [(int(i), int(j)) for i, j in zip(*np.where(unique), strict=True)]

  scored = [(len(pairs(float(c))), float(c)) for c in candidates]
  if not scored:
    raise ValueError("no matching IMU triples within the allowed clock offset")
  count, guess = max(scored)
  matching = pairs(guess)
  rival = max((n for n, c in scored if abs(c - guess) > 0.2), default=0)
  confidence = count / max(1, rival)
  if count < 3 or confidence < 1.5:
    raise ValueError("IMU alignment is ambiguous between competing marker sequences")
  deltas = np.array([d[i, j] for i, j in matching for d in (first, last)])
  offset = float(np.median(deltas))
  residual = float(np.max(np.abs(deltas - offset)))
  span = max(a[i].last for i, _ in matching) - min(a[i].first for i, _ in matching)
  if span < duration * 0.5:
    raise ValueError("shared IMU triples do not span half the capture")
  if residual > 0.04:
    raise ValueError(
      "IMU clock drift or impact jitter exceeds 40 ms; manual alignment needed"
    )
  return ImuSyncResult(offset, confidence, count, residual)


def sync_two_imus(t_a: Array, a: Array, t_b: Array, b: Array) -> ImuSyncResult:
  markers_a = triple_taps(t_a, a, TapConfig())
  markers_b = triple_taps(t_b, b, TapConfig())
  return match_markers(
    markers_a, markers_b, min(float(t_a[-1] - t_a[0]), float(t_b[-1] - t_b[0]))
  )
