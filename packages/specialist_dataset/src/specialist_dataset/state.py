"""Numpy state-stream helpers: kept frames, row lattice, opening, hands and gestures."""

import math
from functools import reduce

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

__all__ = [
  "finger",
  "hands",
  "hold",
  "image_frames",
  "lattice_brackets",
  "lerp",
  "nearest",
  "opening",
  "tap",
]

_MAX_BRACKET = 0.0025
_PAIR_TOLERANCE = 0.05
_MIN_ANGLE = 3.0
_MIN_AGREEMENT = 0.75
_HOLD_SPEED = 0.15
_HOLD_MIN_DURATION = 0.4
_HOLD_HEAD = 0.1
_TAP_Z = 12.0
_TAP_BURST_GAP = 0.5
_TAP_TAIL = 0.9


def hold(
  times: Float64Array,
  anchor_distance: Float64Array,
  speed: Float64Array,
  span: tuple[float, float],
) -> float:
  """When the start hold ends; refuses one too short or ending past the first tenth."""
  slow = speed <= _HOLD_SPEED
  if not slow.any():
    raise ValueError("the gripper never slows near an anchor")
  i = int(np.argmin(np.where(slow, anchor_distance, np.inf)))
  lo, hi = i, i
  while lo > 0 and slow[lo - 1]:
    lo -= 1
  while hi < len(slow) - 1 and slow[hi + 1]:
    hi += 1
  if times[hi] - times[lo] < _HOLD_MIN_DURATION:
    raise ValueError(
      f"the closest slow anchor approach lasts {times[hi] - times[lo]:.2f} s, "
      f"under the {_HOLD_MIN_DURATION} s hold"
    )
  at = (times[hi] - span[0]) / (span[1] - span[0])
  if at > _HOLD_HEAD:
    raise ValueError(f"the hold ends at {at:.0%} of the capture, past its first tenth")
  return float(times[hi])


def _spike_times(
  times: Float64Array, accel: Float64Array
) -> tuple[Float64Array, Float64Array]:
  """Peak time and robust z-score of each acceleration-norm run above ``_TAP_Z``."""
  norm = np.linalg.norm(accel, axis=1)
  spread = 1.4826 * np.median(np.abs(norm - np.median(norm)))
  z = (norm - np.median(norm)) / spread
  above = np.concatenate(([False], z > _TAP_Z, [False]))
  edges = np.flatnonzero(np.diff(above.astype(np.int8)))
  peaks = np.array(
    [
      lo + int(np.argmax(z[lo:hi]))
      for lo, hi in zip(edges[::2], edges[1::2], strict=True)
    ],
    dtype=np.int64,
  )
  return times[peaks], z[peaks]


def tap(times: dict[str, Float64Array], accel: dict[str, Float64Array]) -> float:
  """When the end tap starts; refuses a tap before the recording's last tenth."""
  bursts: list[list[tuple[float, str, float]]] = [[]]
  for event in sorted(
    (t, slot, score)
    for slot in times
    for t, score in zip(*_spike_times(times[slot], accel[slot]), strict=True)
  ):
    if bursts[-1] and event[0] - bursts[-1][-1][0] > _TAP_BURST_GAP:
      bursts.append([])
    bursts[-1].append(event)
  shared = [b for b in bursts if {e[1] for e in b} == set(times)]
  if not shared:
    raise ValueError("no burst of acceleration spikes reaches every camera")
  strongest = max(
    shared,
    key=lambda b: min(max(e[2] for e in b if e[1] == slot) for slot in times),
  )
  t0 = min(float(t[0]) for t in times.values())
  t1 = max(float(t[-1]) for t in times.values())
  strike = strongest[0][0]
  if strike < t0 + _TAP_TAIL * (t1 - t0):
    raise ValueError(
      f"the strongest tap burst sits at {(strike - t0) / (t1 - t0):.0%} of "
      "the recording, before its last tenth"
    )
  return float(strike)


def nearest(
  times: Float64Array, queries: Float64Array
) -> tuple[Int64Array, Float64Array]:
  """The index into ascending ``times`` nearest each query, and the gap."""
  if not len(times):
    raise ValueError("no times to search")
  hi = np.clip(np.searchsorted(times, queries), 0, len(times) - 1)
  lo = np.clip(hi - 1, 0, len(times) - 1)
  index = np.where(np.abs(times[hi] - queries) < np.abs(times[lo] - queries), hi, lo)
  return index.astype(np.int64), np.abs(times[index] - queries)


def image_frames(
  times: dict[str, Float64Array], fps: float, window: tuple[float, float]
) -> dict[str, Int64Array]:
  """Each lens's indices of the session frames every lens has, inside ``window``."""
  session = {s: np.rint(np.asarray(t) * fps).astype(np.int64) for s, t in times.items()}
  wanted = reduce(np.intersect1d, session.values())
  wanted = wanted[
    (wanted >= math.ceil(window[0] * fps - 0.5))
    & (wanted <= math.floor(window[1] * fps + 0.5))
  ]
  if not wanted.size:
    raise ValueError(f"no frame of every lens falls in the window {window} s")
  return {
    s: np.searchsorted(grid, wanted).astype(np.int64) for s, grid in session.items()
  }


def lattice_brackets(
  times: Float64Array, f_row: float, span: tuple[float, float]
) -> tuple[Int64Array, Int64Array, Float64Array]:
  """``(rows, before, weight)`` per lattice row in ``span``; refuses wide gaps."""
  t = np.asarray(times, dtype=np.float64)
  if len(t) < 2 or (np.diff(t) <= 0.0).any():
    raise ValueError("sample times are not strictly increasing")
  rows = np.arange(
    math.ceil(max(t[0], span[0]) * f_row), math.floor(min(t[-1], span[1]) * f_row) + 1
  )
  if not rows.size:
    raise ValueError(f"no row falls in the window {span} s")
  instants = rows / f_row
  before = np.clip(np.searchsorted(t, instants, side="right") - 1, 0, len(t) - 2)
  width = t[before + 1] - t[before]
  if (wide := width > _MAX_BRACKET).any():
    raise ValueError(
      f"rows {rows[wide][:6].tolist()} fall in a gap of {width.max() * 1e3:.1f} ms "
      "between samples"
    )
  return rows, before, (instants - t[before]) / width


def lerp(
  values: Float64Array, before: Int64Array, weight: Float64Array
) -> Float64Array:
  """``values`` linearly interpolated at the instants ``lattice_brackets`` located."""
  v = np.asarray(values, dtype=np.float64)
  w = weight.reshape(-1, *(1,) * (v.ndim - 1))
  out: Float64Array = v[before] + w * (v[before + 1] - v[before])
  return out


def finger(
  times: Float64Array, angles: Float64Array
) -> tuple[Float64Array, Float64Array]:
  """One finger's ``(times, angles)`` cut to the finite rows, where its tag is seen."""
  t = np.asarray(times, dtype=np.float64)
  a = np.asarray(angles, dtype=np.float64)
  keep = np.isfinite(a)
  if not keep.any():
    raise ValueError("the finger was never seen")
  return t[keep], a[keep]


def opening(
  left: tuple[Float64Array, Float64Array],
  right: tuple[Float64Array, Float64Array],
  at: Float64Array,
) -> Float64Array:
  """Coupled angle ``(left - right) / 2`` at ``at``, refusing rows past either span."""
  (t_left, a_left), (t_right, a_right) = left, right
  if at[0] < max(t_left[0], t_right[0]) or at[-1] > min(t_left[-1], t_right[-1]):
    raise ValueError("a row falls outside the solved fingers' span")
  return (np.interp(at, t_left, a_left) - np.interp(at, t_right, a_right)) / 2.0


def hands(
  times: dict[str, Float64Array], translations: dict[str, Float64Array]
) -> tuple[str, str]:
  """``(left, right)`` slots, left counter-clockwise seen from the world origin."""
  if len(times) != 2:
    raise ValueError(f"hand detection needs two cameras, got {sorted(times)}")
  a, b = sorted(times)
  index, gap = nearest(np.asarray(times[b], float), np.asarray(times[a], float))
  paired = gap <= _PAIR_TOLERANCE
  pa = np.asarray(translations[a], dtype=np.float64)[paired, :2]
  pb = np.asarray(translations[b], dtype=np.float64)[index[paired], :2]
  norm = np.linalg.norm(pa, axis=1) * np.linalg.norm(pb, axis=1)
  sine = (pa[:, 0] * pb[:, 1] - pa[:, 1] * pb[:, 0])[norm > 0] / norm[norm > 0]
  if not sine.size:
    raise ValueError(f"{a} and {b} share no keyframes to compare")
  mean = float(sine.mean())
  angle = math.degrees(math.asin(max(-1.0, min(1.0, mean))))
  agreement = float(np.mean(np.sign(sine) == np.sign(mean)))
  if abs(angle) < _MIN_ANGLE or agreement < _MIN_AGREEMENT:
    raise ValueError(
      f"cannot tell the hands apart: seen from the anchor {b} sits {angle:+.1f} deg "
      f"from {a}, and the keyframes agree on the side {agreement:.0%} of the time"
    )
  return (b, a) if mean > 0.0 else (a, b)
