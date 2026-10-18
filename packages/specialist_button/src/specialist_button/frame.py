"""The marker frame: a down and an up chirp, then four digits sent by frequency."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

FS = 48_000
BANDS = ((800.0, 1150.0), (3250.0, 3700.0))  # L and H, the bands a button reproduces
CHIRP_DURATION = 0.25
SLOT_DURATION = 0.1
SWEEP = 0.7  # share of its sub-band that a digit slot sweeps
BASE = 6  # values of one digit, and sub-bands per band
N_DIGITS = 4  # d1, d2, event, check
N_SOURCES = BASE**2
N_EVENTS = BASE
RAMP_DURATION = 0.005
PREAMBLE_DURATION = 2 * CHIRP_DURATION
FRAME_DURATION = PREAMBLE_DURATION + N_DIGITS * len(BANDS) * SLOT_DURATION

# The joined axis places the bands end to end, from 0 at the start of L.
_BAND_START = np.array([lo for lo, _ in BANDS])
_AXIS_START = np.concatenate([[0.0], np.cumsum([hi - lo for lo, hi in BANDS])])
AXIS_WIDTH = float(_AXIS_START[-1])


def _axis_integral(u: NDArray[np.float64]) -> NDArray[np.float64]:
  """Integral of the frequency along the joined axis from 0 to `u`."""
  total = np.zeros_like(u)
  for f0, a, b in zip(_BAND_START, _AXIS_START[:-1], _AXIS_START[1:], strict=True):
    v = np.clip(u, a, b) - a
    total += f0 * v + 0.5 * v**2
  return total


def chirp_phase(
  t: ArrayLike, duration: float, reverse: bool = False
) -> NDArray[np.float64]:
  """Phase at times `t` of the chirp that crosses the joined axis once, upward or reversed."""
  rate = AXIS_WIDTH / duration
  t = np.asarray(t, dtype=float)
  u = AXIS_WIDTH - rate * t if reverse else rate * t
  if reverse:
    cycles = _axis_integral(np.asarray(AXIS_WIDTH)) - _axis_integral(u)
  else:
    cycles = _axis_integral(u)
  return 2 * np.pi * cycles / rate


def sub_band(band: int, digit: int) -> tuple[float, float]:
  """Edges of sub-band `digit` of band `band`."""
  lo, hi = BANDS[band]
  width = (hi - lo) / BASE
  return lo + digit * width, lo + (digit + 1) * width


def slot_frequency(band: int, digit: int) -> NDArray[np.float64]:
  """Instantaneous frequency of the slot that sends `digit` in band `band`."""
  lo, hi = sub_band(band, digit)
  t = np.arange(round(SLOT_DURATION * FS)) / FS
  return lo + (hi - lo) * ((1 - SWEEP) / 2 + SWEEP * t / SLOT_DURATION)


def frame_digits(source_id: int, event: int) -> tuple[int, int, int, int]:
  """The digits d1, d2, event and check of a frame."""
  if not 0 <= source_id < N_SOURCES:
    raise ValueError(f"source id {source_id} is outside 0..{N_SOURCES - 1}")
  if not 0 <= event < N_EVENTS:
    raise ValueError(f"event {event} is outside 0..{N_EVENTS - 1}")
  d1, d2 = divmod(source_id, BASE)
  return d1, d2, event, (d1 + d2 + event) % BASE


def generate_frame(source_id: int, event: int) -> NDArray[np.float64]:
  """Phase-continuous frame waveform at `FS` with peak 1 and a raised-cosine ramp at each end."""
  t = np.arange(round(CHIRP_DURATION * FS)) / FS
  parts, phase = [], 0.0
  for reverse in (True, False):
    parts.append(phase + chirp_phase(t, CHIRP_DURATION, reverse))
    phase += float(chirp_phase(CHIRP_DURATION, CHIRP_DURATION, reverse))
  for digit in frame_digits(source_id, event):
    for band in range(len(BANDS)):
      cycles = np.cumsum(slot_frequency(band, digit)) / FS
      parts.append(phase + 2 * np.pi * np.concatenate([[0.0], cycles[:-1]]))
      phase += 2 * np.pi * cycles[-1]
  x = np.sin(np.concatenate(parts))
  n = round(RAMP_DURATION * FS)
  ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
  x[:n] *= ramp
  x[-n:] *= ramp[::-1]
  return x
