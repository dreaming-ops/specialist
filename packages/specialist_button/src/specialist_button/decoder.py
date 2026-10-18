"""Find every marker frame in a recording and read its source id and event."""

import itertools
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import fft as sfft
from scipy import signal

from specialist_button.frame import (
  BANDS,
  BASE,
  CHIRP_DURATION,
  N_DIGITS,
  PREAMBLE_DURATION,
  SLOT_DURATION,
  chirp_phase,
  sub_band,
)

FS_DECODE = 16_000
SLOT_WINDOW = 0.15  # each slot is measured this long from its start
NFFT = 8192
START_PAD = 8  # leading zeros, so a frame at the first sample still has a peak
CLOCK_ERRORS = tuple(float(e) for e in np.arange(-0.002, 0.00201, 0.0005))

CODEWORDS = np.array(
  [
    (a, b, e, (a + b + e) % BASE)
    for a, b, e in itertools.product(range(BASE), repeat=3)
  ]
)

Floats = NDArray[np.float64]
Complexes = NDArray[np.complex128]


@dataclass(frozen=True)
class DecoderConfig:
  threshold: float = 0.25  # smallest correlation of each preamble chirp
  pair_tolerance: float = 0.0005  # error of the down-to-up peak spacing
  pair_search: float = 0.003  # half-width of the search for the up peak
  min_share: float = 0.4  # smallest share of a reported frame
  margin_db: float = 1.0  # best over second-best codeword of an unflagged frame
  collision_ratio: float = 0.7  # rival amplitude, re a frame's, that flags it
  collision_gap: float = 0.02  # closer preambles are one frame and its echoes
  merge_gap: float = 0.003  # closer candidates are one correlation peak
  min_separation: float = 0.4  # a closer frame is a second frame or a copy
  second_share: float = 0.6  # smallest share of a second frame
  band_margin: float = 50.0  # raised-cosine edge outside each band


DEFAULT_CONFIG = DecoderConfig()


@dataclass(frozen=True)
class Detection:
  onset: float  # start of the frame from the start of the recording
  source_id: int
  event: int
  score: float  # the smaller of the down and the up correlation
  flags: tuple[str, ...]  # "ambiguous", "collision"
  accepted: bool  # the receiver's own source and no flag
  digits: tuple[int, ...]  # d1, d2, event, check
  share: float  # mean share of the slot energy in the decoded sub-bands
  margin_db: float  # best over second-best codeword
  down_score: float
  up_score: float
  pair_error: float  # down-to-up peak spacing minus the chirp length
  clock_error: float  # clock error of the best-matching template


class _Preamble(NamedTuple):
  score: float
  down: int  # sample of the down-chirp peak
  up: int  # sample of the up-chirp peak
  clock_error: float
  down_track: Floats
  up_track: Floats


@cache
def _templates(clock_error: float = 0.0) -> tuple[Complexes, Complexes]:
  """The down and up chirps at `FS_DECODE` as a clock fast by `clock_error` plays them."""
  n = round(CHIRP_DURATION / (1 + clock_error) * FS_DECODE)
  t = np.arange(n) / FS_DECODE * (1 + clock_error)
  down, up = (np.exp(1j * chirp_phase(t, CHIRP_DURATION, r)) for r in (True, False))
  return down, up


def _band_weights(n: int, margin: float) -> Floats:
  """Weight of every bin of an n-point transform: 1 inside the bands, 0 at negative frequencies."""
  f = sfft.fftfreq(n, 1 / FS_DECODE)
  w = np.zeros(n)
  for lo, hi in BANDS:
    inside = np.minimum(f - (lo - margin), (hi + margin) - f) / margin
    w = np.maximum(w, 0.5 - 0.5 * np.cos(np.pi * np.clip(inside, 0.0, 1.0)))
  return w


def _analytic_in_band(x: Floats, margin: float) -> Complexes:
  """Analytic signal of the part of `x` inside the two bands."""
  n = sfft.next_fast_len(len(x))
  xa: Complexes = sfft.ifft(2 * sfft.fft(x, n) * _band_weights(n, margin))[: len(x)]
  return xa


def _correlations(xa: Complexes, chirps: list[Complexes]) -> list[Floats]:
  """Normalized correlation of `xa` with each template at every start sample, over one shared transform."""
  n = sfft.next_fast_len(len(xa))
  spectrum = sfft.fft(xa, n)
  cumulative = np.concatenate([[0.0], np.cumsum(np.abs(xa) ** 2)])
  tracks = []
  for template in chirps:
    m = len(template)
    dot = sfft.ifft(spectrum * np.conj(sfft.fft(template, n)))[: len(xa) - m + 1]
    energy = cumulative[m:] - cumulative[:-m]
    floor = 1e-12 * energy.max()
    if floor <= 0.0:
      tracks.append(np.zeros(len(dot)))
      continue
    norms = np.sqrt(np.maximum(energy, floor)) * np.linalg.norm(template)
    tracks.append(np.where(energy > floor, np.abs(dot) / norms, 0.0))
  return tracks


def _pair_peaks(
  down: Floats, up: Floats, gap: int, config: DecoderConfig
) -> list[tuple[float, int, int]]:
  """(score, down peak, up peak) of every down peak followed by an up peak `gap` samples later."""
  search = round(config.pair_search * FS_DECODE)
  tolerance = config.pair_tolerance * FS_DECODE
  pairs = []
  for p in signal.find_peaks(down, height=config.threshold)[0]:
    a = p + gap - search
    if a + 2 * search >= len(up):
      continue
    q = a + int(np.argmax(up[a : a + 2 * search + 1]))
    if up[q] >= config.threshold and abs(q - p - gap) <= tolerance:
      pairs.append((float(min(down[p], up[q])), int(p), q))
  return pairs


def _find_preambles(xa: Complexes, config: DecoderConfig) -> list[_Preamble]:
  """Preambles in `xa` over the clock-error templates, one per correlation peak, strongest first."""
  chirps = [_templates(clock_error) for clock_error in CLOCK_ERRORS]
  if len(xa) < max(len(down) + len(up) for down, up in chirps):
    return []
  tracks = _correlations(xa, [template for pair in chirps for template in pair])
  candidates = [
    _Preamble(score, p, q, clock_error, tracks[2 * k], tracks[2 * k + 1])
    for k, clock_error in enumerate(CLOCK_ERRORS)
    for score, p, q in _pair_peaks(
      tracks[2 * k], tracks[2 * k + 1], len(chirps[k][0]), config
    )
  ]
  separation = config.merge_gap * FS_DECODE
  kept: list[_Preamble] = []
  for candidate in sorted(candidates, key=lambda c: -c.score):
    if all(abs(candidate.down - other.down) >= separation for other in kept):
      kept.append(candidate)
  return kept


@cache
def _sub_band_bins() -> NDArray[np.bool_]:
  """Bin mask of every sub-band, indexed band, digit, bin."""
  f = sfft.fftfreq(NFFT, 1 / FS_DECODE)
  return np.array(
    [
      [(f >= lo) & (f < hi) for lo, hi in (sub_band(b, k) for k in range(BASE))]
      for b in range(len(BANDS))
    ]
  )


def _sub_band_energy(segment: Complexes) -> Floats:
  """Energy of a stretch of the analytic signal in every sub-band, indexed band, digit."""
  spectrum = np.abs(sfft.fft(segment * np.hanning(len(segment)), NFFT)) ** 2
  result: Floats = _sub_band_bins() @ spectrum
  return result


def _read_digits(xa: Complexes, t0: int) -> tuple[tuple[int, ...], float, float]:
  """(digits, share, margin in dB) of the frame whose preamble starts at sample `t0`."""
  start = t0 + round(PREAMBLE_DURATION * FS_DECODE)
  gain = _sub_band_energy(xa[t0:start]) + 1e-30
  n, window = round(SLOT_DURATION * FS_DECODE), round(SLOT_WINDOW * FS_DECODE)
  slots = np.zeros((N_DIGITS, len(BANDS), BASE))
  for i in range(N_DIGITS):
    for band in range(len(BANDS)):
      a = start + (len(BANDS) * i + band) * n
      e = _sub_band_energy(xa[a : a + window])[band] / gain[band]
      slots[i, band] = e / e.sum()
  fit = slots.mean(axis=1)[np.arange(N_DIGITS), CODEWORDS].mean(axis=1)
  order = np.argsort(fit)
  best, second = fit[order[-1]], fit[order[-2]]
  margin = 10 * np.log10(best / max(second, 1e-30))
  return tuple(int(v) for v in CODEWORDS[order[-1]]), float(best), float(margin)


def _amplitude(xa: Complexes, p: int, clock_error: float) -> float:
  """Amplitude of the down chirp that starts at sample `p`."""
  template = _templates(clock_error)[0]
  return float(np.abs(np.vdot(template, xa[p : p + len(template)])) / len(template))


def is_repeat(
  frame: Detection, near: list[Detection], config: DecoderConfig = DEFAULT_CONFIG
) -> bool:
  """Whether `frame` is a copy of one of the frames `near` it rather than a second frame."""
  second = (
    not frame.flags
    and frame.share >= config.second_share
    and all(other.digits != frame.digits for other in near)
  )
  return bool(near) and not second


def _select_frames(
  detections: list[Detection], config: DecoderConfig
) -> list[Detection]:
  """The frames to report, unflagged first and by falling share, then in time order."""
  kept: list[Detection] = []
  for d in sorted(detections, key=lambda d: (bool(d.flags), -d.share)):
    near = [k for k in kept if abs(k.onset - d.onset) < config.min_separation]
    if not is_repeat(d, near, config):
      kept.append(d)
  return sorted(kept, key=lambda d: d.onset)


def _peak_offset(y: Floats, i: int) -> float:
  """Offset of the vertex of the parabola through y[i-1], y[i], y[i+1]."""
  if not 0 < i < len(y) - 1:
    return 0.0
  curvature = y[i - 1] - 2 * y[i] + y[i + 1]
  if curvature == 0:
    return 0.0
  return float(np.clip(0.5 * (y[i - 1] - y[i + 1]) / curvature, -0.5, 0.5))


def _to_decode_rate(audio: ArrayLike, fs: int) -> Floats:
  """Mono copy of `audio` at `FS_DECODE`; of channel columns, the first counts."""
  x = np.asarray(audio, dtype=float)
  if x.ndim > 1:
    x = x[:, 0]
  ratio = Fraction(FS_DECODE, int(fs))
  if ratio == 1:
    return x
  resampled: Floats = signal.resample_poly(x, ratio.numerator, ratio.denominator)
  return resampled


def decode(
  audio: ArrayLike,
  fs: int,
  own_source_id: int | None = None,
  config: DecoderConfig = DEFAULT_CONFIG,
) -> list[Detection]:
  """Every frame that lies whole inside `audio`, sampled at `fs`, in time order."""
  slot, window = round(SLOT_DURATION * FS_DECODE), round(SLOT_WINDOW * FS_DECODE)
  x = np.pad(_to_decode_rate(audio, fs), (START_PAD, window - slot))
  xa = _analytic_in_band(x, config.band_margin)
  frame_end = (
    round(PREAMBLE_DURATION * FS_DECODE) + (N_DIGITS * len(BANDS) - 1) * slot + window
  )
  preambles = [p for p in _find_preambles(xa, config) if p.down + frame_end <= len(xa)]
  read = [
    (p, _read_digits(xa, p.down), _amplitude(xa, p.down, p.clock_error))
    for p in preambles
  ]
  detections = []
  for p, (digits, share, margin), size in read:
    if share < config.min_share:
      continue
    collides = any(
      config.collision_gap * FS_DECODE <= abs(other.down - p.down) < frame_end
      and other_digits != digits
      and other_size >= config.collision_ratio * size
      for other, (other_digits, _, _), other_size in read
    )
    d1, d2, event, _ = digits
    flags = ("ambiguous",) * (margin < config.margin_db) + ("collision",) * collides
    source_id = BASE * d1 + d2
    detections.append(
      Detection(
        onset=(p.down - START_PAD + _peak_offset(p.down_track, p.down)) / FS_DECODE,
        source_id=source_id,
        event=event,
        score=p.score,
        flags=flags,
        accepted=source_id == own_source_id and not flags,
        digits=digits,
        share=share,
        margin_db=margin,
        down_score=float(p.down_track[p.down]),
        up_score=float(p.up_track[p.up]),
        pair_error=float(
          (p.up - p.down) / FS_DECODE - CHIRP_DURATION / (1 + p.clock_error)
        ),
        clock_error=p.clock_error,
      )
    )
  return _select_frames(detections, config)
