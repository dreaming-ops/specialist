"""Offset between two captures from a cross correlation of their audio envelopes."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfiltfilt

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "AudioSyncResult",
  "MIN_PEAK_RATIO",
  "correlation_offset",
  "resolve_ffmpeg",
  "sync_two_captures",
]

_DEMUX_RATE = 16_000
_ENVELOPE_CUTOFF = 50.0
_DECIMATION = 16
# Largest clock offset (s) the correlation searches.
_MAX_OFFSET = 30.0
# Half-width (s) of the correlation peak's own lobe.
_LOBE_HALF_WIDTH = 0.25
# The offset needs a peak at least this many times its tallest rival.
MIN_PEAK_RATIO = 1.3


@dataclass(frozen=True)
class AudioSyncResult:
  """Alignment ``t_B = t_A + offset`` of two signals, and the peak's ratio to its top rival."""

  offset: float
  peak_ratio: float


@functools.cache
def resolve_ffmpeg() -> str:
  """The ffmpeg from ``$FFMPEG``, else ``imageio-ffmpeg``, else PATH; raises if none."""
  override = os.environ.get("FFMPEG")
  if override:
    return override
  try:
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())
  except Exception:  # any import or resolution failure falls through
    pass
  on_path = shutil.which("ffmpeg")
  if on_path is not None:
    return on_path
  raise RuntimeError(
    "ffmpeg not found: it ships via the imageio-ffmpeg dependency "
    "(run `uv sync`); or set $FFMPEG; or put ffmpeg on PATH"
  )


def _decode_mono_pcm(src: Path) -> Float64Array:
  """Mono PCM at ``_DEMUX_RATE`` from any container ffmpeg reads."""
  if not src.is_file():
    raise FileNotFoundError(f"audio source not found: {src}")
  cmd = [
    resolve_ffmpeg(),
    "-v",
    "error",
    "-i",
    str(src),
    "-ac",
    "1",
    "-ar",
    str(_DEMUX_RATE),
    "-f",
    "f32le",
    "-",
  ]
  result = subprocess.run(cmd, capture_output=True, check=False)
  if result.returncode != 0:
    raise RuntimeError(f"ffmpeg failed for {src}:\n{result.stderr.decode()}")
  pcm: Float64Array = np.frombuffer(result.stdout, dtype=np.float32).astype(np.float64)
  return pcm


def _envelope(samples: Float64Array, rate: int) -> Float64Array:
  """Rectified, lowpassed amplitude envelope."""
  sos = butter(4, _ENVELOPE_CUTOFF, btype="low", fs=rate, output="sos")
  envelope: Float64Array = sosfiltfilt(sos, np.abs(samples))
  return envelope


def correlation_offset(
  env_a: Float64Array, env_b: Float64Array, rate: int
) -> tuple[float, float]:
  """``(offset, peak_ratio)`` by cross correlation, with ``t_B = t_A + offset``."""
  a = np.asarray(env_a[::_DECIMATION], dtype=np.float64)
  b = np.asarray(env_b[::_DECIMATION], dtype=np.float64)
  if a.size < 2 or b.size < 2:
    raise ValueError("envelopes are too short to correlate")
  decimated_rate = rate / _DECIMATION
  a = a - a.mean()
  b = b - b.mean()
  # Zero padding biases the peak toward small offsets; captures far outlast offsets.
  size = 1 << int(np.ceil(np.log2(a.size + b.size)))
  cross = np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size))
  correlation = np.fft.fftshift(np.fft.irfft(cross, size))
  offsets = (np.arange(size) - size // 2) / decimated_rate
  candidates = np.where(np.abs(offsets) <= _MAX_OFFSET, correlation, -np.inf)
  peak = int(np.argmax(candidates))
  height = float(candidates[peak])
  shift = 0.0
  if 0 < peak < size - 1:
    left, right = float(correlation[peak - 1]), float(correlation[peak + 1])
    denominator = left - 2.0 * height + right
    if denominator != 0.0:
      shift = 0.5 * (left - right) / denominator
  lobe = np.abs(offsets - offsets[peak]) <= _LOBE_HALF_WIDTH
  best_rival = float(np.where(lobe, -np.inf, candidates).max())
  ratio = height / best_rival if best_rival > 0.0 else float("inf")
  return float(offsets[peak] + shift / decimated_rate), ratio


def sync_two_captures(source_a: Path, source_b: Path) -> AudioSyncResult:
  """Offset with ``t_B = t_A + offset``; raises RuntimeError on an ambiguous peak."""
  env_a = _envelope(_decode_mono_pcm(source_a), _DEMUX_RATE)
  env_b = _envelope(_decode_mono_pcm(source_b), _DEMUX_RATE)
  offset, peak_ratio = correlation_offset(env_a, env_b, _DEMUX_RATE)
  if peak_ratio < MIN_PEAK_RATIO:
    raise RuntimeError(
      f"audio alignment is ambiguous: the correlation peak is only "
      f"{peak_ratio:.2f} times its tallest rival, so the captures may share no "
      f"audible event; pass the offset explicitly if it is known"
    )
  return AudioSyncResult(offset=offset, peak_ratio=peak_ratio)
