"""Audio sync: envelope correlation and two-capture offsets."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from specialist.loader.audio_sync import (
  _envelope,
  correlation_offset,
  resolve_ffmpeg,
  sync_two_captures,
)


def _burst_signal(
  rate: int, duration: float, onsets: list[float], seed: int
) -> np.ndarray:
  """Noise bursts at ``onsets`` over a quiet floor, with a waveform set by ``seed``."""
  rng = np.random.default_rng(seed)
  samples = rng.normal(0.0, 0.001, int(duration * rate))
  for onset in onsets:
    start = int(onset * rate)
    span = int(0.02 * rate)
    samples[start : start + span] += rng.normal(0.0, 1.0, span)
  return samples


def _write_wav(path: Path, samples: np.ndarray, rate: int) -> Path:
  """Write mono samples as a 16-bit wav through the loader's own ffmpeg."""
  subprocess.run(
    [
      resolve_ffmpeg(),
      "-y",
      "-loglevel",
      "error",
      "-f",
      "f32le",
      "-ar",
      str(rate),
      "-ac",
      "1",
      "-i",
      "-",
      "-c:a",
      "pcm_s16le",
      str(path),
    ],
    input=(0.25 * samples).astype(np.float32).tobytes(),
    check=True,
  )
  return path


def test_correlation_offset_recovers_a_known_offset() -> None:
  """B hears the same bursts 0.4 s later on its own clock."""
  rate = 16000
  onsets = [1.0, 2.7, 5.3, 8.9]
  a = _burst_signal(rate, 12.0, onsets, seed=1)
  b = _burst_signal(rate, 12.0, [t + 0.4 for t in onsets], seed=2)
  offset, ratio = correlation_offset(_envelope(a, rate), _envelope(b, rate), rate)
  assert abs(offset - 0.4) < 0.005
  assert ratio > 1.5


def test_correlation_offset_survives_a_missing_transient() -> None:
  """A transient too faint for any onset threshold still carries correlation weight."""
  rate = 16000
  onsets = [1.0, 4.0, 7.5]
  a = _burst_signal(rate, 10.0, onsets, seed=3)
  b = _burst_signal(rate, 10.0, [t + 0.25 for t in onsets], seed=4)
  b[int(4.25 * rate) : int(4.27 * rate)] *= 0.05
  offset, _ = correlation_offset(_envelope(a, rate), _envelope(b, rate), rate)
  assert abs(offset - 0.25) < 0.01


def test_correlation_ratio_drops_on_a_periodic_pattern() -> None:
  """Evenly spaced bursts put a rival peak one spacing away."""
  rate = 16000
  onsets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  a = _burst_signal(rate, 9.0, onsets, seed=5)
  b = _burst_signal(rate, 9.0, [t + 0.3 for t in onsets], seed=6)
  offset, ratio = correlation_offset(_envelope(a, rate), _envelope(b, rate), rate)
  assert abs(offset - 0.3) < 0.01
  assert ratio < 2.0


def test_sync_two_captures_recovers_a_known_offset(tmp_path: Path) -> None:
  """Two wav captures share only the bursts, B's copy 0.35 s later on its own clock."""
  rate = 16000
  onsets = [0.7, 2.3, 4.9, 6.1]
  shift = 0.35
  wav_a = _write_wav(
    tmp_path / "a.wav", _burst_signal(rate, 8.0, onsets, seed=11), rate
  )
  wav_b = _write_wav(
    tmp_path / "b.wav",
    _burst_signal(rate, 8.0, [t + shift for t in onsets], seed=12),
    rate,
  )
  result = sync_two_captures(wav_a, wav_b)
  assert result.offset == pytest.approx(shift, abs=0.01)
  assert result.peak_ratio > 1.3


def test_sync_two_captures_rejects_captures_sharing_nothing(tmp_path: Path) -> None:
  rate = 16000
  wav_a = _write_wav(tmp_path / "a.wav", _burst_signal(rate, 8.0, [], seed=21), rate)
  wav_b = _write_wav(tmp_path / "b.wav", _burst_signal(rate, 8.0, [], seed=22), rate)
  with pytest.raises(RuntimeError, match="alignment is ambiguous"):
    sync_two_captures(wav_a, wav_b)
