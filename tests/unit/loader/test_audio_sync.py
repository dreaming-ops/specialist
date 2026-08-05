"""Tests for ffmpeg resolution in the audio-sync loader."""

from __future__ import annotations

import numpy as np
import pytest

from specialist.loader.audio_sync import _best_lag, resolve_ffmpeg


def test_resolve_ffmpeg_honors_env_override(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """``$FFMPEG`` takes priority so a specific binary can be forced."""
  resolve_ffmpeg.cache_clear()
  monkeypatch.setenv("FFMPEG", "/custom/path/ffmpeg")
  assert resolve_ffmpeg() == "/custom/path/ffmpeg"
  resolve_ffmpeg.cache_clear()


def test_best_lag_recovers_known_offset() -> None:
  """A clean, non-periodic clap train shifted by a known offset returns it."""
  a = np.array([1.0, 4.0, 9.5])
  lag, n = _best_lag(a, a + 0.75, match_tolerance=0.05)
  assert abs(lag - 0.75) < 1e-6
  assert n == 3


def test_best_lag_periodic_claps_picks_consensus_not_sidelobe() -> None:
  """The lag is the offset most clap pairs agree on, not a side-lobe one clap away."""
  a = np.array([5.636, 15.807, 16.134, 16.468])
  b = np.array([4.502, 15.815, 16.146, 16.451])
  lag, n = _best_lag(a, b, match_tolerance=0.1)
  assert abs(lag) < 0.05, f"expected ~0 offset, got {lag}"
  assert n >= 3
