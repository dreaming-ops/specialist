"""Fallback alignment must reject plausible but ambiguous tap sequences."""

import numpy as np
import pytest
from specialist_dataset.episodes import Marker

from specialist.loader.imu_sync import match_markers, sync_two_imus


def markers(starts: list[float], offset: float = 0) -> list[Marker]:
  return [Marker(t + offset, t + offset + 0.5) for t in starts]


def test_recovers_negative_offset_with_missing_and_extra_markers() -> None:
  a = markers([2, 67, 83, 156, 170, 260, 990])
  b = markers([2, 67, 156, 170, 260, 990], -1.87) + markers([420])
  result = match_markers(a, b, duration=1040)
  assert result.offset == pytest.approx(-1.87)
  assert result.matches == 6
  assert result.confidence >= 1.5
  assert result.residual_seconds < 1e-9


def test_periodic_sequence_is_rejected() -> None:
  a = markers([2, 12, 22, 32, 42, 52, 62])
  b = markers([2, 12, 22, 32, 42, 52, 62], -1.87)
  with pytest.raises(ValueError, match="ambiguous"):
    match_markers(a, b, duration=65)


@pytest.mark.parametrize("starts", [[], [2], [2, 67]])
def test_too_few_markers_are_rejected(starts: list[float]) -> None:
  with pytest.raises(ValueError, match="at least three"):
    match_markers(markers(starts), markers(starts, -1.87), duration=1040)


def test_markers_only_near_start_cannot_validate_long_capture() -> None:
  starts = [2, 67, 83, 156]
  with pytest.raises(ValueError, match="span half"):
    match_markers(markers(starts), markers(starts, -1.87), duration=1040)


def test_excessive_clock_drift_is_rejected() -> None:
  starts = [2, 67, 156, 990]
  a = markers(starts)
  b = [
    Marker(m.first + delta, m.last + delta)
    for m, delta in zip(a, [-1.92, -1.87, -1.87, -1.82], strict=True)
  ]
  with pytest.raises(ValueError, match="drift|span half|ambiguous"):
    match_markers(a, b, duration=1040)


def test_out_of_range_offset_is_rejected() -> None:
  with pytest.raises(ValueError):
    match_markers(markers([2, 67, 156, 990]), markers([2, 67, 156, 990], 200), 1040)


def test_sensor_streams_recover_shift_and_ignore_weak_rebounds() -> None:
  t = np.arange(0, 110, 0.002)
  a = np.zeros((len(t), 3))
  a[:, 2] = 9.81
  b = a.copy()
  for start in [2, 17, 49, 103]:
    for tap in [0, 0.25, 0.5]:
      for data, shift in [(a, 0), (b, -1.87)]:
        data[np.argmin(abs(t - (start + tap + shift))), 2] = 210
        data[np.argmin(abs(t - (start + tap + shift + 0.14))), 2] = 30
  result = sync_two_imus(t, a, t, b)
  assert result.offset == pytest.approx(-1.87, abs=0.002)
  assert result.matches == 4
