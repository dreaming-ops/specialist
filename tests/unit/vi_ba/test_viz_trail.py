"""The live trajectory strip shows only a trailing time window per keyframe."""

from __future__ import annotations

import numpy as np

from specialist.vi_ba.viz import _trail_strips


def test_strips_window_to_trailing_duration() -> None:
  traj = np.arange(18, dtype=np.float64).reshape(6, 3)
  times = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 30.0])
  strips = _trail_strips(traj, times, duration=2.0)
  assert len(strips) == 6
  # Keyframe i always ends its own strip.
  for i, strip in enumerate(strips):
    assert np.array_equal(strip[-1], traj[i])
  # Window [t-2, t]: kf2 keeps kf0..2, kf3 stands alone, kf4 keeps kf3..4.
  assert np.array_equal(strips[2], traj[0:3])
  assert np.array_equal(strips[3], traj[3:4])
  assert np.array_equal(strips[4], traj[3:5])
  assert np.array_equal(strips[5], traj[5:6])


def test_full_history_when_duration_covers_clip() -> None:
  traj = np.zeros((4, 3), dtype=np.float64)
  times = np.array([0.0, 0.5, 1.0, 1.5])
  strips = _trail_strips(traj, times, duration=100.0)
  for i, strip in enumerate(strips):
    assert strip.shape == (i + 1, 3)
