"""The keyframe stride, its session-clock grid, and the finger stride that divides it."""

from __future__ import annotations

import numpy as np
import pytest
from specialist_dataset import schema

from specialist.vi_ba.single_camera import finger_stride, keyframe_grid


def test_stride_targets_a_tenth_second() -> None:
  assert schema.keyframe_stride(60.0) == 6  # 0.10s
  assert schema.keyframe_stride(30.0) == 3  # 0.10s
  assert schema.keyframe_stride(59.94) == 6  # NTSC 60
  assert schema.keyframe_stride(29.97) == 3  # NTSC 30


def test_floors_at_one_for_very_low_fps() -> None:
  assert schema.keyframe_stride(5.0) == 1


def test_grid_snaps_to_the_stride_lattice() -> None:
  # Coverage starts at sample 4, so sample 6, the first on-lattice one, sets the phase.
  times = np.arange(20) / 60.0
  covered = np.arange(4, 20)
  samples, _ = keyframe_grid(times, covered, fps=60.0)
  assert samples.tolist() == [6, 12, 18]


def test_grid_without_a_lattice_sample_starts_at_coverage() -> None:
  # Coverage so short no covered sample hits the lattice: fall back to phase 0.
  times = np.arange(20) / 60.0
  covered = np.arange(7, 11)
  samples, _ = keyframe_grid(times, covered, fps=60.0)
  assert samples.tolist() == [7]


@pytest.mark.parametrize(
  ("keyframe_stride", "fps", "expected"),
  [
    (6, 60.0, 2),  # divisors 1,2,3,6; 30 Hz cap allows 2
    (3, 30.0, 1),  # cap is 1 at native 30 fps
    (3, 29.97, 1),  # NTSC: fps/30 < 1, default keeps every frame
    (4, 60.0, 2),  # 3 would hit the cap but does not divide 4
    (5, 60.0, 1),  # no divisor of 5 fits under the cap
  ],
)
def test_finger_stride_divides_the_keyframe_stride(
  keyframe_stride: int, fps: float, expected: int
) -> None:
  stride = finger_stride(keyframe_stride, fps)
  assert stride == expected
  assert keyframe_stride % stride == 0
