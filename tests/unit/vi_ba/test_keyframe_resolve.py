"""The keyframe stride resolves to a fixed ~0.10 s interval regardless of fps."""

from __future__ import annotations

from specialist.vi_ba.pipeline import resolve_keyframe_every


def test_auto_targets_tenth_second() -> None:
  assert resolve_keyframe_every(0, 60.0) == 6  # 0.10s
  assert resolve_keyframe_every(0, 30.0) == 3  # 0.10s
  assert resolve_keyframe_every(0, 59.94) == 6  # NTSC 60
  assert resolve_keyframe_every(0, 29.97) == 3  # NTSC 30


def test_explicit_value_overrides_auto() -> None:
  assert resolve_keyframe_every(2, 60.0) == 2
  assert resolve_keyframe_every(9, 30.0) == 9


def test_floors_at_one_for_very_low_fps() -> None:
  assert resolve_keyframe_every(0, 5.0) == 1
