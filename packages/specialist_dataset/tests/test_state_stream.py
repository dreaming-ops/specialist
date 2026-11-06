"""The rows a state stream keeps, and the hand each camera rode."""

import numpy as np
import numpy.typing as npt
import pytest
from specialist_dataset.state import (
  finger,
  hands,
  hold,
  image_frames,
  lattice_brackets,
  lerp,
  nearest,
  opening,
  tap,
)

Float64Array = npt.NDArray[np.float64]


def test_lattice_brackets_place_each_row_between_its_two_samples() -> None:
  times = np.array([0.0002, 0.0012, 0.0021, 0.0032, 0.0044])
  rows, before, weight = lattice_brackets(times, 1000.0, (0.0, 1.0))
  np.testing.assert_array_equal(rows, [1, 2, 3, 4])
  np.testing.assert_array_equal(before, [0, 1, 2, 3])
  np.testing.assert_allclose(weight, [0.8, 0.8 / 0.9, 0.9 / 1.1, 0.8 / 1.2])
  # A series linear in time comes back exactly at the instants, whatever its shape.
  np.testing.assert_allclose(lerp(times * 1000.0, before, weight), [1.0, 2.0, 3.0, 4.0])
  np.testing.assert_allclose(
    lerp(np.stack([times, -times], axis=1) * 1000.0, before, weight)[:, 1],
    [-1.0, -2.0, -3.0, -4.0],
  )
  # A row on a sample is that sample.
  rows, before, weight = lattice_brackets(
    np.array([0.0, 0.001, 0.002]), 1000.0, (0.0, 1.0)
  )
  np.testing.assert_array_equal(rows, [0, 1, 2])
  np.testing.assert_allclose(weight, [0.0, 0.0, 1.0])
  # Only the rows inside the span are served, and a gap outside it is no gap.
  rows, _, _ = lattice_brackets(
    np.array([0.0, 0.001, 0.004, 0.005]), 1000.0, (0.0035, 0.01)
  )
  np.testing.assert_array_equal(rows, [4, 5])
  with pytest.raises(ValueError, match="gap"):
    lattice_brackets(np.array([0.0, 0.001, 0.004]), 1000.0, (0.0, 1.0))
  with pytest.raises(ValueError, match="strictly increasing"):
    lattice_brackets(np.array([0.0, 0.001, 0.001]), 1000.0, (0.0, 1.0))
  with pytest.raises(ValueError, match="no row"):
    lattice_brackets(times, 1000.0, (0.0041, 0.0042))


def test_nearest_refuses_nothing_to_search() -> None:
  with pytest.raises(ValueError, match="no times"):
    nearest(np.array([]), np.array([1.0]))


def test_opening_is_half_the_mirrored_finger_difference() -> None:
  t = np.arange(4) * 0.1
  left = finger(t, np.array([0.2, 0.3, 0.1, 0.0]))
  right = finger(t, -np.array([0.2, 0.3, 0.1, 0.0]))
  np.testing.assert_allclose(opening(left, right, t), [0.2, 0.3, 0.1, 0.0])
  # A row one finger missed interpolates from that finger's own neighboring rows.
  nan = float("nan")
  left = finger(t, np.array([0.2, nan, 0.0, 0.0]))
  np.testing.assert_allclose(left[0], t[[0, 2, 3]])
  np.testing.assert_allclose(
    opening(left, right, np.array([0.1])), [(0.1 - (-0.3)) / 2]
  )
  # Never past either finger's span.
  with pytest.raises(ValueError, match="span"):
    opening(left, finger(t[:3], np.zeros(3)), t)
  with pytest.raises(ValueError, match="never seen"):
    finger(t, np.full(4, nan))


def test_image_frames_keeps_every_frame_inside_the_window() -> None:
  fps = 30.0
  times = (np.arange(-3, 21) / fps).astype(np.float64)
  frames = image_frames({"cam00": times}, fps, (0.0, 0.5))["cam00"]
  # Frames 0..15 fall in [0.0, 0.5] s to half a frame; frame 16 is at 0.533 s.
  np.testing.assert_allclose(times[frames] * fps, np.arange(16), atol=1e-9)


def test_image_frames_are_one_set_of_session_frames_every_lens_has() -> None:
  """An early lens and an off-grid lens keep the same session frames."""
  fps = 30.0
  early = np.arange(-7, 30) / fps
  late = np.arange(0, 30) / fps + 0.004
  frames = image_frames({"a": early, "b": late}, fps, (-0.5, 0.75))
  np.testing.assert_allclose(early[frames["a"]] * fps, np.arange(24), atol=1e-6)
  np.testing.assert_allclose(late[frames["b"]] * fps, np.arange(24) + 0.12, atol=1e-6)
  with pytest.raises(ValueError, match="window"):
    image_frames({"a": early}, fps, (5.0, 6.0))


def _session(left_y: float, right_y: float, offset: float = 0.0) -> tuple[dict, dict]:
  """Two grippers 0.5 m ahead of the anchor along +x; +y is the operator's left."""
  t = np.arange(20) * 0.1
  wobble = 0.01 * np.sin(np.arange(20))
  times = {"cam00": t, "cam01": t + offset}
  translations = {
    "cam00": np.stack([0.5 + wobble, np.full(20, left_y), np.zeros(20)], axis=1),
    "cam01": np.stack([0.5 - wobble, np.full(20, right_y), np.zeros(20)], axis=1),
  }
  return times, translations


def test_hands_name_the_counter_clockwise_gripper_left() -> None:
  assert hands(*_session(+0.1, -0.1)) == ("cam00", "cam01")
  assert hands(*_session(-0.1, +0.1)) == ("cam01", "cam00")
  assert hands(*_session(+0.1, -0.1, offset=0.01)) == ("cam00", "cam01")


def test_hands_refuse_a_thin_margin_or_a_split_verdict() -> None:
  with pytest.raises(ValueError, match="cannot tell"):
    hands(*_session(+0.01, -0.005))
  times, translations = _session(+0.1, -0.1)
  translations["cam01"][::2, 1] = +0.3
  with pytest.raises(ValueError, match="cannot tell"):
    hands(times, translations)
  with pytest.raises(ValueError, match="two cameras"):
    hands({"cam00": times["cam00"]}, {"cam00": translations["cam00"]})


def _approach(at: tuple[float, float]) -> tuple[Float64Array, ...]:
  t = np.arange(0.0, 100.0, 0.05)
  speed = np.full_like(t, 1.0)
  speed[(t >= at[0]) & (t <= at[1])] = 0.05
  distance = np.full_like(t, 2.0)
  distance[(t >= at[0]) & (t <= at[1])] = 0.3
  return t, distance, speed


def test_hold_is_the_slow_span_at_the_closest_slow_approach() -> None:
  t, distance, speed = _approach((2.0, 4.0))
  distance[np.abs(t - 50.0) < 0.05] = 0.1
  assert hold(t, distance, speed, (0.0, 100.0)) == pytest.approx(4.0, abs=0.1)


def test_hold_refuses_a_touch_in_passing_or_a_late_gesture() -> None:
  with pytest.raises(ValueError, match="under the 0.4 s hold"):
    hold(*_approach((2.0, 2.2)), (0.0, 100.0))
  with pytest.raises(ValueError, match="past its first tenth"):
    hold(*_approach((50.0, 52.0)), (0.0, 100.0))


def _tapping(t: Float64Array, at: list[float], magnitude: float) -> Float64Array:
  accel = np.zeros((len(t), 3))
  accel[:, 2] = 9.8 + 0.2 * np.sin(3.0 * t)
  for strike in at:
    accel[np.abs(t - strike) < 0.003, 2] += magnitude
  return accel


def test_tap_dates_the_first_strike_of_the_strongest_shared_burst() -> None:
  t = np.arange(0.0, 100.0, 0.001)
  times = {"cam00": t, "cam01": t}
  accel = {
    "cam00": _tapping(t, [10.0, 96.0, 96.3, 96.6, 98.5], 200.0),
    "cam01": _tapping(t, [96.15, 96.45, 98.6], 200.0),
  }
  accel["cam00"][np.abs(t - 98.5) < 0.003, 2] -= 160.0
  accel["cam01"][np.abs(t - 98.6) < 0.003, 2] -= 160.0
  assert tap(times, accel) == pytest.approx(96.0, abs=0.002)


def test_tap_accepts_a_simultaneous_pair() -> None:
  t = np.arange(0.0, 100.0, 0.001)
  times = {"cam00": t, "cam01": t}
  accel = {slot: _tapping(t, [97.0], 200.0) for slot in times}
  assert tap(times, accel) == pytest.approx(97.0, abs=0.002)


def test_tap_refuses_a_one_sided_or_early_gesture() -> None:
  t = np.arange(0.0, 100.0, 0.001)
  times = {"cam00": t, "cam01": t}
  one_sided = {
    "cam00": _tapping(t, [96.0, 96.3], 200.0),
    "cam01": _tapping(t, [], 200.0),
  }
  with pytest.raises(ValueError, match="reaches every camera"):
    tap(times, one_sided)
  early = {slot: _tapping(t, [50.0], 200.0) for slot in times}
  with pytest.raises(ValueError, match="before its last tenth"):
    tap(times, early)
