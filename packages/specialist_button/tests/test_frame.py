"""The frame waveform matches its definition."""

import numpy as np
import pytest
from scipy import signal
from specialist_button import frame as mf


def test_frame_length_and_peak() -> None:
  x = mf.generate_frame(17, 3)
  assert len(x) == round(1.3 * mf.FS) == round(mf.FRAME_DURATION * mf.FS)
  assert np.abs(x).max() == pytest.approx(1.0, abs=1e-3)


def test_amplitude_is_constant_between_the_end_ramps() -> None:
  x = mf.generate_frame(17, 3)
  ramp = round(mf.RAMP_DURATION * mf.FS)
  windows = np.abs(x[ramp:-ramp])
  windows = windows[: len(windows) // 96 * 96].reshape(-1, 96)  # 2 ms each
  assert windows.max(axis=1).min() > 0.97
  assert abs(x[0]) < 1e-9 and abs(x[-1]) < 1e-3


@pytest.mark.parametrize("reverse", [False, True])
def test_chirp_stays_inside_the_bands_at_one_sweep_rate(reverse: bool) -> None:
  duration = mf.CHIRP_DURATION
  t = np.arange(round(duration * mf.FS) + 1) / mf.FS
  f = np.diff(mf.chirp_phase(t, duration, reverse)) * mf.FS / (2 * np.pi)
  inside = np.zeros(len(f), dtype=bool)
  for lo, hi in mf.BANDS:
    inside |= (f > lo - 1) & (f < hi + 1)
  assert inside.mean() > 0.999  # samples straddling the jump between bands
  rate = np.abs(np.diff(f)) * mf.FS
  smooth = rate < 10 * mf.AXIS_WIDTH / duration
  assert np.allclose(rate[smooth], mf.AXIS_WIDTH / duration, rtol=1e-3)
  assert (~smooth).sum() <= 2


def test_down_chirp_is_the_time_reverse_of_the_up_chirp() -> None:
  t = np.arange(round(mf.CHIRP_DURATION * mf.FS) + 1) / mf.FS
  up = np.diff(mf.chirp_phase(t, mf.CHIRP_DURATION))
  down = np.diff(mf.chirp_phase(t, mf.CHIRP_DURATION, reverse=True))
  assert np.allclose(down, up[::-1], atol=1e-9)


def test_frame_starts_with_the_down_chirp_and_sends_each_digit_in_both_bands() -> None:
  x = mf.generate_frame(7, 2)
  f = np.diff(np.unwrap(np.angle(signal.hilbert(x)))) * mf.FS / (2 * np.pi)
  assert f[round(0.01 * mf.FS)] > 3600 and f[round(0.24 * mf.FS)] < 900
  preamble, slot = round(mf.PREAMBLE_DURATION * mf.FS), round(mf.SLOT_DURATION * mf.FS)
  for i, digit in enumerate(mf.frame_digits(7, 2)):
    for band in range(len(mf.BANDS)):
      a = preamble + (2 * i + band) * slot
      middle = f[a + slot // 10 : a + slot - slot // 10]
      lo, hi = mf.sub_band(band, digit)
      assert lo < middle.min() and middle.max() < hi


def test_sub_bands_split_each_band_in_six() -> None:
  for band, (lo, hi) in enumerate(mf.BANDS):
    edges = [mf.sub_band(band, k) for k in range(mf.BASE)]
    assert edges[0][0] == lo and edges[-1][1] == pytest.approx(hi)
    assert all(
      a[1] == pytest.approx(b[0]) for a, b in zip(edges[:-1], edges[1:], strict=True)
    )


def test_digits_and_check() -> None:
  assert mf.frame_digits(0, 0) == (0, 0, 0, 0)
  assert mf.frame_digits(35, 5) == (5, 5, 5, 3)
  assert mf.frame_digits(7, 2) == (1, 1, 2, 4)
  for source_id, event in [(-1, 0), (36, 0), (0, -1), (0, 6)]:
    with pytest.raises(ValueError):
      mf.frame_digits(source_id, event)
