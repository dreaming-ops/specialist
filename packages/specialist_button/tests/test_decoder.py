"""The decoder reads frames through noise, echoes, collisions and real rooms, and stays silent on noise."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import fft as sfft
from scipy import signal
from scipy.io import wavfile
from specialist_button import (
  FS,
  N_EVENTS,
  N_SOURCES,
  DecoderConfig,
  decode,
  generate_frame,
)
from specialist_button.frame import (
  BANDS,
  FRAME_DURATION,
  PREAMBLE_DURATION,
  SLOT_DURATION,
  chirp_phase,
)

FIXTURES = Path(__file__).parent / "fixtures"
PAIRS = [(s, e) for s in range(N_SOURCES) for e in range(N_EVENTS)]
LEAD = 2.0


def clip(source_id: int, event: int) -> NDArray[np.float64]:
  """`LEAD` of silence, one frame, 0.3 s of silence."""
  silence = np.zeros(round(LEAD * FS)), np.zeros(round(0.3 * FS))
  return np.concatenate([silence[0], generate_frame(source_id, event), silence[1]])


def in_band_power(x: NDArray[np.float64], span: slice) -> float:
  n = sfft.next_fast_len(len(x), real=True)
  f = sfft.rfftfreq(n, 1 / FS)
  keep = np.zeros(len(f), dtype=bool)
  for lo, hi in BANDS:
    keep |= (f >= lo) & (f <= hi)
  return float(np.mean(sfft.irfft(sfft.rfft(x, n) * keep, n)[span] ** 2))


def noisy_clip(
  rng: np.random.Generator, source_id: int, event: int, snr_db: float
) -> tuple[NDArray[np.float64], float]:
  """A frame after a random lead in white noise at an in-band SNR; returns the clip and the onset."""
  lead = round(rng.uniform(0.4, 0.9) * FS)
  frame = generate_frame(source_id, event)
  clean = np.concatenate([np.zeros(lead), frame, np.zeros(round(0.3 * FS))])
  noise = rng.standard_normal(len(clean))
  span = slice(lead, lead + len(frame))
  gain = np.sqrt(
    in_band_power(clean, span) / in_band_power(noise, span) / 10 ** (snr_db / 10)
  )
  return clean + gain * noise, lead / FS


def overlap(
  first: tuple[int, int],
  second: tuple[int, int],
  offset: float,
  second_gain: float = 1.0,
) -> NDArray[np.float64]:
  """Two frames in one clip, the second `offset` after the first."""
  x = np.zeros(round(3.0 * FS))
  for (source_id, event), start, gain in (
    (first, 0.5, 1.0),
    (second, 0.5 + offset, second_gain),
  ):
    a = round(start * FS)
    x[a : a + round(FRAME_DURATION * FS)] += gain * generate_frame(source_id, event)
  return x


def test_round_trip_of_every_frame() -> None:
  for source_id, event in PAIRS:
    found = decode(clip(source_id, event), FS, source_id)
    assert len(found) == 1, (source_id, event)
    d = found[0]
    assert (d.source_id, d.event, d.flags, d.accepted) == (source_id, event, (), True)
    assert d.onset == pytest.approx(LEAD, abs=1e-4)
    assert d.score > 0.95


def test_only_the_own_source_is_accepted() -> None:
  found = decode(clip(7, 2), FS, own_source_id=8)
  assert len(found) == 1 and found[0].source_id == 7 and not found[0].accepted


@pytest.mark.parametrize("fs", [16_000, 44_100, 96_000])
def test_other_sample_rates(fs: int) -> None:
  found = decode(signal.resample_poly(clip(23, 4), fs // 100, FS // 100), fs, 23)
  assert len(found) == 1 and found[0].accepted and found[0].event == 4
  assert found[0].onset == pytest.approx(LEAD, abs=2e-4)


def test_level_does_not_matter() -> None:
  for gain in (1e-4, 30.0):
    found = decode(gain * clip(11, 1), FS, 11)
    assert len(found) == 1 and found[0].accepted


def test_two_frames_in_one_recording() -> None:
  gap = np.zeros(FS // 2)
  x = np.concatenate([gap, generate_frame(3, 0), gap, generate_frame(30, 5), gap])
  found = decode(x, FS, 30)
  assert [(d.source_id, d.event, d.accepted) for d in found] == [
    (3, 0, False),
    (30, 5, True),
  ]
  assert found[1].onset - found[0].onset == pytest.approx(
    FRAME_DURATION + 0.5, abs=1e-4
  )


def test_two_overlapping_frames_of_one_level_are_collisions() -> None:
  found = decode(overlap((3, 0), (30, 5), 0.157), FS, 30)
  assert found and all("collision" in d.flags and not d.accepted for d in found)


def test_a_much_weaker_overlapping_frame_leaves_the_strong_one_readable() -> None:
  found = decode(overlap((3, 0), (30, 5), 0.157, second_gain=10 ** (-12 / 20)), FS, 3)
  assert any(d.accepted and d.event == 0 for d in found)
  assert all(d.flags or (d.source_id, d.event) in {(3, 0), (30, 5)} for d in found)


def test_two_equal_frames_on_one_slot_grid_give_no_wrong_frame() -> None:
  found = decode(overlap((3, 0), (30, 5), 0.2), FS, 30)
  assert {(d.source_id, d.event) for d in found if not d.flags} <= {(3, 0), (30, 5)}


def test_an_echo_of_a_frame_is_no_second_frame() -> None:
  found = decode(overlap((3, 0), (3, 0), 0.02, second_gain=0.5), FS, 3)
  assert len(found) == 1 and found[0].accepted


def test_a_strong_frame_right_after_a_weak_one_is_reported() -> None:
  found = decode(overlap((12, 1), (3, 0), 0.39, second_gain=10 ** (15 / 20)), FS, 3)
  assert any(d.accepted and d.event == 0 for d in found)


def test_a_frame_with_a_wrong_check_digit_is_ambiguous() -> None:
  x = generate_frame(7, 2)
  last = slice(len(x) - round(0.2 * FS), len(x))
  x[last] = generate_frame(7, 3)[last]
  found = decode(np.concatenate([x, np.zeros(FS // 10)]), FS, 7)
  assert len(found) == 1 and "ambiguous" in found[0].flags and not found[0].accepted


def test_one_lost_slot_is_corrected() -> None:
  rng = np.random.default_rng(0)
  x = np.concatenate([generate_frame(7, 2), np.zeros(FS // 10)])
  start = round((PREAMBLE_DURATION + 2 * SLOT_DURATION) * FS)  # the L slot of d2
  x[start : start + round(SLOT_DURATION * FS)] = 0.5 * rng.standard_normal(
    round(SLOT_DURATION * FS)
  )
  found = decode(x + 0.01 * rng.standard_normal(len(x)), FS, 7)
  assert len(found) == 1 and found[0].accepted and found[0].event == 2


def test_a_frame_cut_by_the_end_of_the_recording_is_not_reported() -> None:
  assert decode(clip(7, 2)[: round((LEAD + FRAME_DURATION - 0.05) * FS)], FS, 7) == []


def test_a_frame_that_ends_with_the_recording_is_reported() -> None:
  assert decode(clip(7, 2)[: round((LEAD + FRAME_DURATION) * FS)], FS, 7)[0].accepted


def test_a_frame_of_the_first_design_is_ignored() -> None:
  def chirp(duration: float, reverse: bool = False) -> NDArray[np.float64]:
    return np.cos(chirp_phase(np.arange(round(duration * FS)) / FS, duration, reverse))

  first_design = np.concatenate(
    [chirp(0.2), chirp(0.2, reverse=True)] + [chirp(0.1)] * 4
  )
  assert decode(np.concatenate([np.zeros(FS), first_design, np.zeros(FS)]), FS) == []


def test_silence_and_short_input() -> None:
  assert decode(np.zeros(FS), FS) == []
  assert decode(np.zeros(100), FS) == []


def test_threshold_is_a_parameter() -> None:
  audio, _ = noisy_clip(np.random.default_rng(1), 5, 5, snr_db=-3.0)
  scores = [d.score for d in decode(audio, FS, 5, DecoderConfig(threshold=0.15))]
  assert scores and max(scores) < 0.9
  assert decode(audio, FS, 5, DecoderConfig(threshold=max(scores) + 0.01)) == []


def test_every_frame_is_accepted_in_white_noise_at_5_db() -> None:
  rng = np.random.default_rng(2)
  for source_id, event in PAIRS:
    audio, onset = noisy_clip(rng, source_id, event, snr_db=5.0)
    found = decode(audio, FS, source_id)
    assert len(found) == 1 and found[0].accepted and found[0].event == event
    assert found[0].onset == pytest.approx(onset, abs=5e-4)


def test_white_noise_alone_gives_no_detection() -> None:
  rng = np.random.default_rng(3)
  for _ in range(10):
    assert decode(rng.standard_normal(60 * 16_000), 16_000) == []


def test_three_frames_recorded_in_an_echoing_room_are_read() -> None:
  fs, x = wavfile.read(FIXTURES / "echoing_room.wav")
  found = decode(x / 32767, fs, 7)
  assert [round(d.onset, 2) for d in found] == [0.65, 4.43, 7.02]
  assert all(d.accepted and d.event == 2 for d in found)
