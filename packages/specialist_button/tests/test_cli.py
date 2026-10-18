"""The beep detector, the live playback pass, and file decoding, on synthetic and recorded audio."""

import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import signal
from scipy.io import wavfile
from specialist_button import FRAME_DURATION, FS, cli, generate_frame

FIXTURES = Path(__file__).parent / "fixtures"


def beep_onset(x: NDArray[np.float64], block: int = 512) -> float | None:
  """Beep onset with the recording fed in blocks, as a stream delivers it."""
  detector = cli.BeepDetector()
  for i in range(0, len(x), block):
    detector.feed(x[i : i + block])
  return None if detector.onset is None else detector.onset / FS


class Replay:
  """Stream stand-in that hands out a recording 0.1 s at a time and then ends like Ctrl-C."""

  def __init__(self, recording: NDArray[np.float64]) -> None:
    self.recording, self.n = recording, 0

  def audio(self) -> NDArray[np.float64]:
    self.n += FS // 10
    if self.n > len(self.recording):
      raise KeyboardInterrupt
    return self.recording[: self.n]

  def stop(self) -> NDArray[np.float64]:
    return self.recording


def play_back(
  recording: NDArray[np.float64],
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> list[str]:
  """The printed lines of a playback pass over `recording`."""
  monkeypatch.setattr(cli.time, "sleep", lambda _: None)
  cli.playback_pass(Replay(recording))  # type: ignore[arg-type]
  return [
    part for line in capsys.readouterr().out.splitlines() for part in line.split("\r")
  ]


def test_the_beep_and_the_frame_of_a_recorded_programming_pass() -> None:
  fs, x = wavfile.read(FIXTURES / "programming_pass.wav")
  recording = x / 32767
  onset = beep_onset(recording)
  assert fs == FS and onset == pytest.approx(1.0, abs=0.01)
  delay = cli.frame_delay(recording, round(onset * FS), 7)
  assert delay == pytest.approx(cli.DELAY, abs=0.01)


def test_a_click_alone_is_no_beep() -> None:
  rng = np.random.default_rng(0)
  x = 1e-3 * rng.standard_normal(3 * FS)
  burst = rng.standard_normal(round(0.004 * FS)) * np.hanning(round(0.004 * FS))
  x[2 * FS : 2 * FS + len(burst)] += 0.5 * burst
  assert beep_onset(x) is None


def test_playback_pass_prints_each_frame_once_with_its_level(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  rng = np.random.default_rng(0)
  recording = 1e-4 * rng.standard_normal(7 * FS)
  for (source_id, event), start, gain in (
    ((3, 0), 1.0, 0.1),
    ((30, 5), 1.157, 0.1),
    ((7, 2), 3.5, 0.02),
    ((9, 1), 5.0, 0.1),
  ):
    a = round(start * FS)
    recording[a : a + round(FRAME_DURATION * FS)] += gain * generate_frame(
      source_id, event
    )
  check = slice(
    round((5.0 + FRAME_DURATION - 0.2) * FS), round((5.0 + FRAME_DURATION) * FS)
  )
  n = check.stop - check.start
  recording[check] = 1e-4 * rng.standard_normal(n) + 0.1 * generate_frame(9, 2)[-n:]
  lines = play_back(recording, monkeypatch, capsys)
  frames = [line.split() for line in lines if " s  level" in line]
  assert [(f[8], f[10]) for f in frames[:3]] == [("3", "0"), ("30", "5"), ("7", "2")]
  assert [round(float(f[3])) for f in frames[1:3]] == [-20, -37]
  assert frames[3][7] == "unreadable:" and "source" not in frames[3]
  assert any(line.startswith("4 frames, 1 unreadable") for line in lines)
  assert sum("frames, level" in line for line in lines) == 3
  assert not any("source 9" in line for line in lines)


def test_playback_pass_prints_a_frame_with_strong_echoes_once(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  recording = 1e-4 * np.random.default_rng(1).standard_normal(8 * FS)
  frame = generate_frame(7, 2)
  for start in (1.0, 4.0):
    for delay, gain in ((0.0, 0.1), (0.07, 0.08), (0.15, 0.07)):
      a = round((start + delay) * FS)
      recording[a : a + len(frame)] += gain * frame
  lines = play_back(recording, monkeypatch, capsys)
  assert any(line.startswith("2 frames, 0 unreadable") for line in lines)


def test_playback_pass_counts_each_press_in_an_echoing_room_once(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  fs, x = wavfile.read(FIXTURES / "echoing_room.wav")
  lines = play_back(signal.resample_poly(x / 32767, FS // fs, 1), monkeypatch, capsys)
  assert any(line.startswith("3 frames, 0 unreadable") for line in lines)


def test_decode_reads_a_wav_of_another_rate_and_two_channels(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  x = np.concatenate(
    [np.zeros(FS // 2), 0.1 * generate_frame(12, 4), np.zeros(FS // 2)]
  )
  x = signal.resample_poly(x, 441, 480)
  stereo = np.stack([x, np.zeros(len(x))], axis=1)
  wavfile.write(
    tmp_path / "other.wav", 44_100, np.round(stereo * 32767).astype(np.int16)
  )
  monkeypatch.setattr(sys, "argv", ["button", "decode", str(tmp_path / "other.wav")])
  cli.main()
  out = capsys.readouterr().out
  assert "source 12  event 4" in out and "1 frames, 0 unreadable" in out


def test_program_rejects_a_source_id_out_of_range() -> None:
  with pytest.raises(SystemExit, match="outside 0..35"):
    cli.program(36, 0)
