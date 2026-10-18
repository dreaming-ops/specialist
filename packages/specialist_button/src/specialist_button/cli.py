"""Program a talking button with a marker frame, hear buttons live, or decode a recording."""

import collections
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import tyro
from numpy.typing import ArrayLike, NDArray
from scipy import signal
from scipy.io import wavfile

from specialist_button.decoder import DEFAULT_CONFIG, Detection, decode, is_repeat
from specialist_button.frame import FRAME_DURATION, FS, generate_frame

if TYPE_CHECKING:
  from specialist_button.device import Duplex

PEAK_DBFS = -28.0  # peak of the played frame
BEEP_BAND = (2600.0, 3200.0)  # the record beep is a 2906 Hz tone of 37 ms
WIDE_BAND = (1000.0, 8000.0)
LEVEL_BAND = (800.0, 3700.0)
BEEP_TO_STORAGE = 0.57  # start of the beep to the first stored sample
STORED_LEAD = 0.45  # stored silence before the frame, while the button's gain settles
DELAY = BEEP_TO_STORAGE + STORED_LEAD  # start of the beep to the start of the frame
LOOP_LATENCY = 0.156  # `Duplex.play` to the sound's arrival in the recording
SETTLE = 0.5  # start of the stream that holds its start-up noise
BEEP_TIMEOUT = 60.0
CHUNK = 60.0  # stretch of a file decoded at once
OVERLAP = 2.0  # longer than a frame and its echoes

Floats = NDArray[np.float64]
Out = Annotated[Path, tyro.conf.arg(help="folder of the saved recording")]


class BeepDetector:
  """Finds the button's record beep in a 48 kHz stream fed block by block."""

  HOP = 240  # samples per 5 ms frame
  RUN = 3  # beep frames in a row that make a beep
  RATIO = 100.0  # beep-band power over the room's
  TONALITY = 0.5  # beep-band power over the wide band's

  def __init__(self) -> None:
    self._filters = [
      signal.butter(4, band, "bandpass", fs=FS, output="sos")
      for band in (BEEP_BAND, WIDE_BAND)
    ]
    self._states = [np.zeros((sos.shape[0], 2)) for sos in self._filters]
    self._rest = np.zeros((2, 0))  # filtered samples short of a whole frame
    self._room: collections.deque[float] = collections.deque(maxlen=200)
    self._frames = 0
    self._run = 0
    self.onset: int | None = None  # first sample of the beep

  def feed(self, block: ArrayLike) -> int | None:
    """Process the next samples; returns the beep onset once found."""
    filtered = []
    for i, sos in enumerate(self._filters):
      y, self._states[i] = signal.sosfilt(
        sos, np.asarray(block, dtype=float), zi=self._states[i]
      )
      filtered.append(y)
    samples = np.concatenate([self._rest, np.array(filtered)], axis=1)
    whole = samples.shape[1] // self.HOP
    self._rest = samples[:, whole * self.HOP :]
    powers = (samples[:, : whole * self.HOP] ** 2).reshape(2, whole, self.HOP)
    for narrow, wide in powers.mean(axis=2).T:
      if self.onset is None:
        self._frame(float(narrow), float(wide))
      self._frames += 1
    return self.onset

  def _frame(self, narrow: float, wide: float) -> None:
    armed = len(self._room) >= 40
    room = float(np.median(self._room)) if armed else np.inf
    if (
      armed and narrow > self.RATIO * max(room, 1e-14) and narrow > self.TONALITY * wide
    ):
      self._run += 1
      if self._run == self.RUN:
        self.onset = (self._frames - self.RUN + 1) * self.HOP
    else:
      self._run = 0
      if self._frames * self.HOP >= SETTLE * FS:
        self._room.append(narrow)


def band_level_db(x: ArrayLike, fs: int = FS) -> float:
  """Level of `x` inside `LEVEL_BAND`, dB re full scale."""
  x = np.asarray(x, dtype=float)
  power = np.abs(np.fft.rfft(x)) ** 2
  f = np.fft.rfftfreq(len(x), 1 / fs)
  inside = (f >= LEVEL_BAND[0]) & (f <= LEVEL_BAND[1])
  return float(10 * np.log10(2 * power[inside].sum() / len(x) ** 2 + 1e-12))


class FrameLog:
  """The frames of one pass, each counted once however many decodes meet it."""

  def __init__(self) -> None:
    self.onsets: list[float] = []  # from the start of the pass
    self.frames: list[Detection] = []
    self.levels: list[float] = []

  def add(self, onset: float, frame: Detection, level_db: float) -> str | None:
    """Log a decoded frame; returns its printed line, or None for a copy of a logged one."""
    near = [
      f
      for t, f in zip(self.onsets, self.frames, strict=True)
      if abs(onset - t) < DEFAULT_CONFIG.min_separation
    ]
    if is_repeat(frame, near):
      return None
    self.onsets.append(onset)
    self.frames.append(frame)
    self.levels.append(level_db)
    # A flagged frame names no source, because its digits are a guess.
    if frame.flags:
      read = f"unreadable: {', '.join(frame.flags)}"
    else:
      read = f"source {frame.source_id:2d}  event {frame.event}"
    return (
      f"  {onset:6.1f} s  level {level_db:5.1f} dB  score {frame.score:.2f}  {read}"
    )

  def summary(self) -> str:
    """Frame count, and per source and event the median level, score and clock error."""
    unreadable = sum(bool(d.flags) for d in self.frames)
    lines = [f"{len(self.frames)} frames, {unreadable} unreadable"]
    readable = [
      (d, lv) for d, lv in zip(self.frames, self.levels, strict=True) if not d.flags
    ]
    for key in sorted({(d.source_id, d.event) for d, _ in readable}):
      own = [
        (lv, d.score, d.clock_error)
        for d, lv in readable
        if (d.source_id, d.event) == key
      ]
      level, score, clock = np.median(own, axis=0)
      lines.append(
        f"  source {key[0]}, event {key[1]}: {len(own)} frames, level {level:.1f} dB, "
        f"score {score:.2f}, clock {100 * clock:+.2f}%"
      )
    return "\n".join(lines)


def wait_for_beep(stream: Duplex) -> int:
  """Feed the microphone to a beep detector until the record beep; returns its onset sample."""
  detector, fed = BeepDetector(), 0
  while detector.onset is None:
    if stream.elapsed() > BEEP_TIMEOUT:
      sys.exit(f"No beep within {BEEP_TIMEOUT:.0f} s.")
    blocks = stream.blocks[fed:]
    fed += len(blocks)
    for block in blocks:
      detector.feed(block)
    time.sleep(0.003)
  return detector.onset


def frame_delay(recording: Floats, onset: int, source_id: int) -> float | None:
  """Time from the beep onset to the frame in a programming recording, or None without a frame."""
  start = max(0, onset + round((DELAY - 0.3) * FS))
  heard = decode(
    recording[start : start + round((FRAME_DURATION + 0.6) * FS)], FS, source_id
  )
  return (start - onset) / FS + heard[0].onset if heard else None


def program_pass(stream: Duplex, source_id: int, event: int) -> Floats:
  """Play the frame `DELAY` after the button's record beep, so the button stores it; returns the recording."""
  frame = 10 ** (PEAK_DBFS / 20) * generate_frame(source_id, event)
  try:
    stream.wait(SETTLE + 0.3)  # room noise for the beep detector
    print(f"Frame of source {source_id}, event {event}")
    print(
      "Long-press the record key until the button beeps. Keep recording until STOP."
    )
    onset = wait_for_beep(stream)
    print("  BEEP")
    while stream.elapsed() < onset / FS + DELAY - LOOP_LATENCY:
      time.sleep(0.002)
    stream.play(frame)
    print("  STOP")
    stream.wait(0.5)
  finally:
    recording = stream.stop().astype(float)
  delay = frame_delay(recording, onset, source_id)
  if delay is None:
    print("WARNING: the microphone did not hear the frame. Check the speaker.")
  else:
    print(
      f"The frame started {1e3 * delay:.0f} ms after the beep (target {1e3 * DELAY:.0f} ms)."
    )
  return recording


def playback_pass(stream: Duplex) -> Floats:
  """Print every frame the microphone hears, under a live level bar, until Ctrl-C; returns the recording."""
  print("Press the buttons. Ctrl-C ends the pass.")
  log = FrameLog()
  done = 0  # first sample to decode
  try:
    while True:
      time.sleep(0.1)
      audio = stream.audio()
      # A frame that ends now started at most a frame and a slot window ago.
      done = max(done, len(audio) - round((FRAME_DURATION + 0.5) * FS))
      for d in decode(audio[done:], FS):
        start = done + round(d.onset * FS)
        level = band_level_db(audio[start : start + round(FRAME_DURATION * FS)])
        line = log.add(start / FS, d, level)
        if line is not None:
          print(f"\r{line}{'':<20}")
      # A frame overlapping the last one ends after it, so decoding resumes there.
      if log.onsets:
        done = max(done, round(log.onsets[-1] * FS) - FS // 100)
      bar = "#" * int(np.clip((band_level_db(audio[-FS // 10 :]) + 70) / 2, 0, 30))
      print(f"\r  {len(audio) / FS:6.1f} s  {bar:<30}", end="", flush=True)
  except KeyboardInterrupt:
    pass
  finally:
    recording = stream.stop().astype(float)
  print(f"\r{'':<50}\r{log.summary()}")
  return recording


def decode_file(
  path: tyro.conf.Positional[Annotated[Path, tyro.conf.arg(metavar="FILE")]],
) -> None:
  """Print every frame in a WAV file of 16 kHz or more; of several channels, the first counts."""
  fs, samples = wavfile.read(path, mmap=True)
  scale = (
    np.iinfo(samples.dtype).max if np.issubdtype(samples.dtype, np.integer) else 1.0
  )
  log = FrameLog()
  step, length = round((CHUNK - OVERLAP) * fs), round(CHUNK * fs)
  for first in range(0, len(samples), step):
    audio = samples[first : first + length]
    audio = (audio[:, 0] if audio.ndim > 1 else audio) / scale
    for d in decode(audio, fs):
      start = round(d.onset * fs)
      level = band_level_db(audio[start : start + round(FRAME_DURATION * fs)], fs)
      line = log.add(first / fs + d.onset, d, level)
      if line is not None:
        print(line)
  print(log.summary())


def _save(recording: Floats, out: Path, suffix: str = "") -> None:
  out.mkdir(parents=True, exist_ok=True)
  path = out / f"{time.strftime('%Y%m%d-%H%M%S')}{suffix}.wav"
  wavfile.write(path, FS, recording.astype(np.float32))
  print(path.resolve())


def _open() -> Duplex:
  from specialist_button.device import Duplex

  try:
    return Duplex().start()
  except RuntimeError as error:
    sys.exit(str(error))


def program(
  source_id: tyro.conf.Positional[Annotated[int, tyro.conf.arg(metavar="SOURCE_ID")]],
  event: tyro.conf.Positional[Annotated[int, tyro.conf.arg(metavar="EVENT")]],
  out: Out = Path("logs/button"),
) -> None:
  """Store the frame of a source id (0..35) and an event (0..5) in a button."""
  try:
    generate_frame(source_id, event)
  except ValueError as error:
    sys.exit(str(error))
  _save(program_pass(_open(), source_id, event), out, "_program")


def playback(out: Out = Path("logs/button")) -> None:
  """Print every frame the microphone hears, until Ctrl-C."""
  _save(playback_pass(_open()), out)


def main() -> None:
  tyro.extras.subcommand_cli_from_dict(
    {"program": program, "playback": playback, "decode": decode_file},
    description=__doc__,
  )


if __name__ == "__main__":
  main()
