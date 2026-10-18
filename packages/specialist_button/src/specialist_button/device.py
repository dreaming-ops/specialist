"""A 48 kHz duplex stream on the Mac's built-in microphone and speaker."""

import atexit
import re
import subprocess
import threading
import time
from typing import Any

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from specialist_button.frame import FS

_BUILTIN = r"(MacBook|iMac|Built-in|Studio Display)"
_OUT_CHANNEL = 0  # one speaker, so the two sides form no comb filter


def set_volume(output: int | None = None, inp: int | None = None) -> None:
  """Set the system output and input volume, 0..100."""
  if output is not None:
    _osascript(f"set volume output volume {output} without output muted")
  if inp is not None:
    _osascript(f"set volume input volume {inp}")


def _osascript(command: str) -> None:
  subprocess.run(["osascript", "-e", command], capture_output=True, check=False)


class Duplex:
  """Records the microphone from start to stop and plays a waveform on demand."""

  def __init__(self) -> None:
    i_in, i_out = sd.default.device
    devices = sd.query_devices()
    name_in, name_out = devices[i_in]["name"], devices[i_out]["name"]
    if not (
      re.search(f"{_BUILTIN}.*Microphone", name_in)
      and re.search(f"{_BUILTIN}.*Speakers", name_out)
    ):
      raise RuntimeError(
        f"the default devices are {name_in!r} and {name_out!r}; select the built-in "
        "microphone and speakers in System Settings > Sound"
      )
    self.blocks: list[NDArray[np.float32]] = []
    self.frames = 0
    self._out: NDArray[np.float32] | None = None
    self._pos = 0
    self._done = threading.Event()
    self._stream = sd.Stream(
      samplerate=FS,
      device=(i_in, i_out),
      channels=(1, 2),
      dtype="float32",
      latency="high",
      callback=self._callback,
    )
    if self._stream.samplerate != FS:
      raise RuntimeError(f"the stream runs at {self._stream.samplerate} Hz, not {FS}")

  def _callback(
    self,
    indata: NDArray[np.float32],
    outdata: NDArray[np.float32],
    frames: int,
    _time: Any,
    _status: Any,
  ) -> None:
    self.blocks.append(indata[:, 0].copy())
    outdata.fill(0)
    out = self._out
    if out is not None:
      n = min(frames, len(out) - self._pos)
      outdata[:n, _OUT_CHANNEL] = out[self._pos : self._pos + n]
      self._pos += n
      if self._pos >= len(out):
        self._out = None
        self._done.set()
    self.frames += frames

  def start(self) -> Duplex:
    """Set the speaker and microphone to full volume, the speaker to 0 at exit, and start."""
    set_volume(output=100, inp=100)
    atexit.register(set_volume, output=0)
    self._stream.start()
    return self

  def elapsed(self) -> float:
    return self.frames / FS

  def wait(self, seconds: float) -> None:
    """Block until `seconds` more of audio are recorded."""
    target = self.frames + int(seconds * FS)
    while self.frames < target:
      time.sleep(0.01)

  def play(self, x: NDArray[np.float64]) -> None:
    """Play `x` on the speaker and block until it ends."""
    self._done.clear()
    self._pos = 0
    self._out = np.asarray(x, dtype=np.float32)
    self._done.wait()

  def audio(self) -> NDArray[np.float32]:
    if not self.blocks:
      return np.zeros(0, dtype=np.float32)
    return np.concatenate(self.blocks)

  def stop(self) -> NDArray[np.float32]:
    self._stream.stop()
    self._stream.close()
    return self.audio()
