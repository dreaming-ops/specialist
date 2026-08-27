"""Frame times and H.264 frame decoding for a base recording's lens videos."""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import Any

import datafusion
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from specialist_dataset import schema
from specialist_dataset.read import Recording

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

__all__ = ["LensVideo", "LensFrames", "decode_samples", "lens_times"]

_DECODE_THREADS = 16

_SAMPLE = "VideoStream:sample"
_SYNC = "VideoStream:is_keyframe"


def _column(entity: str, component: str) -> datafusion.Expr:
  return datafusion.col(f'"{schema.selector(entity, component)}"')


def _index() -> datafusion.Expr:
  return datafusion.col(f'"{schema.TIMELINE}"')


def _samples(recording: Recording, entity: str) -> datafusion.DataFrame:
  """One lens entity's samples, with their sync-sample markers."""
  reader = recording.reader(f"/{entity}")
  if schema.selector(entity, _SAMPLE) not in reader.schema().names:
    raise RuntimeError(f"{recording.path} has no video samples at /{entity}")
  samples: datafusion.DataFrame = reader.filter(_column(entity, _SAMPLE).is_not_null())
  return samples


def lens_times(recording: Recording, entity: str) -> Float64Array:
  """One lens entity's frame times (s), ascending, one per encoded sample."""
  table = _samples(recording, entity).select(_index()).to_arrow_table()
  return schema.seconds(table.column(schema.TIMELINE))


class LensVideo:
  """One lens's encoded samples and the timeline they sit on."""

  def __init__(self, recording: Recording, slot: str, lens: str) -> None:
    self.entity = schema.video(slot, lens)
    samples = _samples(recording, self.entity)
    table = samples.select(
      _index(),
      _column(self.entity, _SAMPLE),
      _column(self.entity, _SYNC),
    ).to_arrow_table()
    times_ns = schema.times_ns(table.column(schema.TIMELINE))
    self.times: Float64Array = times_ns / schema.NS_PER_SECOND
    self.sync_samples = _sync_samples(table.column(schema.selector(self.entity, _SYNC)))
    if not self.sync_samples.size:
      raise RuntimeError(
        f"{self.entity}: no sync-sample markers; every decode would start at sample 0"
      )
    # Joining the chunks would copy the lens's whole video onto the heap.
    chunks = table.column(schema.selector(self.entity, _SAMPLE)).chunks
    self._chunk_first: Int64Array = np.cumsum(
      [0, *(len(chunk) for chunk in chunks)], dtype=np.int64
    )
    self._chunks = [_packet_arrays(chunk) for chunk in chunks]

  def packet(self, row: int) -> bytes:
    # An empty chunk repeats its neighbor's first row, which ``right`` skips.
    chunk = int(np.searchsorted(self._chunk_first, row, side="right")) - 1
    sample_offsets, packet_offsets, data = self._chunks[chunk]
    sample = int(sample_offsets[row - self._chunk_first[chunk]])
    lo = int(packet_offsets[sample])
    hi = int(packet_offsets[sample + 1])
    encoded: bytes = data[lo:hi].tobytes()
    return encoded

  def decode_start(self, sample: int) -> int:
    """The sync sample a decode of ``sample`` must start from."""
    position = int(np.searchsorted(self.sync_samples, sample, side="right")) - 1
    return int(self.sync_samples[position]) if position >= 0 else 0


def _packet_arrays(
  chunk: pa.ListArray,
) -> tuple[Int64Array, Int64Array, npt.NDArray[np.uint8]]:
  """One chunk's ``(sample offsets, packet offsets, bytes)``."""
  packets = chunk.values
  return (
    chunk.offsets.to_numpy(),
    packets.offsets.to_numpy(),
    packets.values.to_numpy(zero_copy_only=True),
  )


def _sync_samples(markers: pa.ChunkedArray) -> Int64Array:
  """Sample rows carrying a true ``is_keyframe`` marker."""
  flags = [
    bool(value[0]) if value else False for value in markers.combine_chunks().to_pylist()
  ]
  out: Int64Array = np.flatnonzero(flags).astype(np.int64)
  return out


def decode_samples(video: LensVideo, samples: Int64Array) -> Iterator[tuple[int, Any]]:
  """``(sample, frame)`` for ascending ``samples``, one pass from a sync sample."""
  import av

  wanted = {int(s) for s in samples}
  first, last = video.decode_start(int(samples[0])), int(samples[-1])
  context = av.CodecContext.create("h264", "r")
  context.thread_count = _DECODE_THREADS
  context.thread_type = "FRAME"
  packets = itertools.chain(
    (av.Packet(video.packet(source)) for source in range(first, last + 1)), [None]
  )
  emitted, found = first, 0
  packet = None
  try:
    for packet in packets:
      for decoded in context.decode(packet):
        if emitted in wanted:
          found += 1
          yield emitted, decoded
        emitted += 1
  finally:
    # Freeing an undrained threaded decoder blocks forever.
    if packet is not None:
      for _ in context.decode(None):
        pass
  if found != len(wanted):
    raise RuntimeError(
      f"{video.entity}: {len(wanted) - found} of {len(wanted)} samples never decoded, "
      f"from sample {emitted}"
    )


def _luma(decoded: Any) -> np.ndarray:
  """The full-resolution luma plane of a 4:2:0 frame, without a swscale conversion."""
  plane = decoded.planes[0]
  rows = np.frombuffer(plane, dtype=np.uint8).reshape(-1, plane.line_size)
  return np.ascontiguousarray(rows[: int(decoded.height), : int(decoded.width)])


class LensFrames(Sequence[np.ndarray]):
  """A lens's frames at ascending ``samples``, decoded and held one band at a time."""

  def __init__(self, video: LensVideo, samples: Int64Array, band: int) -> None:
    self._video = video
    self._samples = np.asarray(samples, dtype=np.int64)
    if self._samples.size and np.any(np.diff(self._samples) < 0):
      raise ValueError("samples must be ascending")
    self.band = band
    self._start = -1
    self._frames: list[np.ndarray] = []
    self._carry: tuple[int, np.ndarray] | None = None
    self._len = int(self._samples.size)

  def __len__(self) -> int:
    return self._len

  def _decode_band(self, first: int) -> list[np.ndarray]:
    band = self._samples[first : first + self.band]
    frames = {
      sample: _luma(decoded) for sample, decoded in decode_samples(self._video, band)
    }
    return [frames[int(s)] for s in band]

  def __getitem__(self, index: int) -> np.ndarray:  # type: ignore[override]
    if index < 0:
      index += self._len
    if not 0 <= index < self._len:
      raise IndexError(index)
    if self._carry is not None and index == self._carry[0]:
      return self._carry[1]
    first = (index // self.band) * self.band
    if first != self._start:
      if self._frames:
        self._carry = (self._start + len(self._frames) - 1, self._frames[-1])
      self._frames = self._decode_band(first)
      self._start = first
    return self._frames[index - first]
