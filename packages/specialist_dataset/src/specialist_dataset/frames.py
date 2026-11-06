"""Frame times and H.264 frame decoding for a base recording's lens videos."""

import itertools
import os
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import datafusion
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from specialist_dataset import schema
from specialist_dataset.read import Recording

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

__all__ = [
  "LensVideo",
  "LensFrames",
  "decode_samples",
  "lens_times",
  "lens_video",
  "release_videos",
]

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


# Per (recording file, entity): frame times, and videos until ``release_videos``.
_times: dict[tuple[Path, str], Float64Array] = {}
_videos: dict[tuple[Path, str], LensVideo] = {}
_cache_lock = threading.Lock()


def lens_times(recording: Recording, entity: str) -> Float64Array:
  """One lens entity's frame times (s), ascending, one per encoded sample."""
  key = (recording.path, entity)
  with _cache_lock:
    hit = _times.get(key)
  if hit is None:
    table = _samples(recording, entity).select(_index()).to_arrow_table()
    hit = schema.seconds(table.column(schema.TIMELINE))
    with _cache_lock:
      _times[key] = hit
  return hit


def lens_video(recording: Recording, slot: str, lens: str, stage: Path) -> LensVideo:
  """One lens's encoded samples, loaded once per process and staged under ``stage``."""
  key = (recording.path, schema.video(slot, lens))
  with _cache_lock:
    hit = _videos.get(key)
  if hit is None:
    hit = LensVideo(recording, slot, lens, stage)
    with _cache_lock:
      _videos[key] = hit
      _times[key] = hit.times
  return hit


def release_videos() -> None:
  """Close every loaded video and delete its staged packets, keeping the frame times."""
  with _cache_lock:
    videos = list(_videos.values())
    _videos.clear()
  for video in videos:
    video.close()


class LensVideo:
  """One lens's encoded samples, staged as a packets file under ``stage``."""

  def __init__(self, recording: Recording, slot: str, lens: str, stage: Path) -> None:
    self.entity = schema.video(slot, lens)
    self.path = stage / f"{slot}_{lens}.packets"
    samples = _samples(recording, self.entity).select(
      _index(),
      _column(self.entity, _SAMPLE),
      _column(self.entity, _SYNC),
    )
    try:
      self._stage(samples)
    except BaseException:
      self.path.unlink(missing_ok=True)
      raise

  def _stage(self, samples: datafusion.DataFrame) -> None:
    times: list[Int64Array] = []
    markers: list[npt.NDArray[np.bool_]] = []
    first: list[Int64Array] = []
    last: list[Int64Array] = []
    written = 0
    with self.path.open("wb") as out:
      for batch in samples.execute_stream():
        table = batch.to_pyarrow()
        times.append(schema.times_ns(table.column(schema.TIMELINE)))
        markers.append(_marker_flags(table.column(schema.selector(self.entity, _SYNC))))
        lo, hi, data = _packet_ranges(
          table.column(schema.selector(self.entity, _SAMPLE))
        )
        first.append(lo + written)
        last.append(hi + written)
        out.write(data)
        written += data.nbytes
    times_ns = np.concatenate(times) if times else np.empty(0, dtype=np.int64)
    if np.any(np.diff(times_ns) <= 0):
      raise RuntimeError(f"{self.entity}: samples are not in time order")
    self.times: Float64Array = times_ns / schema.NS_PER_SECOND
    flags = np.concatenate(markers) if markers else np.empty(0, dtype=bool)
    self.sync_samples: Int64Array = np.flatnonzero(flags).astype(np.int64)
    if not self.sync_samples.size:
      raise RuntimeError(
        f"{self.entity}: no sync-sample markers; every decode would start at sample 0"
      )
    self._first: Int64Array = np.concatenate(first)
    self._last: Int64Array = np.concatenate(last)
    self._fd: int | None = os.open(self.path, os.O_RDONLY)

  def packet(self, sample: int) -> bytes:
    if self._fd is None:
      raise RuntimeError(f"{self.entity}: closed")
    lo, hi = int(self._first[sample]), int(self._last[sample])
    return os.pread(self._fd, hi - lo, lo)

  def decode_start(self, sample: int) -> int:
    """The sync sample a decode of ``sample`` must start from."""
    position = int(np.searchsorted(self.sync_samples, sample, side="right")) - 1
    return int(self.sync_samples[position]) if position >= 0 else 0

  def close(self) -> None:
    """Close and delete the staged packets."""
    if self._fd is not None:
      os.close(self._fd)
      self._fd = None
    self.path.unlink(missing_ok=True)


def _packet_ranges(
  column: pa.ListArray,
) -> tuple[Int64Array, Int64Array, npt.NDArray[np.uint8]]:
  """Each sample's first-packet byte range in the batch's bytes, and those bytes."""
  packets = column.values
  sample_offsets = column.offsets.to_numpy().astype(np.int64)
  packet_offsets = packets.offsets.to_numpy().astype(np.int64)
  data: npt.NDArray[np.uint8] = packets.values.to_numpy(zero_copy_only=True)
  first_sample = sample_offsets[:-1]
  return packet_offsets[first_sample], packet_offsets[first_sample + 1], data


def _marker_flags(markers: pa.Array) -> npt.NDArray[np.bool_]:
  """Per sample, whether it carries a true ``is_keyframe`` marker."""
  return np.array([bool(v and v[0]) for v in markers.to_pylist()], dtype=bool)


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
  """A lens's frames at ascending ``samples``, decoded by band, the next one ahead."""

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
    self._decoder = ThreadPoolExecutor(max_workers=1)
    self._ahead: tuple[int, Future[list[np.ndarray]]] | None = None

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
      self._frames = self._band_at(first)
      self._start = first
    return self._frames[index - first]

  def _band_at(self, first: int) -> list[np.ndarray]:
    """The band at ``first``, from a matching look-ahead, with the next band queued."""
    ahead, self._ahead = self._ahead, None
    if ahead is not None and ahead[0] == first:
      frames = ahead[1].result()
    else:
      if ahead is not None:
        ahead[1].result()
      frames = self._decode_band(first)
    following = first + self.band
    if following < self._len:
      self._ahead = (following, self._decoder.submit(self._decode_band, following))
    return frames
