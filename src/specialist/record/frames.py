"""Keyframe decoding straight out of a base recording's video stream."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from rerun.experimental import RrdReader

from specialist.record import schema

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

__all__ = ["LensVideo", "LensFrames", "lens_times"]

# Frame threading delays output, so each band flushes at its end.
_DECODE_THREADS = 16


_SCAN_CACHE: dict[tuple[str, float], dict[str, tuple[pa.Table, list[int]]]] = {}
_SCAN_LOCK = threading.Lock()


def _scan_video(base_rrd: Path) -> dict[str, tuple[pa.Table, list[int]]]:
  """Per lens entity, the ``(video_time, sample)`` table and keyframe times (ns)."""
  key = (str(base_rrd), base_rrd.stat().st_mtime)
  with _SCAN_LOCK:
    cached = _SCAN_CACHE.get(key)
    if cached is not None:
      return cached
    batches: dict[str, list[pa.RecordBatch]] = {}
    keyframe_ns: dict[str, list[int]] = {}
    stream = (
      RrdReader(str(base_rrd))
      .store()
      .stream()
      .filter(
        content="/image/**",
        components=["VideoStream:sample", "VideoStream:is_keyframe"],
      )
    )
    for chunk in stream.to_chunks():
      if chunk.is_static:
        continue
      entity = chunk.entity_path.removeprefix("/")
      batch = chunk.to_record_batch()
      names = batch.schema.names
      if "VideoStream:sample" in names:
        batches.setdefault(entity, []).append(
          batch.select([schema.TIMELINE, "VideoStream:sample"])
        )
      if "VideoStream:is_keyframe" in names:
        flags = batch.column(names.index("VideoStream:is_keyframe")).to_pylist()
        stamps = batch.column(names.index(schema.TIMELINE))
        keyframe_ns.setdefault(entity, []).extend(
          t.value
          for t, flag in zip(stamps, flags, strict=True)
          if flag is not None and len(flag) and bool(flag[0])
        )
    tables = {
      entity: (
        _time_ordered(pa.Table.from_batches(group, schema=group[0].schema)),
        keyframe_ns.get(entity, []),
      )
      for entity, group in batches.items()
    }
    _SCAN_CACHE.clear()
    _SCAN_CACHE[key] = tables
    return tables


def _time_ordered(table: pa.Table) -> pa.Table:
  """The table sorted by its timeline column."""
  times = schema.times_ns(table.column(schema.TIMELINE))
  if bool(np.all(times[:-1] <= times[1:])):
    return table
  return table.sort_by(schema.TIMELINE)


def _sample_table(base_rrd: Path, entity: str) -> tuple[pa.Table, list[int]]:
  """One lens entity's sample table out of the cached single-pass scan."""
  tables = _scan_video(base_rrd)
  if entity not in tables:
    raise RuntimeError(f"{base_rrd} has no video samples at /{entity}")
  return tables[entity]


def lens_times(base_rrd: Path, entity: str) -> Float64Array:
  """One lens entity's frame times (s), ascending, one per encoded sample."""
  table, _ = _sample_table(base_rrd, entity)
  return schema.seconds(table.column(schema.TIMELINE))


class LensVideo:
  """One lens's encoded samples and the timeline they sit on."""

  def __init__(self, base_rrd: Path, tracker: str, lens: str) -> None:
    self.entity = schema.image_entity(tracker, lens)
    table, keyframe_ns = _sample_table(base_rrd, self.entity)
    times_ns = schema.times_ns(table.column(schema.TIMELINE))
    self.times: Float64Array = times_ns / schema.NS_PER_SECOND
    self.keyframes = _sync_samples(times_ns, keyframe_ns)
    if not self.keyframes.size:
      raise RuntimeError(
        f"{self.entity}: no keyframe markers; every decode would start at sample 0"
      )
    # Keep the samples in the scan's chunks; joining them copies the whole video.
    chunks = table.column("VideoStream:sample").chunks
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
    position = int(np.searchsorted(self.keyframes, sample, side="right")) - 1
    return int(self.keyframes[position]) if position >= 0 else 0


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


def _sync_samples(times_ns: Int64Array, keyframe_ns: list[int]) -> Int64Array:
  """Sample rows whose timestamp carries a true ``is_keyframe`` marker."""
  stamps = np.sort(np.asarray(keyframe_ns, dtype=np.int64))
  rows = np.searchsorted(times_ns, stamps)
  inside = rows < len(times_ns)
  rows, stamps = rows[inside], stamps[inside]
  rows = rows[times_ns[rows] == stamps]
  out: Int64Array = rows.astype(np.int64)
  return out


class LensFrames(Sequence[np.ndarray]):
  """A lens's frames at ascending ``samples`` of a base recording, one band at a time."""

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
    import av

    band_rows = self._samples[first : first + self.band]
    wanted = {int(r) for r in band_rows}
    last = int(band_rows[-1])
    row = self._video.decode_start(int(band_rows[0]))
    context = av.CodecContext.create("h264", "r")
    context.thread_count = _DECODE_THREADS
    context.thread_type = "FRAME"
    frames: dict[int, np.ndarray] = {}

    def take(decoded: object, at: int) -> None:
      if at in wanted:
        # The stream is 4:2:0, so the first plane is the full-resolution luma.
        plane = decoded.planes[0]  # type: ignore[attr-defined]
        height = int(decoded.height)  # type: ignore[attr-defined]
        width = int(decoded.width)  # type: ignore[attr-defined]
        rows = np.frombuffer(plane, dtype=np.uint8).reshape(-1, plane.line_size)
        frames[at] = np.ascontiguousarray(rows[:height, :width])

    emitted = row
    for source in range(row, last + 1):
      for decoded in context.decode(av.Packet(self._video.packet(source))):
        take(decoded, emitted)
        emitted += 1
    # Frame threading holds the tail of the band until the context is drained.
    for decoded in context.decode(None):
      take(decoded, emitted)
      emitted += 1
    missing = sorted(wanted - set(frames))
    if missing:
      raise RuntimeError(
        f"{self._video.entity}: no frame decoded for rows {missing[:4]} "
        f"of band [{first}, {first + len(band_rows)})"
      )
    return [frames[int(r)] for r in band_rows]

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
