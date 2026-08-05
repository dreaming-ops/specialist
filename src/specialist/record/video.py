"""Insta360 H.264 ingest, losslessly remuxed to fix an SPS rerun's parser rejects."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from rerun.experimental import Chunk, LazyChunkStream, Mp4Reader, Mp4TranscodeOptions

from specialist.loader.audio_sync import resolve_ffmpeg

_NS = 1_000_000_000


def normalize_h264(src: Path, dst: Path) -> Path:
  """Remux ``src``'s video stream into a clean MP4 at ``dst`` (stream copy)."""
  cmd = [
    resolve_ffmpeg(),
    "-y",
    "-loglevel",
    "error",
    "-i",
    str(src),
    "-map",
    "0:v:0",
    "-c",
    "copy",
    "-bsf:v",
    "h264_metadata",
    str(dst),
  ]
  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    raise RuntimeError(f"ffmpeg remux of {src} failed: {result.stderr.strip()}")
  return dst


def _shift_chunk(
  chunk: Chunk,
  timeline: str,
  shift_ns: int,
  window_ns: tuple[int, int] | None,
) -> Iterable[Chunk]:
  if chunk.is_static or timeline not in chunk.timeline_names:
    return [chunk]
  batch = chunk.to_record_batch()
  index = batch.schema.get_field_index(timeline)
  shifted = pc.add(batch.column(index), pa.scalar(shift_ns, type=pa.duration("ns")))
  if window_ns is not None:
    # GOP granularity: a boundary group stays whole and may spill past the window.
    values = [v.value for v in shifted]
    if not values:
      return []
    if max(values) < window_ns[0] or min(values) > window_ns[1]:
      return []
  batch = batch.set_column(index, batch.schema.field(index), shifted)
  return Chunk.from_record_batch(batch, index=timeline, entity_path=chunk.entity_path)


def video_chunks(
  mp4: Path,
  entity_path: str,
  timeline: str,
  time_shift: float = 0.0,
  window: tuple[float, float] | None = None,
) -> LazyChunkStream:
  """The file's video chunks on the session clock, GOP-trimmed to ``window``."""
  stream = Mp4Reader(
    mp4,
    timeline_name=timeline,
    entity_path=entity_path,
    transcode=Mp4TranscodeOptions(ffmpeg_override=resolve_ffmpeg()),
  ).stream()
  if time_shift == 0.0 and window is None:
    return stream
  shift_ns = round(time_shift * _NS)
  window_ns = None
  if window is not None:
    window_ns = (round(window[0] * _NS), round(window[1] * _NS))
  return stream.flat_map(lambda c: _shift_chunk(c, timeline, shift_ns, window_ns))
