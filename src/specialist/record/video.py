"""Insta360 H.264 ingest, losslessly remuxed to fix an SPS rerun's parser rejects."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from rerun.chunk import Chunk, LazyChunkStream
from rerun.experimental import Mp4Reader, Mp4TranscodeOptions
from specialist_dataset import schema

from specialist.loader.audio_sync import resolve_ffmpeg


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
  shift_ns: int,
  window_ns: tuple[int, int],
  period_ns: float,
  stride: int,
  sync_samples: int,
) -> Iterable[Chunk]:
  if chunk.is_static or schema.TIMELINE not in chunk.timeline_names:
    return [chunk]
  batch = chunk.to_record_batch()
  # The reader marks every sync sample in one chunk, a row each.
  if "VideoStream:is_keyframe" in batch.schema.names and batch.num_rows != sync_samples:
    raise RuntimeError(
      f"{chunk.entity_path}: the reader marked {batch.num_rows} sync samples but "
      f"the container holds {sync_samples} (the reader transcodes a source with "
      "B-frames); the recording would not be randomly accessible"
    )
  index = batch.schema.get_field_index(schema.TIMELINE)
  shifted = pc.add(batch.column(index), pa.scalar(shift_ns, type=pa.duration("ns")))
  # GOP-granular trim: a group may spill past the window to keep every in-window frame.
  values = [v.value for v in shifted]
  if not values or max(values) < window_ns[0] or min(values) > window_ns[1]:
    return []
  batch = batch.set_column(index, batch.schema.field(index), shifted)
  # Bucket by session-clock frame number: every camera shows the same instant.
  session_frames = np.asarray([round(v / period_ns) for v in values], dtype=np.int64)
  batch = batch.append_column(
    pa.field(schema.KEYFRAME_TIMELINE, pa.int64()),
    pa.array(schema.keyframe_index(session_frames, stride), type=pa.int64()),
  )
  return Chunk.from_record_batch(
    batch,
    index=[schema.TIMELINE, schema.KEYFRAME_TIMELINE],
    entity_path=chunk.entity_path,
  )


def _container_metadata(mp4: Path) -> tuple[float, int]:
  """The frame period from the container's packet PTS, and its sync-sample count."""
  import av

  pts: list[int] = []
  sync_samples = 0
  with av.open(str(mp4)) as container:
    for p in container.demux(container.streams.video[0]):
      if p.size and p.pts is not None:
        pts.append(round(p.pts * p.time_base * schema.NS_PER_SECOND))
        sync_samples += p.is_keyframe
  if len(pts) < 2:
    return float(schema.NS_PER_SECOND), sync_samples
  return float(np.median(np.diff(np.sort(pts)))), sync_samples


def video_chunks(
  mp4: Path,
  entity_path: str,
  time_shift: float,
  window: tuple[float, float],
) -> LazyChunkStream:
  """The file's video chunks on the session clock, GOP-trimmed to ``window``."""
  shift_ns = round(time_shift * schema.NS_PER_SECOND)
  window_ns = (
    round(window[0] * schema.NS_PER_SECOND),
    round(window[1] * schema.NS_PER_SECOND),
  )
  period_ns, sync_samples = _container_metadata(mp4)
  stride = schema.keyframe_stride(schema.NS_PER_SECOND / period_ns)
  stream = Mp4Reader(
    mp4,
    timeline_name=schema.TIMELINE,
    entity_path=entity_path,
    transcode=Mp4TranscodeOptions(ffmpeg_override=resolve_ffmpeg()),
  ).stream()
  return stream.flat_map(
    lambda c: _shift_chunk(c, shift_ns, window_ns, period_ns, stride, sync_samples)
  )
