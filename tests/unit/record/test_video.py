"""Video ingest: SPS-normalizing remux and VideoStream chunk production."""

from pathlib import Path

import numpy as np
import pytest

from specialist.record import schema
from specialist.record.video import normalize_h264, video_chunks
from tests.record_fixtures import encode_clip

_FRAMES = 12
_RATE = 30
# Wide enough to keep every frame.
_WHOLE_CLIP = (-10.0, 10.0)


@pytest.fixture(scope="module")
def h264_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
  return encode_clip(
    tmp_path_factory.mktemp("clip") / "clip.mp4", _FRAMES, rate=_RATE, bframes=True
  )


@pytest.fixture(scope="module")
def clean_clip(tmp_path_factory: pytest.TempPathFactory, h264_clip: Path) -> Path:
  return normalize_h264(h264_clip, tmp_path_factory.mktemp("clean") / "clean.mp4")


def _collect(clip: Path, shift: float) -> tuple[list[object], list[int], list[bool]]:
  stream = video_chunks(clip, "image/cam/front", time_shift=shift, window=_WHOLE_CLIP)
  static, times, sync = [], [], []
  for chunk in stream.to_chunks():
    assert chunk.entity_path == "/image/cam/front"
    if chunk.is_static:
      static.append(chunk)
      continue
    batch = chunk.to_record_batch()
    index = batch.schema.get_field_index("video_time")
    times += [v.value for v in batch.column(index)]
    key_index = batch.schema.get_field_index("VideoStream:is_keyframe")
    sync += [bool(v.as_py()) for v in batch.column(key_index)]
  return static, times, sync


def _keyframe_frames(clip: Path, shift: float) -> list[tuple[int, int]]:
  """Every surviving video frame as ``(session time, keyframe index)``, sorted."""
  frames: list[tuple[int, int]] = []
  for chunk in video_chunks(
    clip, "image/t/front", time_shift=shift, window=(0.0, 10.0)
  ).to_chunks():
    if chunk.is_static:
      continue
    batch = chunk.to_record_batch()
    names = batch.schema.names
    assert schema.KEYFRAME_TIMELINE in names, "keyframe index missing from chunk"
    t = batch.column(names.index(schema.TIMELINE))
    k = batch.column(names.index(schema.KEYFRAME_TIMELINE))
    frames.extend((t[i].value, k[i].as_py()) for i in range(batch.num_rows))
  frames.sort()
  return frames


def test_normalize_and_chunks(clean_clip: Path) -> None:
  static, times, sync = _collect(clean_clip, shift=0.0)
  assert len(static) == 1
  assert len(times) == _FRAMES
  assert times == sorted(times)
  assert times[0] == 0
  frame_ns = round(1e9 / _RATE)
  assert abs(times[1] - frame_ns) <= 1
  assert sync[0] is True


def test_time_shift(clean_clip: Path) -> None:
  _, unshifted, _ = _collect(clean_clip, shift=0.0)
  _, shifted, _ = _collect(clean_clip, shift=0.5)
  assert shifted == [t + 500_000_000 for t in unshifted]


def test_window_keeps_and_drops_chunks(clean_clip: Path) -> None:
  def rows(window: tuple[float, float]) -> int:
    stream = video_chunks(clean_clip, "image/cam/front", time_shift=0.0, window=window)
    return sum(c.num_rows for c in stream.to_chunks() if not c.is_static)

  # A chunk overlapping the window survives whole; one entirely outside drops.
  assert rows((0.0, 10.0)) == _FRAMES
  assert rows((0.2, 10.0)) == _FRAMES
  assert rows((100.0, 200.0)) == 0


def test_remux_failure_raises(tmp_path: Path) -> None:
  bogus = tmp_path / "bogus.mp4"
  bogus.write_bytes(b"not a video")
  with pytest.raises(RuntimeError, match="ffmpeg remux"):
    normalize_h264(bogus, tmp_path / "out.mp4")


def test_keyframe_timeline_buckets_on_the_session_clock(h264_clip: Path) -> None:
  """A capture shifted two frames into the session buckets by session frame number."""
  rows = _keyframe_frames(h264_clip, shift=2.0 / _RATE)
  period = float(np.median(np.diff([t for t, _ in rows])))
  stride = schema.keyframe_stride(schema.NS_PER_SECOND / period)
  expected = schema.keyframe_index(np.arange(len(rows)) + 2, stride)
  assert [k for _, k in rows] == expected.tolist()


def test_dropped_sync_sample_markers_fail_ingest(tmp_path: Path) -> None:
  """In a short-GOP B-frame clip, the reader's keyframe markers collapse."""
  clip = encode_clip(tmp_path / "bframes.mp4", _RATE, rate=_RATE, gop=5, bframes=True)
  clean = normalize_h264(clip, tmp_path / "clean.mp4")
  with pytest.raises(RuntimeError, match="randomly accessible"):
    video_chunks(clean, "image/cam/front", time_shift=0.0, window=_WHOLE_CLIP)
