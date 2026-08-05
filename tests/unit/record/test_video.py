"""Video ingest: SPS-normalizing remux and VideoStream chunk production."""

import subprocess
from pathlib import Path

import pytest

from specialist.loader.audio_sync import resolve_ffmpeg
from specialist.record.video import normalize_h264, video_chunks

_FRAMES = 12
_RATE = 30


@pytest.fixture(scope="module")
def h264_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
  path = tmp_path_factory.mktemp("clip") / "clip.mp4"
  subprocess.run(
    [
      resolve_ffmpeg(),
      "-y",
      "-loglevel",
      "error",
      "-f",
      "lavfi",
      "-i",
      f"testsrc=duration={_FRAMES / _RATE}:size=64x64:rate={_RATE}",
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      str(path),
    ],
    check=True,
  )
  return path


def _collect(clip: Path, shift: float) -> tuple[list[object], list[int], list[bool]]:
  stream = video_chunks(clip, "image/cam/front", "video_time", time_shift=shift)
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


def test_normalize_and_chunks(tmp_path: Path, h264_clip: Path) -> None:
  clean = normalize_h264(h264_clip, tmp_path / "clean.mp4")
  static, times, sync = _collect(clean, shift=0.0)
  assert len(static) == 1
  assert len(times) == _FRAMES
  assert times == sorted(times)
  assert times[0] == 0
  frame_ns = round(1e9 / _RATE)
  assert abs(times[1] - frame_ns) <= 1
  assert sync[0] is True


def test_time_shift(tmp_path: Path, h264_clip: Path) -> None:
  clean = normalize_h264(h264_clip, tmp_path / "clean.mp4")
  _, unshifted, _ = _collect(clean, shift=0.0)
  _, shifted, _ = _collect(clean, shift=0.5)
  assert shifted == [t + 500_000_000 for t in unshifted]


def test_window_keeps_and_drops_chunks(tmp_path: Path, h264_clip: Path) -> None:
  clean = normalize_h264(h264_clip, tmp_path / "clean.mp4")

  def rows(window: tuple[float, float]) -> int:
    stream = video_chunks(clean, "image/cam/front", "video_time", window=window)
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
