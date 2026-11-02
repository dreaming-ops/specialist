"""Frame decoding out of a base recording's video stream."""

from pathlib import Path

import pytest
import rerun as rr
from rerun.chunk import Chunk
from specialist_dataset import frames, schema
from specialist_dataset.read import Recording


def _base_with_video(tmp_path: Path, *, sync: bool) -> Path:
  """A base layer of one lens's samples, with or without a sync-sample marker."""
  rrd = tmp_path / "base.rrd"
  samples = [b"\x00\x01", b"\x02\x03", b"\x04\x05"]
  chunk = Chunk.from_columns(
    schema.video("cam00", "front"),
    indexes=[rr.TimeColumn(schema.TIMELINE, duration=[0.0, 0.1, 0.2])],
    columns=rr.VideoStream.columns(sample=samples, is_keyframe=[sync, False, False]),
  )
  with rr.RecordingStream("test", recording_id="s") as stream:
    stream.save(str(rrd))
    rr.send_chunks([chunk], recording=stream)
  return rrd


def test_a_lens_reads_its_sample_times(tmp_path: Path) -> None:
  recording = Recording(_base_with_video(tmp_path, sync=True))
  times = frames.lens_times(recording, schema.video("cam00", "front"))
  assert list(times) == pytest.approx([0.0, 0.1, 0.2])


def test_a_lens_starts_a_decode_at_its_sync_sample(tmp_path: Path) -> None:
  recording = Recording(_base_with_video(tmp_path, sync=True))
  video = frames.LensVideo(recording, "cam00", "front", tmp_path)
  assert list(video.sync_samples) == [0]
  assert video.decode_start(2) == 0
  assert [video.packet(sample) for sample in range(3)] == [
    b"\x00\x01",
    b"\x02\x03",
    b"\x04\x05",
  ]


def test_a_lens_stages_its_packets_on_disk_until_closed(tmp_path: Path) -> None:
  recording = Recording(_base_with_video(tmp_path, sync=True))
  stage = tmp_path / "stage"
  stage.mkdir()
  video = frames.LensVideo(recording, "cam00", "front", stage)
  staged = list(stage.iterdir())
  assert len(staged) == 1 and staged[0].stat().st_size == 6
  video.close()
  assert list(stage.iterdir()) == []


def test_lens_video_refuses_markerless_streams(tmp_path: Path) -> None:
  rrd = _base_with_video(tmp_path, sync=False)
  with pytest.raises(RuntimeError, match="no sync-sample markers"):
    frames.LensVideo(Recording(rrd), "cam00", "front", tmp_path)


def test_a_lens_loads_once_until_released(tmp_path: Path) -> None:
  recording = Recording(_base_with_video(tmp_path, sync=True))
  first = frames.lens_video(recording, "cam00", "front", tmp_path)
  assert frames.lens_video(recording, "cam00", "front", tmp_path) is first
  assert frames.lens_times(recording, schema.video("cam00", "front")) is first.times
  frames.release_videos()
  assert not list(tmp_path.glob("*.packets"))
  assert frames.lens_video(recording, "cam00", "front", tmp_path) is not first
