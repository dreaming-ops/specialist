from pathlib import Path

import cv2
import numpy as np
import pytest

from specialist.loader.capture import probe_capture


def _write_video(path: Path, n_frames: int, width: int, height: int) -> None:
  writer = cv2.VideoWriter(
    str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
  )
  assert writer.isOpened()
  for i in range(n_frames):
    frame = np.full((height, width, 3), (i * 16) % 256, dtype=np.uint8)
    writer.write(frame)
  writer.release()


def test_probe_capture_finds_the_sibling_and_the_span_both_lenses_cover(
  tmp_path: Path,
) -> None:
  front = tmp_path / "VID_x_00_001.mp4"
  rear = tmp_path / "VID_x_10_001.mp4"
  _write_video(front, n_frames=13, width=64, height=64)
  _write_video(rear, n_frames=12, width=64, height=64)
  capture = probe_capture(front)
  assert capture.rear_path == rear
  assert capture.fps == pytest.approx(30.0)
  assert capture.duration == pytest.approx(12 / 30.0)


def test_probe_capture_square_without_sibling_raises(tmp_path: Path) -> None:
  video = tmp_path / "VID_x_00_001.mp4"
  _write_video(video, n_frames=4, width=64, height=64)
  with pytest.raises(RuntimeError, match="sibling"):
    probe_capture(video)


def test_probe_capture_non_square_video_raises(tmp_path: Path) -> None:
  video = tmp_path / "VID_x_00_001.mp4"
  rear = tmp_path / "VID_x_10_001.mp4"
  _write_video(video, n_frames=4, width=128, height=64)
  _write_video(rear, n_frames=4, width=128, height=64)
  with pytest.raises(RuntimeError, match="square"):
    probe_capture(video)
