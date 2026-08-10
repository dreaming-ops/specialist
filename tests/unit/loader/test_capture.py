from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from specialist.loader.capture import capture_duration, probe_capture


def _write_video(path: Path, n_frames: int, width: int, height: int) -> None:
  writer = cv2.VideoWriter(
    str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
  )
  assert writer.isOpened()
  for i in range(n_frames):
    frame = np.full((height, width, 3), (i * 16) % 256, dtype=np.uint8)
    writer.write(frame)
  writer.release()


def test_probe_capture_returns_sibling_and_container_fps(tmp_path: Path) -> None:
  front = tmp_path / "VID_x_00_001.mp4"
  rear = tmp_path / "VID_x_10_001.mp4"
  _write_video(front, n_frames=4, width=64, height=64)
  _write_video(rear, n_frames=4, width=64, height=64)
  rear_path, fps, native_width = probe_capture(front)
  assert rear_path == rear
  assert fps == pytest.approx(30.0)
  assert native_width == 64


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


def test_capture_duration_bounded_by_shorter_lens_file(tmp_path: Path) -> None:
  front = tmp_path / "VID_x_00_001.mp4"
  rear = tmp_path / "VID_x_10_001.mp4"
  _write_video(front, n_frames=13, width=64, height=64)
  _write_video(rear, n_frames=12, width=64, height=64)
  assert capture_duration(front) == pytest.approx(12 / 30.0)
