"""``vi_ba.pipeline``: capture probing and decoding, and the parallax filter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from specialist.loader.mei_model import MeiParams, mei_project
from specialist.vi_ba.pipeline import (
  capture_duration,
  decode_dual_keyframes,
  filter_tracks_by_parallax,
  probe_capture,
)


def _params() -> MeiParams:
  return MeiParams(
    fx=400.0,
    fy=400.0,
    cx=480.0,
    cy=480.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
    xi=1.0,
  )


def _project_one(point_cam: np.ndarray, params: MeiParams) -> np.ndarray:
  return mei_project(point_cam.reshape(1, 3), params)[0]


def test_static_camera_track_kept_only_if_parallax_above_threshold() -> None:
  """Zero-parallax tracks, like body-attached features, drop; a FOV sweep stays."""
  params = _params()
  body_mounted_pixel = _project_one(np.array([0.05, 0.0, 0.5]), params)
  static_track = {kf: body_mounted_pixel.copy() for kf in range(5)}

  moving_track = {
    kf: _project_one(np.array([0.05 + 0.05 * kf, 0.0, 0.5]), params) for kf in range(5)
  }

  tracks = {1: static_track, 2: moving_track}
  kept, rejected = filter_tracks_by_parallax(
    tracks, params, min_parallax=np.deg2rad(0.5)
  )
  assert 1 not in kept and 1 in rejected, "static-bearing track must be rejected"
  assert 2 in kept and 2 not in rejected, "wide-bearing-sweep track must be kept"


def _write_video(path: Path, n_frames: int, width: int, height: int) -> None:
  """Frame ``i`` halves are ``i * 16`` and ``i * 16 + 8`` bright."""
  writer = cv2.VideoWriter(
    str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
  )
  assert writer.isOpened()
  for i in range(n_frames):
    frame = np.full((height, width, 3), i * 16, dtype=np.uint8)
    frame[:, width // 2 :] = i * 16 + 8
    writer.write(frame)
  writer.release()


def _brightness_index(frame: np.ndarray) -> int:
  return int(round(float(frame.mean()) / 16.0))


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


def test_decode_duration_past_file_end_clamps_to_container(tmp_path: Path) -> None:
  front = tmp_path / "VID_x_00_001.mp4"
  rear = tmp_path / "VID_x_10_001.mp4"
  _write_video(front, n_frames=13, width=64, height=64)
  _write_video(rear, n_frames=12, width=64, height=64)
  overrun = decode_dual_keyframes(
    front, rear, 30.0, start=0.0, duration=10.0, keyframe_every=3, keep_color=False
  )
  assert len(overrun[0]) == len(overrun[1]) == 4


def test_decode_lens_pair_reads_both_files_aligned(tmp_path: Path) -> None:
  front_path = tmp_path / "VID_x_00_001.mp4"
  rear_path = tmp_path / "VID_x_10_001.mp4"
  # The master file has one extra trailing frame; the decode clamps to the shared range.
  _write_video(front_path, n_frames=13, width=64, height=64)
  _write_video(rear_path, n_frames=12, width=64, height=64)
  front_gray, rear_gray, front_color, rear_color = decode_dual_keyframes(
    front_path,
    rear_path,
    30.0,
    start=0.0,
    duration=None,
    keyframe_every=3,
    keep_color=True,
  )
  no_color = decode_dual_keyframes(
    front_path,
    rear_path,
    30.0,
    start=0.0,
    duration=None,
    keyframe_every=3,
    keep_color=False,
  )
  assert no_color[2] == [] and no_color[3] == []
  assert len(front_gray) == len(rear_gray) == 4
  half = 64
  for k, (fg, rg) in enumerate(zip(front_gray, rear_gray, strict=True)):
    assert fg.shape == rg.shape == (half, half)  # per-lens decode resolution
    assert _brightness_index(fg[:, : half // 2]) == k * 3  # left half of frame k*3
    assert _brightness_index(rg[:, : half // 2]) == k * 3  # same frame in _10_
  first = cv2.imdecode(np.frombuffer(front_color[0], np.uint8), cv2.IMREAD_COLOR)
  assert isinstance(front_color[0], bytes) and first.shape == (half, half, 3)
  assert isinstance(rear_color[0], bytes)
