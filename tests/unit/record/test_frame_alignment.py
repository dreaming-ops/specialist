"""Keyframe times must land on video frame times in the base recording."""

import subprocess
from pathlib import Path

import gtsam
import numpy as np
import pytest

from specialist.loader.audio_sync import resolve_ffmpeg
from specialist.record.base import write_base_recording
from specialist.record.frames import LensVideo
from specialist.record.layer import write_track_layer
from specialist.record.read import check_frame_alignment
from specialist.record.session import SessionRecording
from specialist.record.video import normalize_h264, video_chunks
from specialist.vi_ba.summary import Summary
from specialist.vi_ba.viz import CameraVizData
from specialist.vi_ba.viz_build import RunOutputs

_FPS = 30.0
_FRAMES = 90
# Most frames are not IDR frames, so the keyframe grid falls between them.
_GOP = 5
_KEYFRAME_STRIDE = 3


@pytest.fixture(scope="module")
def base_rrd(tmp_path_factory: pytest.TempPathFactory) -> Path:
  """A base recording holding one lens's encoded H.264 samples."""
  tmp = tmp_path_factory.mktemp("alignment")
  clip = tmp / "front.mp4"
  subprocess.run(
    [
      resolve_ffmpeg(),
      "-y",
      "-loglevel",
      "error",
      "-f",
      "lavfi",
      "-i",
      f"testsrc=duration={_FRAMES / _FPS}:size=64x64:rate={_FPS:g}",
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      "-bf",
      "0",
      "-g",
      str(_GOP),
      str(clip),
    ],
    check=True,
  )
  clean = normalize_h264(clip, tmp / "front_clean.mp4")
  out = tmp / "base.rrd"
  write_base_recording(
    out,
    "s",
    [video_chunks(clean, "image/tracker_0/front", time_shift=0.0, window=(-1.0, 10.0))],
    [],
    {},
  )
  return out


def _write_layer(path: Path, kf_times: np.ndarray) -> None:
  poses = [gtsam.Pose3() for _ in kf_times]
  outputs = RunOutputs(
    viz_trackers=[
      CameraVizData(name="tracker_0", body_poses=poses, keyframe_times=kf_times)
    ],
    per_tracker=[],
    world_T_anchor={0: gtsam.Pose3()},
  )
  summary = Summary(("a",), ("tracker_0",), 0, 1.0, 0.5, 1)
  write_track_layer(
    rrd_path=path,
    outputs=outputs,
    summary=summary,
    reference_tag_id=0,
    anchor_tag_size=0.1,
    session_id="s",
    mask_pad_px=-1,
    open_viewer=False,
  )


def _grid() -> np.ndarray:
  return np.arange(0, _FRAMES, _KEYFRAME_STRIDE) / _FPS


def test_base_holds_every_frame_not_just_the_idr_frames(base_rrd: Path) -> None:
  """The recording carries far more samples than IDR frames."""
  video = LensVideo(base_rrd, "tracker_0", "front")
  assert len(video.times) == _FRAMES
  assert 0 < len(video.keyframes) < _FRAMES


def test_frame_times_cover_every_sample(base_rrd: Path) -> None:
  frame_times = SessionRecording(base_rrd).frame_times
  times = frame_times["tracker_0"]
  assert len(times) == _FRAMES
  np.testing.assert_allclose(times, np.arange(_FRAMES) / _FPS, atol=1e-6)


def test_aligned_layer_passes(base_rrd: Path, tmp_path: Path) -> None:
  """A keyframe grid landing mostly on non-IDR frames passes."""
  _write_layer(tmp_path / "layer.rrd", _grid())
  check_frame_alignment(SessionRecording(base_rrd).frame_times, tmp_path / "layer.rrd")


def test_subframe_offset_rejected(base_rrd: Path, tmp_path: Path) -> None:
  _write_layer(tmp_path / "layer.rrd", _grid() + 0.0026)
  with pytest.raises(ValueError, match="miss the video frames"):
    check_frame_alignment(
      SessionRecording(base_rrd).frame_times, tmp_path / "layer.rrd"
    )


def test_unknown_tracker_rejected(base_rrd: Path, tmp_path: Path) -> None:
  _write_layer(tmp_path / "layer.rrd", _grid())
  with pytest.raises(ValueError, match="no video frame times"):
    check_frame_alignment(
      {"tracker_9": np.arange(_FRAMES) / _FPS}, tmp_path / "layer.rrd"
    )
