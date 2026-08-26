"""Keyframe times must land on video frame times in the base recording."""

from pathlib import Path

import gtsam
import numpy as np
import pytest
from specialist_dataset import schema
from specialist_dataset.frames import LensVideo
from specialist_dataset.read import Recording, check_frame_alignment

from specialist.record.base import write_base_recording
from specialist.record.session import SessionRecording
from specialist.record.video import normalize_h264, video_chunks
from specialist.vi_ba.viz import CameraEstimate
from tests.record_fixtures import encode_clip, urdf_file, write_run

_FPS = 30.0
_FRAMES = 90
# Most frames are not IDR frames, so the keyframe grid falls between them.
_GOP = 5
_KEYFRAME_STRIDE = 3


@pytest.fixture(scope="module")
def base_rrd(tmp_path_factory: pytest.TempPathFactory) -> Path:
  """A base recording holding one lens's encoded H.264 samples."""
  tmp = tmp_path_factory.mktemp("alignment")
  clip = encode_clip(tmp / "front.mp4", _FRAMES, rate=_FPS, gop=_GOP)
  clean = normalize_h264(clip, tmp / "front_clean.mp4")
  out = tmp / "base.rrd"
  write_base_recording(
    out,
    "s",
    [
      video_chunks(
        clean, schema.video("cam00", "front"), time_shift=0.0, window=(-1.0, 10.0)
      )
    ],
    [],
  )
  return out


def _write_layer(root: Path, kf_times: np.ndarray) -> Path:
  camera = CameraEstimate(
    slot="cam00",
    body_poses=[gtsam.Pose3() for _ in kf_times],
    keyframe_times=kf_times,
    body_T_base_tag=gtsam.Pose3(),
    base_tag_id=2000,
    keyframe_frames=np.arange(len(kf_times), dtype=np.int64) * 3,
    highrate_times=kf_times,
    highrate_positions=np.zeros((len(kf_times), 3)),
    highrate_quats=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(kf_times), 1)),
    highrate_vels=np.zeros((len(kf_times), 3)),
  )
  estimate, _viz = write_run(
    root,
    "s",
    [camera],
    urdf_path=urdf_file(root),
    frame_period=1.0 / 30.0,
  )
  return estimate


def _grid() -> np.ndarray:
  return np.arange(0, _FRAMES, _KEYFRAME_STRIDE) / _FPS


def test_base_holds_every_frame_not_just_the_idr_frames(base_rrd: Path) -> None:
  """The recording carries far more samples than IDR frames."""
  video = LensVideo(Recording(base_rrd), "cam00", "front")
  assert len(video.times) == _FRAMES
  assert 0 < len(video.sync_samples) < _FRAMES


def test_frame_times_cover_every_sample(base_rrd: Path) -> None:
  frame_times = SessionRecording(base_rrd).frame_times
  times = frame_times["cam00"]
  assert len(times) == _FRAMES
  np.testing.assert_allclose(times, np.arange(_FRAMES) / _FPS, atol=1e-6)


def test_aligned_layer_passes(base_rrd: Path, tmp_path: Path) -> None:
  """A keyframe grid landing mostly on non-IDR frames passes."""
  check_frame_alignment(
    SessionRecording(base_rrd).frame_times, _write_layer(tmp_path, _grid())
  )


def test_subframe_offset_rejected(base_rrd: Path, tmp_path: Path) -> None:
  layer = _write_layer(tmp_path, _grid() + 0.0026)
  with pytest.raises(ValueError, match="miss the video frames"):
    check_frame_alignment(SessionRecording(base_rrd).frame_times, layer)


def test_unknown_camera_rejected(base_rrd: Path, tmp_path: Path) -> None:
  layer = _write_layer(tmp_path, _grid())
  with pytest.raises(ValueError, match="no video frame times"):
    check_frame_alignment({"cam09": np.arange(_FRAMES) / _FPS}, layer)
