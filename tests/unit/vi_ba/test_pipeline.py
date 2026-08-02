"""``vi_ba.pipeline``: capture probing and decoding, and the parallax filter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from specialist.loader.mei_model import MeiParams, mei_project, mei_unproject
from specialist.vi_ba.pipeline import (
  capture_duration,
  decode_dual_keyframes,
  filter_tracks_by_parallax,
  probe_capture,
  unproject_tracks,
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


def test_unproject_tracks_batched_slices_match_per_track_scalar() -> None:
  """Bearing rows match exactly, kfs come back sorted and tracks below min_obs drop."""
  params = _params()
  tracks = {
    7: {
      2: np.array([120.0, 90.0]),
      0: np.array([300.0, 310.0]),
      1: np.array([50.0, 470.0]),
    },
    3: {5: np.array([460.0, 12.0]), 4: np.array([470.0, 495.0])},
    9: {0: np.array([240.0, 240.0])},  # single obs -> below min_obs
  }
  out = {
    tid: (kfs, bearings)
    for tid, kfs, bearings in unproject_tracks(tracks, params, min_obs=2)
  }
  assert set(out) == {7, 3}  # the single-observation track is filtered out
  for tid, (kfs, bearings) in out.items():
    assert kfs == sorted(tracks[tid])
    expected = mei_unproject(
      np.asarray([tracks[tid][k] for k in kfs], dtype=np.float64), params
    )
    assert np.array_equal(bearings, expected)


def _write_video(path: Path, n_frames: int, width: int, height: int) -> None:
  """Frame ``i`` halves are ``i * 16`` and ``i * 16 + 8`` bright, wrapping every 16."""
  writer = cv2.VideoWriter(
    str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
  )
  assert writer.isOpened()
  for i in range(n_frames):
    frame = np.full((height, width, 3), (i * 16) % 256, dtype=np.uint8)
    frame[:, width // 2 :] = (i * 16 + 8) % 256
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


def test_decode_captures_are_bounded_across_the_whole_process() -> None:
  """64 threads in ``_decode_slot`` stay within the process-wide capture cap."""
  import threading
  import time

  from specialist.vi_ba import pipeline

  peak = 0
  live = 0
  guard = threading.Lock()

  def hold() -> None:
    nonlocal peak, live
    with pipeline._decode_slot():
      with guard:
        live += 1
        peak = max(peak, live)
      time.sleep(0.005)
      with guard:
        live -= 1

  threads = [threading.Thread(target=hold) for _ in range(64)]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()
  assert peak <= pipeline._DECODE_MAX_CONCURRENT_CAPTURES
  assert peak > 0


def test_a_short_middle_segment_is_flagged() -> None:
  """A middle segment that stops early misaligns every later keyframe."""
  from specialist.vi_ba import pipeline

  per_file = [[([0] * 10, []), ([0] * 9, []), ([0] * 10, [])]]
  assert pipeline._short_segments(per_file, [0, 100, 200, 300], 10) == [(0, 1)]


def test_a_short_final_segment_is_legitimate() -> None:
  """Only the last segment may run out: that is the end of the stream."""
  from specialist.vi_ba import pipeline

  per_file = [[([0] * 10, []), ([0] * 3, [])]]
  assert pipeline._short_segments(per_file, [0, 100, 200], 10) == []


def test_full_segments_flag_nothing() -> None:
  from specialist.vi_ba import pipeline

  per_file = [[([0] * 10, []), ([0] * 10, [])], [([0] * 10, []), ([0] * 10, [])]]
  assert pipeline._short_segments(per_file, [0, 100, 200], 10) == []


def test_per_lens_decode_matches_the_dual_decode(tmp_path: Path) -> None:
  from specialist.vi_ba.pipeline import decode_lens_keyframes, shared_keyframe_bound

  front = tmp_path / "VID_x_00_001.mp4"
  rear = tmp_path / "VID_x_10_001.mp4"
  _write_video(front, n_frames=13, width=64, height=64)
  _write_video(rear, n_frames=12, width=64, height=64)

  f_gray, r_gray, _, _ = decode_dual_keyframes(
    front, rear, 30.0, start=0.0, duration=None, keyframe_every=3, keep_color=False
  )
  bound = shared_keyframe_bound(front, rear)
  solo_f, _ = decode_lens_keyframes(
    front, bound, 30.0, start=0.0, duration=None, keyframe_every=3, keep_color=False
  )
  solo_r, _ = decode_lens_keyframes(
    rear, bound, 30.0, start=0.0, duration=None, keyframe_every=3, keep_color=False
  )
  n = min(len(solo_f), len(solo_r))
  assert len(f_gray) == len(r_gray) == n
  for a, b in zip(f_gray, solo_f[:n], strict=True):
    assert np.array_equal(a, b)
  for a, b in zip(r_gray, solo_r[:n], strict=True):
    assert np.array_equal(a, b)


def test_early_stop_names_eof_apart_from_a_failed_decode(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """A failed decode and end of stream look alike, so position tells them apart."""
  from specialist.vi_ba.pipeline import _report_early_stop

  video = tmp_path / "VID_x_00_001.mp4"
  _write_video(video, n_frames=6, width=64, height=64)
  cap = cv2.VideoCapture(str(video))
  while cap.grab():
    pass
  _report_early_stop(cap, video, 6, container_frames=6)
  assert "end of stream" in capsys.readouterr().out

  cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
  cap.grab()
  _report_early_stop(cap, video, 3, container_frames=6)
  assert "decode failed" in capsys.readouterr().out
  cap.release()


def test_banded_keyframes_match_the_eager_decode(tmp_path: Path) -> None:
  """Banded reads match the eager decode frame by frame, and an earlier band reloads."""
  from specialist.vi_ba.pipeline import (
    BandedKeyframes,
    decode_lens_keyframes,
    shared_keyframe_bound,
  )

  front = tmp_path / "VID_x_00_001.mp4"
  rear = tmp_path / "VID_x_10_001.mp4"
  _write_video(front, n_frames=15, width=64, height=64)
  _write_video(rear, n_frames=15, width=64, height=64)

  bound = shared_keyframe_bound(front, rear)
  eager, _ = decode_lens_keyframes(
    front, bound, 30.0, start=0.0, duration=None, keyframe_every=2, keep_color=False
  )
  banded = BandedKeyframes(front, bound, 30.0, 0.0, None, keyframe_every=2, band=3)

  assert len(banded) == len(eager)
  for i, expected in enumerate(eager):
    assert np.array_equal(banded[i], expected), f"frame {i} differs"
  # re-reading an earlier band must reload it rather than return stale pixels
  assert np.array_equal(banded[0], eager[0])


def test_a_transient_short_read_is_retried_and_recovers(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """One spurious mid-segment end costs a re-decode of that span, never the run."""
  import threading

  from specialist.vi_ba import pipeline

  video = tmp_path / "VID_x_00_001.mp4"
  _write_video(video, n_frames=220, width=64, height=64)
  clean, _ = pipeline.decode_lens_keyframes(
    video, 220, 30.0, start=0.0, duration=None, keyframe_every=3, keep_color=False
  )

  real_capture = cv2.VideoCapture
  lock = threading.Lock()
  fired: list[bool] = []

  class OneShortRead:
    """Fails once in the span from frame 0, the only span that never seeks."""

    def __init__(self, path: str) -> None:
      self._cap = real_capture(path)
      self._seeked = False

    def _fails_now(self) -> bool:
      with lock:
        if self._seeked or fired:
          return False
        fired.append(True)
        return True

    def set(self, prop: int, value: float) -> bool:
      self._seeked = True
      return bool(self._cap.set(prop, value))

    def get(self, prop: int) -> float:
      return float(self._cap.get(prop))

    def grab(self) -> bool:
      return False if self._fails_now() else bool(self._cap.grab())

    def read(self) -> tuple[bool, np.ndarray | None]:
      if self._fails_now():
        return False, None
      ok, frame = self._cap.read()
      return bool(ok), frame

    def release(self) -> None:
      self._cap.release()

  monkeypatch.setattr(pipeline.cv2, "VideoCapture", OneShortRead)
  recovered, _ = pipeline.decode_lens_keyframes(
    video, 220, 30.0, start=0.0, duration=None, keyframe_every=3, keep_color=False
  )

  assert fired, "the injected short read never fired, so nothing was retried"
  assert len(recovered) == len(clean)
  for a, b in zip(recovered, clean, strict=True):
    assert np.array_equal(a, b)


def test_a_band_that_comes_up_short_is_named(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The decode guard forgives a band's mid-file short read; a count check names it."""
  from specialist.vi_ba import pipeline

  video = tmp_path / "VID_x_00_001.mp4"
  _write_video(video, n_frames=60, width=64, height=64)
  frames = pipeline.BandedKeyframes(
    video, 60, 30.0, start=0.0, duration=None, keyframe_every=3, band=4
  )
  real = pipeline.decode_lens_keyframes
  monkeypatch.setattr(
    pipeline,
    "decode_lens_keyframes",
    lambda *a, **k: (real(*a, **k)[0][:-1], []),
  )
  with pytest.raises(RuntimeError, match="band"):
    frames[0]
