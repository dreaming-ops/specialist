"""``SessionRecording`` reads back from a base rrd everything the solve needs."""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import pytest
from rerun.experimental import RrdReader
from specialist_dataset import schema
from specialist_dataset.frames import LensFrames

from specialist.record.base import (
  BODY_R_IMU,
  calibration_chunks,
  capture_chunk,
  exposure_chunks,
  imu_chunks,
  write_base_recording,
)
from specialist.record.offset_v3 import parse_offset_v3
from specialist.record.session import (
  SessionRecording,
  covered_samples,
  fps_from_times,
  nearest_samples,
)
from specialist.record.video import normalize_h264, video_chunks
from tests.record_fixtures import encode_clip, synthetic_insv

_FPS = 30
# The rear lens starts a frame before the front and ends two frames after it.
_FRONT_FRAMES = 18
_REAR_FRAMES = 21
_REAR_SHIFT = -1.0 / _FPS
# Wide enough to keep every frame and IMU sample.
_WINDOW = (-1.0, 10.0)


@pytest.fixture(scope="module")
def base_rrd(tmp_path_factory: pytest.TempPathFactory) -> Path:
  tmp = tmp_path_factory.mktemp("session")
  insv = synthetic_insv()
  streams = []
  for lens, n_frames, shift in (
    ("front", _FRONT_FRAMES, 0.0),
    ("rear", _REAR_FRAMES, _REAR_SHIFT),
  ):
    clip = encode_clip(tmp / f"{lens}.mp4", n_frames, rate=_FPS, gop=5)
    clean = normalize_h264(clip, tmp / f"{lens}_clean.mp4")
    streams.append(
      video_chunks(clean, schema.video("cam00", lens), time_shift=shift, window=_WINDOW)
    )
  sidecars = [
    *imu_chunks("cam00", insv, time_shift=0.0, window=_WINDOW),
    *exposure_chunks("cam00", insv, time_shift=0.0, window=_WINDOW),
    *calibration_chunks("cam00", insv),
    capture_chunk("cam00", insv, "a_00_.insv", -0.25, 4.5),
  ]
  out = tmp / "base.rrd"
  write_base_recording(out, "sess", streams, sidecars)
  return out


def test_identity_slots(base_rrd: Path) -> None:
  session = SessionRecording(base_rrd)
  assert session.session_id == "sess"
  assert session.slots == ["cam00"]


def test_fps_is_pts_derived(base_rrd: Path) -> None:
  session = SessionRecording(base_rrd)
  front = session.lens_times("cam00", "front")
  rear = session.lens_times("cam00", "rear")
  assert fps_from_times(front) == pytest.approx(_FPS, rel=1e-6)
  assert len(front) == _FRONT_FRAMES
  assert len(rear) == _REAR_FRAMES
  assert front[0] == pytest.approx(0.0, abs=1e-9)
  assert rear[0] == pytest.approx(_REAR_SHIFT, abs=1e-6)


def test_fps_from_times_undoes_ntsc_rounding() -> None:
  times = np.arange(300) * 1001 / 30000
  assert fps_from_times(times) == pytest.approx(30000 / 1001, rel=1e-9)
  with pytest.raises(ValueError, match="two frame times"):
    fps_from_times(np.array([0.0]))


def test_grid_alignment_matches_rear_samples_by_pts(base_rrd: Path) -> None:
  """Nearest-PTS matching on the front grid absorbs the rear lens's edge trim."""
  session = SessionRecording(base_rrd)
  front = session.lens_video("cam00", "front")
  rear = session.lens_video("cam00", "rear")
  grid = front.times[::2]
  samples = nearest_samples(rear.times, grid)
  np.testing.assert_array_equal(samples, np.arange(1, _FRONT_FRAMES + 1, 2))
  assert float(np.abs(rear.times[samples] - grid).max()) < 0.5 / _FPS


def test_nearest_samples_raises_past_half_a_frame() -> None:
  times = np.arange(15) / 30.0
  grid = np.arange(0, 20, 2) / 30.0  # tail grid points have no matching sample
  with pytest.raises(ValueError, match="half a frame"):
    nearest_samples(times, grid)
  # a sub-half-frame jitter is matched, not rejected
  samples = nearest_samples(times, times[::2] + 0.25 / 30.0)
  np.testing.assert_array_equal(samples, np.arange(0, 15, 2))


def test_covered_samples_spans_then_clips_to_the_common_overlap() -> None:
  """Samples outside the three-way overlap drop, with half a frame of slack per edge."""
  fps = 30.0
  front = np.arange(10) / fps
  rear = np.arange(-1, 11) / fps
  imu = np.arange(-5, 55) / (5 * fps)
  np.testing.assert_array_equal(covered_samples(front, rear, imu, fps), np.arange(10))
  # Rear and IMU miss samples 2 and 6 by a third of a frame; the slack keeps both.
  late_rear = np.arange(10) / fps + 2.3 / fps
  short_imu = np.arange(58) / (10 * fps)
  np.testing.assert_array_equal(
    covered_samples(front, late_rear, short_imu, fps), np.arange(2, 7)
  )


def test_lens_frames_at_chosen_samples_are_exact(base_rrd: Path) -> None:
  session = SessionRecording(base_rrd)
  video = session.lens_video("cam00", "front")
  reference = LensFrames(video, np.arange(_FRONT_FRAMES), band=64)
  chosen = np.array([0, 3, 5, 11, 17])
  banded = LensFrames(video, chosen, band=2)
  assert len(banded) == len(chosen)
  for i, sample in enumerate(chosen):
    assert np.array_equal(banded[i], reference[int(sample)]), f"sample {sample} differs"
  # The samples index the lens's own PTS table.
  np.testing.assert_allclose(video.times[chosen], chosen / _FPS, atol=1e-6)


def test_seam_reach_back_does_not_redecode(base_rrd: Path) -> None:
  """An ascending pair walk across band seams costs one monotone decode sweep."""
  session = SessionRecording(base_rrd)
  video = session.lens_video("cam00", "front")
  reference = LensFrames(video, np.arange(_FRONT_FRAMES), band=64)
  banded = LensFrames(video, np.arange(_FRONT_FRAMES), band=4)
  decodes: list[int] = []
  original = banded._decode_band

  def counting(first: int) -> list[np.ndarray]:
    decodes.append(first)
    return original(first)

  banded._decode_band = counting  # type: ignore[method-assign]
  for k in range(1, _FRONT_FRAMES):
    cur, prev = banded[k], banded[k - 1]
    assert np.array_equal(cur, reference[k])
    assert np.array_equal(prev, reference[k - 1])
  assert decodes == [0, 4, 8, 12, 16]


def test_decoded_shape_and_lens_setups(base_rrd: Path) -> None:
  session = SessionRecording(base_rrd)
  assert session.decoded_shape("cam00") == (64, 64)
  front, rear = session.lens_setups("cam00", 64)
  expected = parse_offset_v3(synthetic_insv().offset_v3)
  scale = 2 * 64 / expected.front.frame_width
  assert front.params.fx == pytest.approx(expected.front.fx * scale)
  assert front.params.cx == pytest.approx(expected.front.cx * scale)
  assert front.params.xi == pytest.approx(expected.front.xi)
  assert front.params.k1 == pytest.approx(expected.front.k1)
  half_raw = expected.rear.frame_width / 2.0
  assert rear.params.cx == pytest.approx((expected.rear.cx - half_raw) * scale)
  # Transform3D stores float32, so the round-trip is exact to ~1e-7 only.
  np.testing.assert_allclose(
    front.body_T_lens.matrix(), expected.body_T_front_lens.matrix(), atol=1e-6
  )
  np.testing.assert_allclose(
    rear.body_T_lens.matrix(), expected.body_T_rear_lens.matrix(), atol=1e-6
  )


def test_imu_comes_back_in_si_without_reconversion(base_rrd: Path) -> None:
  """The base stores gyro in rad/s already."""
  session = SessionRecording(base_rrd)
  imu = session.imu("cam00")
  assert imu.t[0] == pytest.approx(-0.5, abs=1e-9)
  np.testing.assert_allclose(imu.gyro[0], [np.pi / 2, 0.0, 0.0], atol=1e-12)
  np.testing.assert_allclose(imu.accel[0], [0.0, 0.0, 9.81], atol=1e-12)
  assert np.all(np.diff(imu.t) >= 0)


def test_imu_calibration_roundtrips(base_rrd: Path) -> None:
  session = SessionRecording(base_rrd)
  insv = synthetic_insv()
  numbers = insv.gyro_calib_numbers
  expected = gtsam.imuBias.ConstantBias(numbers[3:6], numbers[0:3])
  bias = session.factory_bias("cam00")
  np.testing.assert_allclose(
    np.asarray(bias.accelerometer()), np.asarray(expected.accelerometer()), atol=1e-12
  )
  np.testing.assert_allclose(
    np.asarray(bias.gyroscope()), np.asarray(expected.gyroscope()), atol=1e-12
  )
  np.testing.assert_allclose(session.body_R_imu("cam00"), BODY_R_IMU, atol=1e-12)
  assert session.rolling_shutter_readout("cam00") == pytest.approx(0.0157)


def _value(rrd: Path, entity: str, component: str) -> float:
  """One static scalar off an entity."""
  row = SessionRecording(rrd)._entity_row(entity.lstrip("/"))
  return float(np.asarray(row[component]).reshape(-1)[0])


def test_capture_telemetry_reaches_the_base(base_rrd: Path) -> None:
  """Identity, clock shift, exposure and AE reports all ride the base recording."""
  rows: dict[str, int] = {}
  columns: dict[str, set[str]] = {}
  for chunk in RrdReader(str(base_rrd)).store().stream().to_chunks():
    e = chunk.entity_path
    rows[e] = rows.get(e, 0) + chunk.num_rows
    batch_names = chunk.to_record_batch().schema.names
    columns.setdefault(e, set()).update(
      n.split(":")[-1] for n in batch_names if n not in chunk.timeline_names
    )
  cap = f"/{schema.camera('cam00')}"
  assert {"serial_number", "camera_type", "fw_version", "photo_rot"} <= columns[cap]
  assert {"source", "clock_shift", "clock_confidence"} <= columns[cap]
  # The shift is the one the streams carry.
  assert _value(base_rrd, cap, "clock_shift") == pytest.approx(-0.25)
  row = SessionRecording(base_rrd)._entity_row(cap.lstrip("/"))
  assert str(np.asarray(row["source"]).reshape(-1)[0]) == "a_00_.insv"
  assert rows[f"/{schema.exposure('cam00')}"] == 3
  assert rows[f"/{schema.autoexposure('cam00')}"] == 2
  assert "iso_value" in columns[f"/{schema.autoexposure('cam00')}"]
