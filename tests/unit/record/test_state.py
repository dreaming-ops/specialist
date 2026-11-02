"""The state layer of a synthetic base and estimate, read back as a trainer sees it."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from rerun.chunk import RrdReader
from scipy.spatial.transform import Rotation
from scripts.state import Args, main
from specialist_camera import to_image
from specialist_dataset import schema
from specialist_dataset.frames import LensVideo, decode_samples
from specialist_dataset.layer import encode
from specialist_dataset.read import Recording

from specialist.record.base import (
  calibration_chunks,
  imu_chunks,
  write_base_recording,
)
from specialist.record.video import normalize_h264, video_chunks
from tests.record_fixtures import (
  BASE_TAG,
  SPIN,
  TILT,
  camera_estimate,
  encode_clip,
  synthetic_insv,
  urdf_file,
  write_run,
)

_FPS = 30
# Frames 0..11; the estimate fixture's keyframes sit on frames 0, 3, 6, 9.
_FRAMES = 12
_WINDOW = (-1.0, 31.0)
_SCALE = 4
_F_ROW = 1000.0
# The second camera's clock sits a sub-frame off the first's, as two real captures do.
_SHIFT = {"cam00": 0.0, "cam01": -0.0043}
# One camera slower than the lattice and one faster, as two real IMUs are.
_RATE = {"cam00": 997.0, "cam01": 1003.0}
# In the IMU span's last tenth, so a zero margin leaves the keyframe span untouched.
_STRIKE = 29.0


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
  """A layer-first root: a two-camera base with rear lenses, and its estimate."""
  root = tmp_path_factory.mktemp("dataset")
  imu_t = np.arange(-1.0, 30.0, 0.005)
  accel = np.zeros((len(imu_t), 3))
  accel[:, 2] = 9.81 + 0.05 * np.sin(20.0 * imu_t)
  accel[np.abs(imu_t - _STRIKE) < 0.003, 2] += 200.0
  insv = replace(
    synthetic_insv(),
    imu_timestamps=imu_t,
    gyro=np.zeros((len(imu_t), 3)),
    accel=accel,
  )
  clip = normalize_h264(
    encode_clip(root / "clip.mp4", _FRAMES, rate=_FPS, gop=4), root / "clean.mp4"
  )
  streams, sidecars = [], []
  for slot, shift in _SHIFT.items():
    streams.append(video_chunks(clip, schema.video(slot, "rear"), shift, _WINDOW))
    sidecars += imu_chunks(slot, insv, 0.0, _WINDOW) + calibration_chunks(slot, insv)
  base = schema.layer_rrd(root, schema.BASE, "sess")
  base.parent.mkdir()
  write_base_recording(base, "sess", streams, sidecars)
  cameras = [
    camera_estimate("cam00", hold=0.7, rate=_RATE["cam00"]),
    camera_estimate("cam01", offset=1.0, hold=0.7, rate=_RATE["cam01"]),
  ]
  write_run(root, "sess", cameras, urdf_path=urdf_file(root), frame_period=1.0 / _FPS)
  return root


def _state(dataset: Path, out_dir: Path, **kwargs: object) -> Path:
  base = schema.layer_rrd(dataset, schema.BASE, "sess")
  args = {
    "scale": _SCALE,
    "f_row": _F_ROW,
    "urdf_path": dataset / "robot.urdf",
    **kwargs,
  }
  main(Args(base=base, output_dir=out_dir, **args))  # type: ignore[arg-type]
  return schema.layer_rrd(out_dir, schema.state_layer(_SCALE, _F_ROW), "sess")


def _frames(
  recording: Recording, hand: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
  entity = schema.state_image(hand)
  table = recording.series(f"/{entity}")
  blobs = table.column(schema.selector(entity, "EncodedImage:blob")).to_pylist()
  images = [
    cv2.imdecode(np.asarray(b[0], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    for b in blobs
  ]
  rows = table.column(schema.ROW_TIMELINE).to_numpy()
  captured = table.column(schema.selector(entity, schema.CAPTURE_ROW)).to_numpy()
  return schema.times_ns(table.column(schema.TIMELINE)), rows, captured, images


def _expected_state(offset: float, rate: float) -> dict[int, dict[str, np.ndarray]]:
  """The closed-form state at each millisecond the ``rate`` Hz samples bracket."""
  last = round(0.3 * rate) / rate
  rows = np.arange(0, int(np.floor(last * _F_ROW)) + 1)
  t = rows / _F_ROW
  turn = Rotation.from_rotvec(np.outer(SPIN * t, [0.0, 0.0, 1.0])) * (
    Rotation.from_rotvec([TILT, 0.0, 0.0])
  )
  arm = turn.apply(np.asarray(BASE_TAG.translation()))
  tag = Rotation.from_quat(np.asarray(BASE_TAG.rotation().toQuaternion().coeffs()))
  quats = (turn * tag).as_quat()
  quats[quats[:, 3] < 0.0] *= -1.0
  position = np.stack([offset + t, np.zeros_like(t), np.zeros_like(t)], 1) + arm
  omega = np.tile([0.0, 0.0, SPIN], (len(t), 1))
  lin_vel = np.array([1.0, 0.0, 0.0]) + np.cross(omega, arm)
  return {
    int(k): {
      "t": t[i],
      "position": position[i],
      "quat": quats[i],
      "lin_vel": lin_vel[i],
      "ang_vel": omega[i],
    }
    for i, k in enumerate(rows)
  }


def test_the_layer_holds_each_hand_by_name(dataset: Path, tmp_path: Path) -> None:
  out = _state(dataset, dataset, margin=0.0)
  base = Recording(schema.layer_rrd(dataset, schema.BASE, "sess"))
  with Recording(out) as recording:
    # The anchor rule makes cam00 the left hand; each hand keeps its own capture times.
    for hand, slot in (("left", "cam00"), ("right", "cam01")):
      assert recording.row(schema.state_hand(hand))["slot"] == [slot]
      times, rows, captured, images = _frames(recording, hand)
      frame_times = np.arange(1, 9) / _FPS + _SHIFT[slot]
      np.testing.assert_allclose(times / 1e9, frame_times, atol=1e-6)
      np.testing.assert_array_equal(rows, [33, 67, 100, 133, 167, 200, 233, 267])
      np.testing.assert_array_equal(captured, np.rint(frame_times * _F_ROW))
      # The hands sit side by side in one plane, so a card shows the pair.
      placed = np.asarray(
        recording.row(schema.state_image(hand))[schema.POSE_TRANSLATION]
      ).reshape(3)
      width = images[0].shape[1]
      np.testing.assert_array_equal(
        placed, [0.0 if hand == "left" else width, 0.0, 0.0]
      )
      video = LensVideo(base, slot, "rear", tmp_path)
      samples = np.arange(1, 9)
      for image, (_, frame) in zip(images, decode_samples(video, samples), strict=True):
        blob = encode(to_image(frame.to_ndarray(format="rgb24"), _SCALE))
        stored = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(image, stored[:, :, ::-1])
    assert np.array_equal(_frames(recording, "left")[1], _frames(recording, "right")[1])
    assert not np.array_equal(
      _frames(recording, "left")[2], _frames(recording, "right")[2]
    )
    account = {
      k: np.asarray(v).reshape(-1)[0]
      for k, v in recording.property(schema.STATE).items()
    }
    assert account["config"] == schema.state_layer(_SCALE, _F_ROW)
    assert account["f_row"] == _F_ROW
    assert account["duration"] == pytest.approx(0.3, abs=1e-3)
    assert account["kept"] == pytest.approx(0.3 / (11 / 30 - 0.0043), abs=1e-3)
    assert int(account["converted_at"]) > 0
    assert (
      account["estimated_at"]
      == np.asarray(
        Recording(schema.layer_rrd(dataset, schema.ESTIMATE, "sess")).property(
          schema.ESTIMATE
        )["estimated_at"]
      ).reshape(-1)[0]
    )
    assert recording.row(schema.STATE) == {}
    assert account["scale"] == _SCALE
    assert account["side"] == "rear"
  with pytest.raises(SystemExit, match="--overwrite"):
    _state(dataset, dataset)


def test_the_state_is_the_base_tag_frame_on_a_dense_row_lattice(
  dataset: Path, tmp_path: Path
) -> None:
  """Each row is the trajectory at its millisecond composed with the tag's mounting."""
  out = _state(dataset, tmp_path, margin=0.0)
  with Recording(out) as recording:
    for hand, slot, offset in (("left", "cam00", 0.0), ("right", "cam01", 1.0)):
      expected = _expected_state(offset, _RATE[slot])
      entity = schema.state_pose(hand)
      table = recording.series(f"/{entity}")
      rows = table.column(schema.ROW_TIMELINE).to_numpy()
      np.testing.assert_array_equal(rows, np.arange(0, 300))
      t, position = recording.values(entity, schema.POSE_TRANSLATION, 3)
      _, quaternion = recording.values(entity, schema.POSE_QUATERNION, 4)
      _, lin_vel = recording.scalars(schema.state_lin_vel(hand), 3)
      _, ang_vel = recording.scalars(schema.state_ang_vel(hand), 3)
      t_open, q = recording.scalars(schema.state_opening(hand), 1)
      want = [expected[int(k)] for k in rows]
      np.testing.assert_allclose(t, rows / _F_ROW, atol=1e-9)
      # Float32 estimate rows set these tolerances; finite differences amplify them.
      np.testing.assert_allclose(position, [w["position"] for w in want], atol=1e-6)
      np.testing.assert_allclose(quaternion, [w["quat"] for w in want], atol=1e-6)
      assert (quaternion[:, 3] >= 0.0).all()
      np.testing.assert_allclose(lin_vel, [w["lin_vel"] for w in want], atol=1e-5)
      np.testing.assert_allclose(ang_vel, [w["ang_vel"] for w in want], atol=1e-4)
      # The opening lands on every row, interpolated from the solved finger angles.
      np.testing.assert_allclose(t_open, t, atol=1e-9)
      np.testing.assert_allclose(q[:, 0], 0.1)


def test_each_hand_places_the_gripper_model(dataset: Path, tmp_path: Path) -> None:
  """The pose places the model's root and the opening turns each finger joint."""
  out = _state(dataset, tmp_path, margin=0.0)
  with Recording(out) as recording:
    for hand, slot in (("left", "cam00"), ("right", "cam01")):
      pose = recording.series(f"/{schema.state_pose(hand)}")
      child = pose.column(
        schema.selector(schema.state_pose(hand), "Transform3D:child_frame")
      )
      assert {f[0] for f in child.to_pylist()} == {f"{slot}_base"}
      # An opening of 0.1 rad turns the left finger +0.1 about z, the right -0.1.
      for joint, turn in (("left_finger", 0.05), ("right_finger", -0.05)):
        entity = schema.state_finger(hand, joint)
        edges = recording.series(f"/{entity}").column(
          schema.selector(entity, "Transform3D:child_frame")
        )
        assert {f[0] for f in edges.to_pylist()} == {f"{slot}_{joint}"}
        _, quaternion = recording.values(entity, schema.POSE_QUATERNION, 4)
        np.testing.assert_allclose(quaternion[:, 2], np.sin(turn), atol=1e-6)


def test_a_frame_whose_capture_row_leaves_the_state_rows_goes_for_both_hands(
  dataset: Path, tmp_path: Path
) -> None:
  out = _state(dataset, tmp_path, margin=0.5985)
  with Recording(out) as recording:
    for hand, slot in (("left", "cam00"), ("right", "cam01")):
      times, rows, captured, _ = _frames(recording, hand)
      np.testing.assert_allclose(
        times / 1e9, np.arange(4, 9) / _FPS + _SHIFT[slot], atol=1e-6
      )
      np.testing.assert_array_equal(rows, [133, 167, 200, 233, 267])
      assert captured.min() >= 99


def test_the_margin_trims_the_window(dataset: Path, tmp_path: Path) -> None:
  out = _state(dataset, tmp_path, margin=0.61)
  with Recording(out) as recording:
    # cam00 rides the left hand and keeps the session clock.
    np.testing.assert_allclose(
      _frames(recording, "left")[0] / 1e9, np.arange(4, 9) / _FPS, atol=1e-6
    )


def test_the_writer_refuses_what_it_cannot_honor(dataset: Path, tmp_path: Path) -> None:
  for message, kwargs in (
    ("does not divide", {"scale": 7}),
    ("window", {"margin": 100.0}),
  ):
    with pytest.raises(SystemExit, match=message):
      _state(dataset, tmp_path, **kwargs)


def test_each_frame_is_a_chunk_of_its_own(dataset: Path, tmp_path: Path) -> None:
  """A random read loads one frame, its image and capture row together."""
  chunks = RrdReader(str(_state(dataset, tmp_path, margin=0.0))).stream().to_chunks()
  images = {schema.state_image(hand) for hand in schema.HANDS}
  frames = [c for c in chunks if c.entity_path.strip("/") in images and not c.is_static]
  assert {c.num_rows for c in frames} == {1}
  for chunk in frames:
    assert {"EncodedImage:blob", schema.CAPTURE_ROW} <= set(
      chunk.to_record_batch().schema.names
    )


def test_the_writer_refuses_an_estimate_solved_with_another_urdf(
  dataset: Path, tmp_path: Path
) -> None:
  other = tmp_path / "other.urdf"
  other.write_text((dataset / "robot.urdf").read_text() + "<!-- edited -->\n")
  with pytest.raises(SystemExit, match="solved cam00 with URDF sha256"):
    _state(dataset, tmp_path, urdf_path=other, margin=0.0)
