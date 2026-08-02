"""Contract tests for the exported JSON, whose only consumer is out of tree."""

from __future__ import annotations

import json
from pathlib import Path

import gtsam
import numpy as np

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.configs import AnchorCfg, Args, WindowCfg
from specialist.vi_ba.export import build_raw_export
from specialist.vi_ba.single_camera import PreparedCamera
from tests.mei_fixtures import narrow_mei_params

_REFERENCE_TAG = 1
_DECODED_WIDTH = 2880


def _pose(x: float, y: float, z: float) -> gtsam.Pose3:
  return gtsam.Pose3(gtsam.Rot3.Rodrigues(0.1, 0.2, 0.3), gtsam.Point3(x, y, z))


def _lens() -> LensSetup:
  return LensSetup(body_T_lens=_pose(0.05, 0.0, 0.0), params=narrow_mei_params())


def _tracker(name: str, n: int = 3) -> PreparedCamera:
  return PreparedCamera(
    name=name,
    body_poses=[_pose(0.1 * k, 0.0, 0.4) for k in range(n)],
    velocities=[np.full(3, 0.1 * k) for k in range(n)],
    biases=[gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)) for _ in range(n)],
    preintegrated=[],
    front_lens=_lens(),
    rear_lens=_lens(),
    front_tracks_tri={7: {0: np.array([1.0, 2.0])}},
    rear_tracks_tri={9: {1: np.array([3.0, 4.0])}},
    front_points={7: np.array([0.1, 0.2, 1.0])},
    rear_points={9: np.array([0.3, 0.4, 1.5])},
    decoded_width=_DECODED_WIDTH,
    anchor_obs_by_tag={_REFERENCE_TAG: {0: {"front": _pose(0.0, 0.0, 1.0)}}},
    anchor_locals_by_tag={_REFERENCE_TAG: _pose(0.0, 0.0, 1.0)},
    body_T_base_tag=None,
    joint_angles={},
    keyframe_times=np.arange(n, dtype=np.float64),
    front_frames=[],
    rear_frames=[],
    front_keypoints=[],
    rear_keypoints=[],
    front_tags_per_kf=[[] for _ in range(n)],
    rear_tags_per_kf=[[] for _ in range(n)],
    front_rejected={},
    rear_rejected={},
    rear_static_mask_rects=[],
    finger_reproj={},
    left_finger_tag_id=0,
    right_finger_tag_id=0,
  )


def _raw() -> dict[str, object]:
  cameras = [_tracker("a"), _tracker("b")]
  return build_raw_export(
    cameras,
    {t.name: t.body_poses for t in cameras},
    {t.name: (t.front_points, t.rear_points) for t in cameras},
    {_REFERENCE_TAG: _pose(0.0, 0.0, 1.0), 5: _pose(0.5, 0.0, 1.0)},
    {"a": 0.0, "b": 1.959},
    _REFERENCE_TAG,
    Args(
      input_insvs=(Path("a.insv"), Path("b.insv")),
      window=WindowCfg(duration=None),
      anchor=AnchorCfg(tag_ids=(_REFERENCE_TAG, 5)),
      base_tag_ids=(2000, 2003),
    ),
  )


def test_top_level_keys_are_exactly_the_documented_set() -> None:
  """Readers index the keys by name, so adding or renaming one breaks them."""
  assert set(_raw()) == {
    "anchor_obs_by_tracker",
    "anchor_poses_world",
    "anchor_tag_id",
    "anchor_tag_size",
    "audio_offset_by_tracker",
    "body_T_base_tag_by_tracker",
    "body_poses_by_tracker",
    "gripper_keypoint_offset",
    "highrate_by_tracker",
    "imu_bias_by_tracker",
    "joint_angles_highrate_by_tracker",
    "joint_angles_by_tracker",
    "kf_times_by_tracker",
    "velocities_by_tracker",
    "landmarks_by_tracker",
    "lens_by_tracker",
    "ref_anchor_pose_world",
    "tag_detections_by_tracker",
    "tracks_by_tracker",
  }


def test_poses_round_trip_through_the_json_encoding() -> None:
  """A pose encodes as a row-major 3x3 rotation plus a 3-vector."""
  raw = _raw()
  original = _tracker("a").body_poses[1]
  encoded = raw["body_poses_by_tracker"]["a"][1]  # type: ignore[index]
  assert len(encoded["R"]) == 3 and all(len(row) == 3 for row in encoded["R"])
  assert len(encoded["t"]) == 3
  rebuilt = gtsam.Pose3(gtsam.Rot3(np.array(encoded["R"])), gtsam.Point3(*encoded["t"]))
  assert np.allclose(rebuilt.matrix(), original.matrix(), atol=1e-12)


def test_reference_anchor_is_carried_both_ways_and_agrees() -> None:
  """``ref_anchor_pose_world`` matches the reference entry of ``anchor_poses_world``."""
  raw = _raw()
  assert raw["anchor_tag_id"] == _REFERENCE_TAG
  by_tag = raw["anchor_poses_world"]  # type: ignore[index]
  assert str(_REFERENCE_TAG) in by_tag
  assert by_tag[str(_REFERENCE_TAG)] == raw["ref_anchor_pose_world"]


def test_frame_shape_follows_the_decoded_width() -> None:
  """Export pixel coordinates are in decoded-frame units."""
  lens = _raw()["lens_by_tracker"]["a"]  # type: ignore[index]
  for side in ("front", "rear"):
    assert lens[side]["frame_shape"] == [_DECODED_WIDTH, _DECODED_WIDTH]


def test_integer_keyed_maps_are_stringified() -> None:
  """Each side's landmark and track maps export non-empty, all-string keys."""
  raw = _raw()
  for section in ("landmarks_by_tracker", "tracks_by_tracker"):
    for side in ("front", "rear"):
      keys = raw[section]["a"][side].keys()  # type: ignore[index]
      assert keys and all(isinstance(k, str) for k in keys)


def test_export_is_serializable_without_a_fallback_encoder() -> None:
  """dumps() without ``default=`` proves no numpy scalar or Path leaked in."""
  text = json.dumps(_raw())
  assert json.loads(text)["anchor_tag_id"] == _REFERENCE_TAG


def test_inertial_state_round_trips_per_keyframe() -> None:
  """The export holds each keyframe's velocity and gyro and accel bias as 3-vectors."""
  raw = _raw()
  for name in ("a", "b"):
    vels = raw["velocities_by_tracker"][name]
    assert len(vels) == 3 and all(len(v) == 3 for v in vels)
    biases = raw["imu_bias_by_tracker"][name]
    assert len(biases) == 3
    for b in biases:
      assert set(b) == {"gyro", "accel"}
      assert len(b["gyro"]) == 3 and len(b["accel"]) == 3


def test_highrate_trajectory_is_absent_without_imu() -> None:
  """A tracker without IMU has no ``highrate_by_tracker`` entry in the JSON export."""
  raw = _raw()
  assert raw["highrate_by_tracker"] == {}


def test_finger_highrate_series_is_absent_without_finger_angles() -> None:
  raw = _raw()
  assert raw["joint_angles_highrate_by_tracker"] == {}
