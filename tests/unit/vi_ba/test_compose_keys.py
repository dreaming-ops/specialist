"""``CameraKeys`` gives each camera a disjoint slice of the integer-key space."""

import gtsam
import pytest

from specialist.vi_ba.compose import _MAX_LANDMARKS_PER_CAMERA, CameraKeys
from specialist.vi_ba.symbols import LANDMARK_CHAR, MAX_KEYFRAMES_PER_CAMERA, POSE_CHAR


def test_camera_keys_no_collision_across_cameras() -> None:
  keys = CameraKeys(["alice", "bob"])
  assert keys.pose_key("alice", 0) != keys.pose_key("bob", 0)
  assert keys.pose_key("alice", 0) != keys.velocity_key("alice", 0)
  assert keys.pose_key("alice", 0) != keys.bias_key("alice", 0)
  assert keys.front_landmark_key("alice", 0) != keys.front_landmark_key("bob", 0)
  assert keys.seat_key("alice") != keys.seat_key("bob")
  assert keys.seat_key("alice") != keys.base_tag_key("alice")


def test_camera_keys_duplicate_slots_raises() -> None:
  with pytest.raises(ValueError):
    CameraKeys(["alice", "alice"])


def test_a_track_id_that_outgrows_its_slice_is_refused() -> None:
  """Ids grow monotonically, so a long capture can reach the next camera's slice."""
  keys = CameraKeys(["cam00", "cam01"])
  assert keys.front_landmark_key("cam00", _MAX_LANDMARKS_PER_CAMERA - 1)
  overflowed = int(gtsam.symbol(LANDMARK_CHAR, _MAX_LANDMARKS_PER_CAMERA + 5))
  assert overflowed == keys.front_landmark_key("cam01", 5)
  for track_id in (_MAX_LANDMARKS_PER_CAMERA, _MAX_LANDMARKS_PER_CAMERA + 1, -1):
    with pytest.raises(ValueError, match="landmark slice"):
      keys.front_landmark_key("cam00", track_id)


def test_a_keyframe_that_outgrows_its_slice_is_refused() -> None:
  """A keyframe past the slice width would take the next camera's pose keys."""
  keys = CameraKeys(["cam00", "cam01"])
  assert keys.pose_key("cam00", MAX_KEYFRAMES_PER_CAMERA - 1)
  overflowed = int(gtsam.symbol(POSE_CHAR, MAX_KEYFRAMES_PER_CAMERA + 5))
  assert overflowed == keys.pose_key("cam01", 5)
  for kf in (MAX_KEYFRAMES_PER_CAMERA, MAX_KEYFRAMES_PER_CAMERA + 1, -1):
    for key_fn in (keys.pose_key, keys.velocity_key, keys.bias_key):
      with pytest.raises(ValueError, match="keyframe slice"):
        key_fn("cam00", kf)


def test_a_joint_angle_keyframe_that_outgrows_its_slice_is_refused() -> None:
  """A keyframe past the slice width would take the other finger's angle keys."""
  keys = CameraKeys(["cam00", "cam01"])
  assert keys.theta_key("cam00", 0, MAX_KEYFRAMES_PER_CAMERA - 1)
  for kf in (MAX_KEYFRAMES_PER_CAMERA, MAX_KEYFRAMES_PER_CAMERA + 1, -1):
    with pytest.raises(ValueError, match="joint-angle slice"):
      keys.theta_key("cam00", 0, kf)
