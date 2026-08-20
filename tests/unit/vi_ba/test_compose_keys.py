"""Each tracker and lens owns a disjoint slice of the integer-key space."""

from __future__ import annotations

import gtsam
import pytest

from specialist.vi_ba.compose import (
  _MAX_LANDMARKS_PER_CAMERA,
  LANDMARK_CHAR,
  CameraKeys,
)


def test_camera_keys_no_collision_across_cameras() -> None:
  keys = CameraKeys(["alice", "bob"])
  assert keys.pose_key("alice", 0) != keys.pose_key("bob", 0)
  assert keys.pose_key("alice", 0) != keys.velocity_key("alice", 0)
  assert keys.pose_key("alice", 0) != keys.bias_key("alice", 0)
  assert keys.front_landmark_key("alice", 0) != keys.rear_landmark_key("alice", 0)
  assert keys.front_landmark_key("alice", 0) != keys.front_landmark_key("bob", 0)


def test_camera_keys_duplicate_slots_raises() -> None:
  with pytest.raises(ValueError):
    CameraKeys(["alice", "alice"])


def test_a_track_id_that_outgrows_its_slice_is_refused() -> None:
  """Ids grow monotonically, so a long capture can reach the rear lens's slot."""
  keys = CameraKeys(["cam00", "cam01"])
  assert keys.front_landmark_key("cam00", _MAX_LANDMARKS_PER_CAMERA - 1)
  overflowed = int(gtsam.symbol(LANDMARK_CHAR, _MAX_LANDMARKS_PER_CAMERA + 5))
  assert overflowed == keys.rear_landmark_key("cam00", 5)
  for track_id in (_MAX_LANDMARKS_PER_CAMERA, _MAX_LANDMARKS_PER_CAMERA + 1, -1):
    with pytest.raises(ValueError, match="landmark slot"):
      keys.front_landmark_key("cam00", track_id)
