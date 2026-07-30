"""Each tracker and lens owns a disjoint slice of the integer-key space."""

from __future__ import annotations

import pytest

from specialist.vi_ba.compose import CameraKeys


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
