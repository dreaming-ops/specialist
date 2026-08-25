"""The URDF reader recovers the finger joints and their AprilTag frames."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from specialist.vi_ba.urdf_joints import read_revolute_joints


def test_reads_finger_joints_and_tag_frames(tmp_path: Path) -> None:
  """Origins are in the root frame, and each tag hangs off its joint's child link."""
  path = tmp_path / "gripper.urdf"
  path.write_text(
    '<?xml version="1.0"?><robot name="gripper">'
    '<link name="base"/><link name="left_finger"/><link name="left_finger_tag"/>'
    '<joint name="left_finger" type="revolute">'
    '<parent link="base"/><child link="left_finger"/><axis xyz="0 0 1"/>'
    '<origin xyz="0 0.012 -0.008" rpy="0 0 0"/>'
    '<limit lower="-0.7" upper="0.9" effort="1" velocity="1"/></joint>'
    '<joint name="left_finger_tag" type="fixed">'
    '<parent link="left_finger"/><child link="left_finger_tag"/>'
    '<origin xyz="0.04 0.023 0" rpy="0 0 0"/></joint>'
    "</robot>"
  )
  joints = read_revolute_joints(path)
  assert set(joints) == {"left_finger"}, "a fixed joint is not a revolute one"

  left = joints["left_finger"]
  np.testing.assert_allclose(
    left.base_T_joint.translation(), [0, 0.012, -0.008], atol=1e-9
  )
  assert (left.lower, left.upper) == pytest.approx((-0.7, 0.9))
  assert left.joint_T_tag0 is not None, "the tag frame did not reach the joint"
  np.testing.assert_allclose(
    left.joint_T_tag0.translation(), [0.04, 0.023, 0.0], atol=1e-9
  )


def test_a_joint_that_declares_no_limit_is_free_not_welded(tmp_path: Path) -> None:
  """URDF makes ``<limit>`` optional, so a joint that declares none spans ±π."""
  path = tmp_path / "unlimited.urdf"
  path.write_text(
    '<?xml version="1.0"?><robot name="g">'
    '<link name="base"/><link name="tip"/>'
    '<joint name="j" type="revolute"><parent link="base"/><child link="tip"/>'
    '<axis xyz="0 0 1"/></joint></robot>'
  )
  joint = read_revolute_joints(path)["j"]
  assert joint.lower < joint.upper, "an unlimited joint cannot turn"
  assert joint.lower == pytest.approx(-math.pi)
  assert joint.upper == pytest.approx(math.pi)
