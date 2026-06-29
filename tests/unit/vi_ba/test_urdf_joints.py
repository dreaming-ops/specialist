"""The URDF reader recovers the finger joints and their AprilTag frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from specialist.vi_ba.urdf_joints import read_revolute_joints


def test_reads_finger_joints_and_tag_frames() -> None:
  description = pytest.importorskip("specialist_description")
  joints = read_revolute_joints(Path(description.get_urdf_path("specialist_handheld")))
  assert set(joints) == {"left_finger", "right_finger"}

  left = joints["left_finger"]
  np.testing.assert_allclose(
    left.base_T_joint.translation(), [0, 0.012, -0.008], atol=1e-3
  )
  assert left.lower < left.upper
  # The finger tag pose comes straight from the URDF tag link.
  assert left.joint_T_tag0 is not None
  np.testing.assert_allclose(
    left.joint_T_tag0.translation(), [0.04, 0.023, 0.0], atol=1e-3
  )

  right = joints["right_finger"]
  assert right.joint_T_tag0 is not None
  np.testing.assert_allclose(
    right.joint_T_tag0.translation(), [0.04, -0.023, 0.0], atol=1e-3
  )
