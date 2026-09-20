from types import SimpleNamespace

import gtsam
import numpy as np

from specialist.vi_ba.compose import CameraKeys
from specialist.vi_ba.reference_solve import adopt_ba_refined_estimates
from specialist.vi_ba.urdf_joints import RevoluteJoint


def test_solved_angle_outside_the_joint_limits_is_dropped() -> None:
  keys = CameraKeys(["cam00"])
  values = gtsam.Values()
  for kf, theta in enumerate((0.1, 1.23, -0.8)):
    values.insert(keys.theta_key("cam00", 1, kf), theta)
  camera = SimpleNamespace(slot="cam00", n_keyframes=3, joint_angles={})
  joint = RevoluteJoint(gtsam.Pose3(), -0.698, 0.349, None)
  joints = {"left_finger": joint, "right_finger": joint}
  adopt_ba_refined_estimates([camera], SimpleNamespace(values=values), keys, joints)
  np.testing.assert_array_equal(
    camera.joint_angles["right_finger"], [0.1, np.nan, np.nan]
  )
