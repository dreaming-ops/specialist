"""Synthetic finger angles round-trip through the 1-D scan and the C++ finger factor."""

from dataclasses import astuple

import gtsam
import numpy as np
from specialist.vi_ba._mei_factor_cpp import make_mei_seated_finger_factor_cpp

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.finger_kinematics import build_finger_reproj, recover_gripper_tags
from specialist.vi_ba.symbols import BASE_TAG_CHAR, JOINT_ANGLE_CHAR, SEAT_CHAR
from specialist.vi_ba.tag_detection import TagDetection
from specialist.vi_ba.tag_discovery import GripperTags
from specialist.vi_ba.urdf_joints import FINGER_JOINTS, RevoluteJoint
from tests.mei_fixtures import (
  WIDE_CALIBRATED_RADIUS,
  synthetic_tag_detection,
  wide_mei_params,
)

_PARAMS = wide_mei_params()
_TAG_SIZE = 0.05
_FINGER_TAG_ID = 2004
# Joint 0.25 m in front, tag tilted off-frontal; joint_T_tag0 is the known CAD pose.
_JOINT = gtsam.Pose3(gtsam.Rot3.Rx(2.7), gtsam.Point3(0, 0, 0.25))
_JOINT_T_TAG0 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.02, 0.03, 0.0))


def _finger_pose(th: float) -> gtsam.Pose3:
  return _JOINT.compose(gtsam.Pose3(gtsam.Rot3.Rz(th), gtsam.Point3(0, 0, 0))).compose(
    _JOINT_T_TAG0
  )


def _detect(lens_T_tag: gtsam.Pose3, tag_id: int) -> TagDetection:
  det = synthetic_tag_detection(lens_T_tag, tag_id, _TAG_SIZE, _PARAMS)
  assert det is not None
  return det


def test_seeds_theta_and_reprojects() -> None:
  lens = LensSetup(
    body_T_lens=gtsam.Pose3(), params=_PARAMS, calibrated_radius=WIDE_CALIBRATED_RADIUS
  )  # lens == body
  joint = RevoluteJoint(
    base_T_joint=_JOINT, lower=-0.7, upper=0.7, joint_T_tag0=_JOINT_T_TAG0
  )
  thetas = [-0.1, 0.0, 0.2, 0.4]
  tags_per_kf: list[list[TagDetection]] = []
  for th in thetas:
    tags_per_kf.append([_detect(_finger_pose(th), _FINGER_TAG_ID)])

  fr = build_finger_reproj(
    joint, tags_per_kf, lens, _FINGER_TAG_ID, gtsam.Pose3(), _TAG_SIZE
  )
  assert fr is not None
  assert set(fr.observations) == set(range(len(thetas)))

  for obs, th in zip(fr.observations.values(), thetas, strict=True):
    assert abs(obs.theta_seed - th) < 0.05

  # The synthetic tags use lens == body, so body_T_base_tag == I.
  noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
  btag_key = gtsam.symbol(BASE_TAG_CHAR, 0)
  seat_key = gtsam.symbol(SEAT_CHAR, 0)
  for kf, th in enumerate(thetas):
    vals = gtsam.Values()
    tkey = gtsam.symbol(JOINT_ANGLE_CHAR, kf)
    vals.insert(tkey, float(th))
    vals.insert(btag_key, gtsam.Pose3())
    vals.insert(seat_key, lens.body_T_lens.rotation())
    for j in range(4):
      f = make_mei_seated_finger_factor_cpp(
        tkey,
        btag_key,
        seat_key,
        fr.observations[kf].pixels[j],
        noise,
        lens.body_T_lens,
        fr.base_T_joint,
        gtsam.Point3(*fr.joint_t_corners[j]),
        np.array(astuple(_PARAMS)),
        fr.observations[kf].obs_bearing,
      )
      assert np.linalg.norm(f.unwhitenedError(vals)) < 0.5, (kf, j)


def test_batched_theta_seeds_are_the_per_sighting_scans() -> None:
  from specialist.vi_ba.finger_kinematics import _seed_thetas, _theta_scan

  rng = np.random.default_rng(23)
  lens = LensSetup(
    body_T_lens=gtsam.Pose3(
      gtsam.Rot3.Expmap(np.array([0.1, -0.2, 0.05])), gtsam.Point3(0.02, 0.0, -0.01)
    ),
    params=wide_mei_params(),
    calibrated_radius=WIDE_CALIBRATED_RADIUS,
  )
  body_T_joint = gtsam.Pose3(
    gtsam.Rot3.Expmap(np.array([0.0, 0.3, 0.0])), gtsam.Point3(0.0, 0.05, 0.3)
  )
  joint_t_corners = rng.uniform(-0.02, 0.02, (4, 3))
  thetas, projected = _theta_scan(joint_t_corners, body_T_joint, lens, -0.5, 1.2)
  pixels = projected[rng.integers(0, len(thetas), 30)] + rng.normal(
    0.0, 2.0, (30, 4, 2)
  )
  pixels[7] = np.nan
  got = _seed_thetas(pixels, joint_t_corners, body_T_joint, lens, -0.5, 1.2)
  for i in range(30):
    errors = np.sum((projected - pixels[i]) ** 2, axis=(1, 2))
    want = thetas[int(np.argmin(np.where(np.isnan(errors), np.inf, errors)))]
    assert got[i] == want


def test_recover_gripper_tags_keeps_finger_tag_zero() -> None:
  """A finger tag with AprilTag id 0 still yields its angles and reprojection inputs."""
  lens = LensSetup(
    body_T_lens=gtsam.Pose3(), params=_PARAMS, calibrated_radius=WIDE_CALIBRATED_RADIUS
  )  # lens == body
  base_tag = gtsam.Pose3(gtsam.Rot3.Rx(2.9), gtsam.Point3(0.0, 0.0, 0.3))
  joint = RevoluteJoint(
    base_T_joint=base_tag.between(_JOINT),
    lower=-0.7,
    upper=0.7,
    joint_T_tag0=_JOINT_T_TAG0,
  )
  thetas = [-0.1, 0.0, 0.2, 0.4]
  tags_per_kf: list[list[TagDetection]] = []
  for th in thetas:
    tags_per_kf.append([_detect(base_tag, 2000), _detect(_finger_pose(th), 0)])

  _, finger_reproj, frame_angles = recover_gripper_tags(
    GripperTags(2000, (0, None)),
    {FINGER_JOINTS[0]: joint},
    tags_per_kf,
    lens,
    _TAG_SIZE,
    len(thetas),
    tags_per_kf,
    np.arange(len(thetas), dtype=np.float64),
  )
  assert set(finger_reproj) == {FINGER_JOINTS[0]}
  _, angles = frame_angles[FINGER_JOINTS[0]]
  np.testing.assert_allclose(angles, thetas, atol=0.05)
