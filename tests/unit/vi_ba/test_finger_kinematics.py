"""Synthetic finger angles round-trip through the 1-D scan and the C++ finger factor."""

from __future__ import annotations

from dataclasses import astuple

import gtsam
import numpy as np
from specialist.vi_ba._mei_factor_cpp import make_mei_finger_factor_cpp

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.finger_kinematics import build_finger_reproj
from specialist.vi_ba.tag_detection import TagDetection
from specialist.vi_ba.urdf_joints import RevoluteJoint
from tests.mei_fixtures import synthetic_tag_detection, wide_mei_params

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
  lens = LensSetup(body_T_lens=gtsam.Pose3(), params=_PARAMS)  # lens == body
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
  btag_key = gtsam.symbol("G", 0)
  for kf, th in enumerate(thetas):
    vals = gtsam.Values()
    tkey = gtsam.symbol("a", kf)
    vals.insert(tkey, float(th))
    vals.insert(btag_key, gtsam.Pose3())
    for j in range(4):
      f = make_mei_finger_factor_cpp(
        tkey,
        btag_key,
        fr.observations[kf].pixels[j],
        noise,
        lens.body_T_lens,
        fr.base_T_joint,
        gtsam.Point3(*fr.joint_t_corners[j]),
        np.array(astuple(_PARAMS)),
        fr.observations[kf].obs_bearing,
      )
      assert np.linalg.norm(f.unwhitenedError(vals)) < 0.5, (kf, j)
