"""Finger-tag kinematics: each tag rides a +z revolute joint off the base tag."""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import mei_project, mei_unproject
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.tag_detection import TagDetection, tag_corners_local
from specialist.vi_ba.urdf_joints import RevoluteJoint

Float64Array = npt.NDArray[np.float64]

_THETA_SCAN_SAMPLES = 48


@dataclass(frozen=True)
class FingerKfObs:
  """One keyframe's finger-tag corner detection, ready for a reprojection factor."""

  theta_seed: float
  pixels: Float64Array  # (4, 2) detected corners, order [BL, BR, TR, TL]
  obs_bearing: Float64Array  # (3,) tag direction in the lens frame (cheirality)


@dataclass(frozen=True)
class FingerReproj:
  """Everything the joint graph needs to constrain one finger's joint angle."""

  base_T_joint: gtsam.Pose3  # URDF joint origin in the base frame; its z is the axis
  joint_t_corners: Float64Array  # (4, 3) tag corners in the joint frame at theta=0
  observations: dict[int, FingerKfObs]


def _seed_theta(
  pixels: Float64Array,
  joint_t_corners: Float64Array,
  body_T_joint: gtsam.Pose3,
  lens: LensSetup,
  lower: float,
  upper: float,
) -> float:
  """Joint angle that best reprojects the corners, by a 1-D scan of the joint range."""
  lens_T_body = lens.body_T_lens.inverse()
  r_lb = lens_T_body.rotation().matrix()
  t_lb = np.asarray(lens_T_body.translation(), dtype=np.float64)
  r_bj = body_T_joint.rotation().matrix()
  t_bj = np.asarray(body_T_joint.translation(), dtype=np.float64)
  cx, cy, cz = joint_t_corners[:, 0], joint_t_corners[:, 1], joint_t_corners[:, 2]
  best_theta, best_err = lower, np.inf
  for theta in np.linspace(lower, upper, _THETA_SCAN_SAMPLES):
    c, s = np.cos(theta), np.sin(theta)
    rz = np.column_stack([c * cx - s * cy, s * cx + c * cy, cz])  # Rz(theta) . c
    p_lens = (rz @ r_bj.T + t_bj) @ r_lb.T + t_lb
    err = float(np.sum((mei_project(p_lens, lens.params) - pixels) ** 2))
    if err < best_err:
      best_err, best_theta = err, float(theta)
  return best_theta


def build_finger_reproj(
  joint: RevoluteJoint,
  tags_per_frame: list[list[TagDetection]],
  lens: LensSetup,
  finger_tag_id: int,
  body_T_base_tag: gtsam.Pose3,
  tag_size: float,
) -> FingerReproj | None:
  """One finger's reprojection inputs per frame that sees its tag, or ``None`` when none does."""
  assert joint.joint_T_tag0 is not None
  base_T_joint = joint.base_T_joint
  body_T_joint_seed = body_T_base_tag.compose(base_T_joint)
  joint_t_corners = np.array(
    [
      joint.joint_T_tag0.transformFrom(gtsam.Point3(*c))
      for c in tag_corners_local(tag_size)
    ],
    dtype=np.float64,
  )
  observations: dict[int, FingerKfObs] = {}
  for i, dets in enumerate(tags_per_frame):
    det = next((d for d in dets if d.tag_id == finger_tag_id), None)
    if det is None:
      continue
    pixels = np.asarray(det.corners, dtype=np.float64)
    bearing = mei_unproject(
      np.asarray(det.center, dtype=np.float64).reshape(1, 2), lens.params
    )[0]
    theta = _seed_theta(
      pixels, joint_t_corners, body_T_joint_seed, lens, joint.lower, joint.upper
    )
    observations[i] = FingerKfObs(theta_seed=theta, pixels=pixels, obs_bearing=bearing)
  if not observations:
    return None
  return FingerReproj(
    base_T_joint=base_T_joint,
    joint_t_corners=joint_t_corners,
    observations=observations,
  )
