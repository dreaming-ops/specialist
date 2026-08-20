"""Finger-tag kinematics: each tag rides a +z revolute joint off the base tag."""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import mei_project
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.configs import CameraArgs
from specialist.vi_ba.geometry import average_pose, rotate_about_z
from specialist.vi_ba.tag_detection import (
  TagDetection,
  lens_T_tag_per_kf,
  tag_center_bearing,
  tag_corners_in_frame,
)
from specialist.vi_ba.urdf_joints import RevoluteJoint, read_revolute_joints

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "FingerKfObs",
  "FingerReproj",
  "build_finger_reproj",
  "pin_angles_to_keyframes",
  "recover_gripper_tags",
]

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
  lens_R_body = lens_T_body.rotation().matrix()
  lens_t_body = np.asarray(lens_T_body.translation(), dtype=np.float64)
  body_R_joint = body_T_joint.rotation().matrix()
  body_t_joint = np.asarray(body_T_joint.translation(), dtype=np.float64)
  thetas = np.linspace(lower, upper, _THETA_SCAN_SAMPLES)
  rz = rotate_about_z(joint_t_corners, thetas)
  p_lens = (rz @ body_R_joint.T + body_t_joint) @ lens_R_body.T + lens_t_body
  projected = mei_project(p_lens.reshape(-1, 3), lens.params).reshape(
    p_lens.shape[:2] + (2,)
  )
  errors = np.sum((projected - pixels) ** 2, axis=(1, 2))
  # A corner past the fold projects to NaN; an all-NaN scan picks the lower bound.
  return float(thetas[int(np.argmin(np.where(np.isnan(errors), np.inf, errors)))])


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
  joint_t_corners = tag_corners_in_frame(joint.joint_T_tag0, tag_size)
  observations: dict[int, FingerKfObs] = {}
  for i, dets in enumerate(tags_per_frame):
    det = next((d for d in dets if d.tag_id == finger_tag_id), None)
    if det is None:
      continue
    pixels = np.asarray(det.corners, dtype=np.float64)
    bearing = tag_center_bearing(det, lens.params)
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


def pin_angles_to_keyframes(
  frame_times: Float64Array,
  frame_angles: Float64Array,
  keyframe_times: Float64Array,
  keyframe_angles: Float64Array,
) -> Float64Array:
  """Frame angles bent linearly onto keyframe angles; NaN-ended intervals stay raw."""
  pinned = np.asarray(frame_angles, dtype=np.float64).copy()
  raw_at_kf = np.interp(keyframe_times, frame_times, frame_angles)
  offsets = keyframe_angles - raw_at_kf
  for k in range(len(keyframe_times) - 1):
    left, right = offsets[k], offsets[k + 1]
    if not (np.isfinite(left) and np.isfinite(right)):
      continue
    span = keyframe_times[k + 1] - keyframe_times[k]
    inside = (frame_times >= keyframe_times[k]) & (frame_times <= keyframe_times[k + 1])
    fraction = (frame_times[inside] - keyframe_times[k]) / span
    pinned[inside] = frame_angles[inside] + (1 - fraction) * left + fraction * right
  return pinned


def recover_gripper_tags(
  args: CameraArgs,
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  n_keyframes: int,
  kf_rotations: list[Float64Array],
  tags_per_frame: list[list[TagDetection]] | None = None,
  frame_times: Float64Array | None = None,
) -> tuple[
  gtsam.Pose3 | None,
  dict[str, FingerReproj],
  dict[str, tuple[Float64Array, Float64Array]],
]:
  """Base-tag body pose and finger outputs; raises ValueError for an unseen base tag."""
  base_tag_obs = lens_T_tag_per_kf(
    tags_per_kf,
    tag_id=args.base_tag_id,
    tag_size=args.gripper.tag_size,
    params=lens.params,
    kf_rotations=kf_rotations,
  )
  body_T_base_tag: gtsam.Pose3 | None = None
  if base_tag_obs:
    body_T_base_tag = average_pose(
      [lens.body_T_lens.compose(p) for p in base_tag_obs.values()], "mean"
    )
  if args.base_tag_id > 0 and body_T_base_tag is None:
    raise ValueError(
      f"base tag {args.base_tag_id} was configured but never detected in the rear "
      f"lens across {n_keyframes} keyframes — its base-tag, finger, and cross-tracker "
      f"factors would all be silently disabled. Wrong id, or the tag out of the "
      f"rear-lens field of view?"
    )

  # URDF joints sit in the base frame; body_T_base_tag carries them to the body.
  joints = read_revolute_joints(args.urdf_path) if args.urdf_path is not None else {}
  finger_reproj: dict[str, FingerReproj] = {}
  frame_angles: dict[str, tuple[Float64Array, Float64Array]] = {}
  for joint_name, finger_id in (
    ("left_finger", args.left_finger_tag_id),
    ("right_finger", args.right_finger_tag_id),
  ):
    if finger_id <= 0 or args.urdf_path is None:
      continue
    joint = joints.get(joint_name)
    if joint is None or joint.joint_T_tag0 is None:
      raise ValueError(
        f"finger tag {finger_id} was configured for '{joint_name}', but "
        f"{args.urdf_path} has no revolute joint '{joint_name}' carrying an "
        f"AprilTag frame (revolute joints present: {sorted(joints)}). Wrong URDF?"
      )
    if body_T_base_tag is None:
      continue
    reproj = build_finger_reproj(
      joint,
      tags_per_kf[:n_keyframes],
      lens,
      finger_id,
      body_T_base_tag,
      args.gripper.tag_size,
    )
    if reproj is None:
      print(
        f"  [warn] [gripper] finger tag {finger_id} ({joint_name}) never detected "
        f"in the rear keyframes; its angle stays unsolved"
      )
    else:
      finger_reproj[joint_name] = reproj
    if tags_per_frame is None or frame_times is None:
      continue
    framed = build_finger_reproj(
      joint, tags_per_frame, lens, finger_id, body_T_base_tag, args.gripper.tag_size
    )
    if framed is not None:
      frame_angles[joint_name] = (
        frame_times[list(framed.observations)],
        np.array([o.theta_seed for o in framed.observations.values()]),
      )
  return body_T_base_tag, finger_reproj, frame_angles
