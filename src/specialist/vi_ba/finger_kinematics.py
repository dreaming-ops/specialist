"""Finger-tag kinematics: each tag rides a revolute joint off the base tag,
``body_T_tag(theta) = body_T_base_tag · base_T_joint · Rz(theta) · joint_T_tag0``."""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt
from specialist_camera import mei_project
from specialist_dataset.fiducials import tag_corners_in_frame

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.geometry import average_pose, consensus_pose, rotate_about_z
from specialist.vi_ba.tag_detection import (
  TagDetection,
  lens_T_tag_per_kf,
  tag_center_bearings,
)
from specialist.vi_ba.tag_discovery import GripperTags
from specialist.vi_ba.urdf_joints import FINGER_JOINTS, RevoluteJoint

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


def _theta_scan(
  joint_t_corners: Float64Array,
  body_T_joint: gtsam.Pose3,
  lens: LensSetup,
  lower: float,
  upper: float,
) -> tuple[Float64Array, Float64Array]:
  """The scanned angles over ``[lower, upper]`` and the corners' pixels at each."""
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
  return thetas, projected


def _seed_thetas(
  pixels: Float64Array,
  joint_t_corners: Float64Array,
  body_T_joint: gtsam.Pose3,
  lens: LensSetup,
  lower: float,
  upper: float,
) -> Float64Array:
  """Per ``(m, 4, 2)`` sighting, the scanned angle that best reprojects it."""
  thetas, projected = _theta_scan(joint_t_corners, body_T_joint, lens, lower, upper)
  errors = np.sum((projected[None] - pixels[:, None]) ** 2, axis=(2, 3))
  # A corner past the fold projects to NaN; an all-NaN scan picks the lower bound.
  best: Float64Array = thetas[
    np.argmin(np.where(np.isnan(errors), np.inf, errors), axis=1)
  ]
  return best


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
  joint_t_corners = tag_corners_in_frame(joint.joint_T_tag0.matrix(), tag_size)
  seen: list[tuple[int, TagDetection]] = []
  for i, dets in enumerate(tags_per_frame):
    det = next((d for d in dets if d.tag_id == finger_tag_id), None)
    if det is not None:
      seen.append((i, det))
  if not seen:
    return None
  pixels = np.stack([np.asarray(det.corners, dtype=np.float64) for _, det in seen])
  thetas = _seed_thetas(
    pixels, joint_t_corners, body_T_joint_seed, lens, joint.lower, joint.upper
  )
  bearings = tag_center_bearings([det for _, det in seen], lens.params)
  observations = {
    kf: FingerKfObs(theta_seed=float(theta), pixels=px, obs_bearing=bearing)
    for (kf, _), theta, px, bearing in zip(seen, thetas, pixels, bearings, strict=True)
  }
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
  """Frame angles bent linearly per interval through the solved keyframe angles."""
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
  tags: GripperTags,
  joints: dict[str, RevoluteJoint],
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  tag_size: float,
  n_keyframes: int,
  tags_per_frame: list[list[TagDetection]],
  frame_times: Float64Array,
) -> tuple[
  gtsam.Pose3,
  dict[str, FingerReproj],
  dict[str, tuple[Float64Array, Float64Array]],
]:
  """The base tag's body pose and, per finger, reprojection inputs and frame angles."""
  base_tag_obs = lens_T_tag_per_kf(
    tags_per_kf[:n_keyframes],
    tag_id=tags.base,
    tag_size=tag_size,
    params=lens.params,
  )
  if not base_tag_obs:
    raise ValueError(
      f"base tag {tags.base} was discovered but never resolved a pose "
      f"across {n_keyframes} rear keyframes — its base-tag, finger, and "
      f"cross-camera factors would all be silently disabled"
    )
  body_T_base_tags = [lens.body_T_lens.compose(p) for p in base_tag_obs.values()]
  body_T_base_tag = consensus_pose(body_T_base_tags)
  if body_T_base_tag is None:
    print("  [warn] [gripper] body_T_base_tag sightings split; seeding on the median")
    body_T_base_tag = average_pose(body_T_base_tags)

  finger_reproj: dict[str, FingerReproj] = {}
  frame_angles: dict[str, tuple[Float64Array, Float64Array]] = {}
  for joint_name, finger_id in zip(FINGER_JOINTS, tags.fingers, strict=True):
    if finger_id is None:
      continue
    joint = joints[joint_name]
    reproj = build_finger_reproj(
      joint, tags_per_kf[:n_keyframes], lens, finger_id, body_T_base_tag, tag_size
    )
    if reproj is None:
      print(
        f"  [warn] [gripper] finger tag {finger_id} ({joint_name}) never detected "
        f"in the rear keyframes; its angle stays unsolved"
      )
    else:
      finger_reproj[joint_name] = reproj
    framed = build_finger_reproj(
      joint, tags_per_frame, lens, finger_id, body_T_base_tag, tag_size
    )
    if framed is not None:
      frame_angles[joint_name] = (
        frame_times[list(framed.observations)],
        np.array([o.theta_seed for o in framed.observations.values()]),
      )
  return body_T_base_tag, finger_reproj, frame_angles
