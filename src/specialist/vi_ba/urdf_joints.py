"""Revolute-joint kinematics from the gripper URDF, in the URDF root frame."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import gtsam
import rerun as rr
from specialist_dataset.layer import FINGER_JOINTS

__all__ = ["FINGER_JOINTS", "RevoluteJoint", "read_revolute_joints"]


@dataclass(frozen=True)
class RevoluteJoint:
  """A URDF ``revolute`` joint about +z; its ``<axis>`` and ``<mimic>`` are not read."""

  base_T_joint: gtsam.Pose3
  lower: float
  upper: float
  # The child link's fixed-joint origin to its tag link.
  joint_T_tag0: gtsam.Pose3 | None


def _pose(
  xyz: tuple[float, float, float], rpy: tuple[float, float, float]
) -> gtsam.Pose3:
  """A URDF joint origin as a ``Pose3``, with R = Rz(yaw) Ry(pitch) Rx(roll)."""
  return gtsam.Pose3(gtsam.Rot3.RzRyRx(*rpy), gtsam.Point3(*xyz))


def read_revolute_joints(urdf_path: Path) -> dict[str, RevoluteJoint]:
  """Revolute joints by name, each with the tag frame of its child's fixed joint."""
  joints = rr.urdf.UrdfTree.from_file_path(str(urdf_path)).joints()
  tag_by_parent = {
    j.parent_link: _pose(j.origin_xyz, j.origin_rpy)
    for j in joints
    if j.joint_type == "fixed"
  }
  return {
    j.name: RevoluteJoint(
      base_T_joint=_pose(j.origin_xyz, j.origin_rpy),
      # A joint that declares no limit reports zero for both.
      lower=j.limit_lower if j.limit_lower != j.limit_upper else -math.pi,
      upper=j.limit_upper if j.limit_lower != j.limit_upper else math.pi,
      joint_T_tag0=tag_by_parent.get(j.child_link),
    )
    for j in joints
    if j.joint_type == "revolute"
  }
