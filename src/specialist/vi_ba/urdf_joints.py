"""Revolute-joint kinematics from the gripper URDF, in the URDF root frame."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import gtsam
import numpy as np


@dataclass(frozen=True)
class RevoluteJoint:
  """A URDF ``revolute`` joint about +z; its ``<axis>`` and ``<mimic>`` are not read."""

  base_T_joint: gtsam.Pose3
  lower: float
  upper: float
  # The child link's fixed-joint origin to its tag link.
  joint_T_tag0: gtsam.Pose3 | None


def _pose_from_origin(elem: ET.Element | None) -> gtsam.Pose3:
  """``<origin xyz rpy>`` to a ``Pose3``, with URDF's R = Rz(yaw)Ry(pitch)Rx(roll)."""
  if elem is None:
    return gtsam.Pose3()
  xyz = [float(v) for v in elem.get("xyz", "0 0 0").split()]
  rpy = [float(v) for v in elem.get("rpy", "0 0 0").split()]
  return gtsam.Pose3(gtsam.Rot3.RzRyRx(rpy[0], rpy[1], rpy[2]), gtsam.Point3(*xyz))


def read_revolute_joints(urdf_path: Path) -> dict[str, RevoluteJoint]:
  """Revolute joints by name, with the tag frame of each child's fixed joint, if any."""
  root = ET.parse(urdf_path).getroot()
  # Fixed-joint origins keyed by parent link (the finger -> tag frame).
  tag_by_parent: dict[str, gtsam.Pose3] = {}
  for joint in root.findall("joint"):
    parent = joint.find("parent")
    if joint.get("type") == "fixed" and parent is not None:
      tag_by_parent[parent.get("link", "")] = _pose_from_origin(joint.find("origin"))

  joints: dict[str, RevoluteJoint] = {}
  for joint in root.findall("joint"):
    if joint.get("type") != "revolute":
      continue
    limit = joint.find("limit")
    child = joint.find("child")
    child_link = child.get("link", "") if child is not None else ""
    joints[joint.get("name", "")] = RevoluteJoint(
      base_T_joint=_pose_from_origin(joint.find("origin")),
      lower=float(limit.get("lower", -np.pi)) if limit is not None else -np.pi,
      upper=float(limit.get("upper", np.pi)) if limit is not None else np.pi,
      joint_T_tag0=tag_by_parent.get(child_link),
    )
  return joints
