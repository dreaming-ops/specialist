"""GTSAM key namespace shared by the graph builders and the solvers."""

from __future__ import annotations

__all__ = [
  "ANCHOR_CHAR",
  "BASE_TAG_CHAR",
  "BIAS_CHAR",
  "JOINT_ANGLE_CHAR",
  "LANDMARK_CHAR",
  "MAX_KEYFRAMES_PER_CAMERA",
  "PER_KEYFRAME_CHARS",
  "POSE_CHAR",
  "SEAT_CHAR",
  "VELOCITY_CHAR",
]

POSE_CHAR = "x"
VELOCITY_CHAR = "v"
BIAS_CHAR = "b"
JOINT_ANGLE_CHAR = "q"
LANDMARK_CHAR = "l"
ANCHOR_CHAR = "A"
BASE_TAG_CHAR = "G"
SEAT_CHAR = "E"
PER_KEYFRAME_CHARS = frozenset({POSE_CHAR, VELOCITY_CHAR, BIAS_CHAR, JOINT_ANGLE_CHAR})
# A per-keyframe key's index is its slice index times this, plus the keyframe.
MAX_KEYFRAMES_PER_CAMERA = 100_000
