"""Per-camera GTSAM symbol-key allocation for multi-camera VI-BA."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import gtsam

__all__ = ["CameraKeys"]

ANCHOR_CHAR = "A"
_POSE_CHAR = "p"
_VEL_CHAR = "v"
_BIAS_CHAR = "b"
_LANDMARK_CHAR = "l"

_MAX_KEYFRAMES_PER_CAMERA = 100_000
_MAX_LANDMARKS_PER_CAMERA = 1_000_000


class CameraKeys:
  """Collision-free GTSAM keys per camera; anchor corners at ``symbol('A', 0..3)``."""

  def __init__(self, slots: Sequence[str]) -> None:
    if len(set(slots)) != len(slots):
      raise ValueError(f"camera slots must be unique: {list(slots)}")
    self._names: list[str] = list(slots)
    self._kf_step = int(_MAX_KEYFRAMES_PER_CAMERA)
    self._lm_step = int(_MAX_LANDMARKS_PER_CAMERA)

  def camera_index(self, slot: str) -> int:
    if slot not in self._names:
      raise KeyError(f"unknown camera {slot!r}")
    return self._names.index(slot)

  def _kf_key(self, char: str, slot: str, keyframe_idx: int) -> int:
    """Collision-free key for a per-keyframe variable in one camera's slice."""
    return int(
      gtsam.symbol(char, self.camera_index(slot) * self._kf_step + keyframe_idx)
    )

  def pose_key(self, slot: str, keyframe_idx: int) -> int:
    return self._kf_key(_POSE_CHAR, slot, keyframe_idx)

  def velocity_key(self, slot: str, keyframe_idx: int) -> int:
    return self._kf_key(_VEL_CHAR, slot, keyframe_idx)

  def bias_key(self, slot: str, keyframe_idx: int) -> int:
    return self._kf_key(_BIAS_CHAR, slot, keyframe_idx)

  def _lm_key(self, slot: str, half: int, track_id: int) -> int:
    """A landmark key in a camera's front (``half=0``) or rear (``half=1``) slot."""
    return int(
      gtsam.symbol(
        _LANDMARK_CHAR,
        (self.camera_index(slot) * 2 + half) * self._lm_step + track_id,
      )
    )

  def front_landmark_key(self, slot: str, track_id: int) -> int:
    return self._lm_key(slot, 0, track_id)

  def rear_landmark_key(self, slot: str, track_id: int) -> int:
    return self._lm_key(slot, 1, track_id)

  def anchor_corner_key(self, corner_idx: int) -> int:
    """Key for one of the anchor tag's four metric corners, ``corner_idx`` 0..3."""
    return int(gtsam.symbol(ANCHOR_CHAR, corner_idx))

  def vi_graph_kwargs(self, slot: str) -> dict[str, Callable[[int], int]]:
    """Per-camera key callables to pass to ``build_vi_graph``."""
    return {
      "pose_key_fn": lambda i: self.pose_key(slot, i),
      "velocity_key_fn": lambda i: self.velocity_key(slot, i),
      "bias_key_fn": lambda i: self.bias_key(slot, i),
      "front_landmark_key_fn": lambda i: self.front_landmark_key(slot, i),
      "rear_landmark_key_fn": lambda i: self.rear_landmark_key(slot, i),
    }
