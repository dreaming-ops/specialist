"""Decode Insta360's factory ``offset_v3``: both lenses' intrinsics and body_T_lens."""

import math
from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt

__all__ = ["LensOffset", "OffsetV3", "parse_offset_v3"]

_BLOCK_SIZE = 19
_TOTAL_SIZE = 1 + 2 * _BLOCK_SIZE + 1


@dataclass(frozen=True)
class LensOffset:
  """One lens's ``offset_v3`` block; ``fx``..``cy`` in stitched-frame pixels, ``cx``
  from the stitched left edge."""

  xi: float
  fx: float
  fy: float
  cx: float
  cy: float
  yaw: float
  pitch: float
  roll: float
  tx: float
  ty: float
  tz: float
  k1: float
  k2: float
  k3: float
  p1: float
  p2: float
  frame_width: int
  frame_height: int

  def to_pose3(self) -> gtsam.Pose3:
    """``body_T_lens``, with ``(tx, ty, tz)`` the body origin in the lens frame."""
    body_R_lens = gtsam.Rot3.Ypr(self.yaw, self.pitch, self.roll)
    lens_t_body = gtsam.Point3(self.tx, self.ty, self.tz)
    return gtsam.Pose3(body_R_lens, -body_R_lens.rotate(lens_t_body))


@dataclass(frozen=True)
class OffsetV3:
  """Both lenses; the front lens's rotation gains a 180° turn about body z."""

  front: LensOffset
  rear: LensOffset

  @property
  def body_T_front_lens(self) -> gtsam.Pose3:
    raw = self.front.to_pose3()
    flip_z = gtsam.Rot3.Rz(math.pi)
    return gtsam.Pose3(flip_z.compose(raw.rotation()), raw.translation())

  @property
  def body_T_rear_lens(self) -> gtsam.Pose3:
    return self.rear.to_pose3()


def _decode_block(block: npt.NDArray[np.float64]) -> LensOffset:
  values = block[:16].tolist()
  values[5:8] = map(math.radians, values[5:8])
  return LensOffset(*values, *map(int, block[16:18]))


def parse_offset_v3(arr: npt.NDArray[np.float64]) -> OffsetV3:
  """Decode the 40-float ``offset_v3`` array into both lenses' blocks."""
  if arr.shape != (_TOTAL_SIZE,):
    raise ValueError(f"offset_v3 must have shape ({_TOTAL_SIZE},); got {arr.shape}")
  if int(arr[0]) != 2:
    raise ValueError(f"offset_v3 lens count must be 2; got {arr[0]}")
  front = _decode_block(arr[1 : 1 + _BLOCK_SIZE])
  rear = _decode_block(arr[1 + _BLOCK_SIZE : 1 + 2 * _BLOCK_SIZE])
  return OffsetV3(front=front, rear=rear)
