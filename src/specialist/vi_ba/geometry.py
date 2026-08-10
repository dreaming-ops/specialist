"""Shared geometric helpers for VI-BA."""

from __future__ import annotations

import gtsam
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "bounding_box_diagonal",
  "matrices_to_quats",
  "max_parallax",
  "path_length",
  "quat_xyzw",
  "quats_to_matrices",
]


def _canonical(quats: Float64Array) -> Float64Array:
  """Every quaternion as its ``w >= 0`` representative."""
  out: Float64Array = np.where((quats[:, 3] < 0.0)[:, None], -quats, quats)
  return out


def quat_xyzw(rotation: gtsam.Rot3) -> Float64Array:
  """A rotation as an xyzw quaternion with ``w >= 0``."""
  q = rotation.toQuaternion()
  raw = np.array([[q.x(), q.y(), q.z(), q.w()]], dtype=np.float64)
  out: Float64Array = _canonical(raw)[0]
  return out


def quats_to_matrices(quaternions: Float64Array) -> Float64Array:
  """Batch xyzw quaternions to rotation matrices, ``(n, 4) -> (n, 3, 3)``."""
  out: Float64Array = Rotation.from_quat(quaternions).as_matrix()
  return out


def matrices_to_quats(matrices: Float64Array) -> Float64Array:
  """Batch rotation matrices to xyzw quaternions, ``(n, 3, 3) -> (n, 4)``."""
  out: Float64Array = _canonical(Rotation.from_matrix(matrices).as_quat())
  return out


def max_parallax(
  centers: Float64Array | list[Float64Array], point: Float64Array
) -> float:
  """Widest pairwise angle (radians) the camera centers subtend at ``point``."""
  offsets = np.asarray(point, dtype=np.float64) - np.asarray(centers, dtype=np.float64)
  norms = np.linalg.norm(offsets, axis=1)
  usable = norms > 1e-12
  if int(usable.sum()) < 2:
    return 0.0
  dirs = offsets[usable] / norms[usable, None]
  cos = dirs @ dirs.T
  # Exclude the self-pairs from the minimum.
  np.fill_diagonal(cos, 1.0)
  return float(np.arccos(np.clip(cos.min(), -1.0, 1.0)))


def bounding_box_diagonal(poses: list[gtsam.Pose3]) -> float:
  """Diagonal of the axis-aligned box enclosing the trajectory's translations."""
  if len(poses) < 2:
    return 0.0
  pts = np.array([np.asarray(p.translation(), dtype=np.float64) for p in poses])
  return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def path_length(poses: list[gtsam.Pose3]) -> float:
  """Total distance traveled along the trajectory, summed keyframe to keyframe."""
  pts = [np.asarray(p.translation(), dtype=np.float64) for p in poses]
  return float(sum(np.linalg.norm(pts[i] - pts[i - 1]) for i in range(1, len(pts))))
