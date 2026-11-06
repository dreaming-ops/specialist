"""Stored xyzw quaternions as rotation matrices."""

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]

__all__ = ["quats_to_matrices"]


def quats_to_matrices(quaternions: Float64Array) -> Float64Array:
  """Batch xyzw quaternions to rotation matrices, ``(n, 4) -> (n, 3, 3)``."""
  quats = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)
  x, y, z, w = (quats / np.linalg.norm(quats, axis=1, keepdims=True)).T
  return np.stack(
    [
      np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
      np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
      np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ],
    axis=1,
  )
