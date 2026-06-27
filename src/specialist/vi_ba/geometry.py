"""Shared geometric helpers for VI-BA."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]


def max_parallax(centers: list[Float64Array], point: Float64Array) -> float:
  """Widest pairwise angle (radians) the camera centers subtend at ``point``."""
  dirs: list[Float64Array] = []
  for c in centers:
    d = point - c
    n = float(np.linalg.norm(d))
    if n > 1e-12:
      dirs.append(d / n)
  if len(dirs) < 2:
    return 0.0
  d_arr = np.asarray(dirs, dtype=np.float64)
  cos = d_arr @ d_arr.T
  iu = np.triu_indices(d_arr.shape[0], k=1)
  return float(np.arccos(np.clip(cos[iu].min(), -1.0, 1.0)))
