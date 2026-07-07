"""Post-bundle cull of landmark observations that disagree with the solve."""

from __future__ import annotations

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import mei_project
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.geometry import max_parallax

Float64Array = npt.NDArray[np.float64]

# Largest reprojection residual (px) a surviving observation may carry.
MAX_RESIDUAL_PX = 7.5
MIN_OBSERVATIONS = 2
# Parallax the surviving observations' camera centers must span.
MIN_PARALLAX = float(np.deg2rad(0.1))
MIN_DEPTH = 1e-3

__all__ = ["cull_outlier_landmarks"]


def cull_outlier_landmarks(
  points: dict[int, Float64Array],
  tracks: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
) -> dict[int, dict[int, Float64Array]]:
  """Tracks ``{tid: {kf: pixel}}`` cut to surviving observations of kept landmarks."""
  kept: dict[int, dict[int, Float64Array]] = {}
  for tid, p in points.items():
    point = np.asarray(p, dtype=np.float64)
    survivors: dict[int, Float64Array] = {}
    centers: list[Float64Array] = []
    for kf, pixel in tracks.get(tid, {}).items():
      if not (0 <= kf < len(body_poses)):
        continue
      world_T_lens = body_poses[kf].compose(lens.body_T_lens)
      p_lens = np.asarray(world_T_lens.transformTo(point), dtype=np.float64)
      if p_lens[2] <= MIN_DEPTH:
        continue
      pred = mei_project(p_lens[None, :], lens.params)[0]
      if not np.all(np.isfinite(pred)):
        continue
      pixel_arr = np.asarray(pixel, dtype=np.float64)
      if float(np.linalg.norm(pred - pixel_arr)) > MAX_RESIDUAL_PX:
        continue
      survivors[kf] = pixel_arr
      centers.append(np.asarray(world_T_lens.translation(), dtype=np.float64))
    if len(survivors) < MIN_OBSERVATIONS:
      continue
    if max_parallax(centers, point) < MIN_PARALLAX:
      continue
    kept[tid] = survivors
  return kept
