"""Post-bundle cull of landmark observations that disagree with the solve."""

from __future__ import annotations

import gtsam
import numpy as np
import numpy.typing as npt
from specialist_camera import mei_project

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.geometry import max_parallax

Float64Array = npt.NDArray[np.float64]

# Largest reprojection residual (px) a surviving observation may carry.
_MAX_RESIDUAL = 7.5
_MIN_OBSERVATIONS = 2
# Parallax the surviving observations' camera centers must span.
_MIN_PARALLAX = float(np.deg2rad(0.1))
_MIN_DEPTH = 1e-3

__all__ = ["cull_and_reproject", "cull_outlier_landmarks"]

_Judged = tuple[
  list[tuple[int, int, Float64Array]],
  Float64Array,
  Float64Array,
  npt.NDArray[np.bool_],
  list[Float64Array],
]


def _judged_observations(
  points: dict[int, Float64Array],
  tracks: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
) -> _Judged:
  """In-range observations with residual and verdict, plus keyframe lens centers."""
  lens_poses = [body_pose.compose(lens.body_T_lens) for body_pose in body_poses]
  lens_centers = [np.asarray(p.translation(), dtype=np.float64) for p in lens_poses]
  observed: list[tuple[int, int, Float64Array]] = []
  in_lens: list[Float64Array] = []
  for tid, p in points.items():
    point = np.asarray(p, dtype=np.float64)
    for kf, pixel in tracks.get(tid, {}).items():
      if not (0 <= kf < len(body_poses)):
        continue
      observed.append((tid, kf, np.asarray(pixel, dtype=np.float64)))
      in_lens.append(np.asarray(lens_poses[kf].transformTo(point), dtype=np.float64))
  if not observed:
    empty = np.empty(0, dtype=np.float64)
    return [], empty.reshape(0, 2), empty, np.empty(0, dtype=np.bool_), lens_centers

  lens_points = np.asarray(in_lens)
  in_front = lens_points[:, 2] > _MIN_DEPTH
  projected = np.full((len(observed), 2), np.nan, dtype=np.float64)
  if in_front.any():
    projected[in_front] = mei_project(lens_points[in_front], lens.params)
  residual = np.linalg.norm(
    projected - np.asarray([pix for _, _, pix in observed]), axis=1
  )
  survives = in_front & np.isfinite(residual) & (residual <= _MAX_RESIDUAL)
  return observed, projected, residual, survives, lens_centers


def _kept_tracks(
  points: dict[int, Float64Array],
  observed: list[tuple[int, int, Float64Array]],
  survives: npt.NDArray[np.bool_],
  lens_centers: list[Float64Array],
) -> dict[int, dict[int, Float64Array]]:
  """Surviving observations per surviving landmark, in the caller's landmark order."""
  by_track: dict[int, dict[int, Float64Array]] = {}
  centers_by_track: dict[int, list[Float64Array]] = {}
  for keep, (tid, kf, pixel) in zip(survives, observed, strict=True):
    if not keep:
      continue
    by_track.setdefault(tid, {})[kf] = pixel
    centers_by_track.setdefault(tid, []).append(lens_centers[kf])

  kept: dict[int, dict[int, Float64Array]] = {}
  for tid in points:
    survivors = by_track.get(tid)
    if survivors is None or len(survivors) < _MIN_OBSERVATIONS:
      continue
    if max_parallax(centers_by_track[tid], points[tid]) < _MIN_PARALLAX:
      continue
    kept[tid] = survivors
  return kept


def cull_outlier_landmarks(
  points: dict[int, Float64Array],
  tracks: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
) -> dict[int, dict[int, Float64Array]]:
  """Tracks ``{tid: {kf: pixel}}`` cut to surviving observations of kept landmarks."""
  observed, _, _, survives, lens_centers = _judged_observations(
    points, tracks, body_poses, lens
  )
  return _kept_tracks(points, observed, survives, lens_centers)


def cull_and_reproject(
  points: dict[int, Float64Array],
  tracks: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
) -> tuple[dict[int, Float64Array], list[Float64Array], list[float]]:
  """Kept landmark points, per-keyframe survivor reprojections and RMSE (px)."""
  observed, projected, residual, survives, lens_centers = _judged_observations(
    points, tracks, body_poses, lens
  )
  observations = _kept_tracks(points, observed, survives, lens_centers)
  kept = {tid: points[tid] for tid in observations}
  n = len(body_poses)
  pixels: list[list[Float64Array]] = [[] for _ in range(n)]
  residuals: list[list[float]] = [[] for _ in range(n)]
  for keep, (tid, kf, _pixel), row, err in zip(
    survives, observed, projected, residual, strict=True
  ):
    if keep and tid in observations:
      pixels[kf].append(row)
      residuals[kf].append(float(err))
  return (
    kept,
    [np.stack(b) if b else np.empty((0, 2)) for b in pixels],
    [
      float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
      for errs in residuals
    ],
  )
