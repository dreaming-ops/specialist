"""Post-bundle cull of landmark observations that disagree with the solve."""

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
  npt.NDArray[np.int64],
  npt.NDArray[np.int64],
  Float64Array,
  Float64Array,
  Float64Array,
  npt.NDArray[np.bool_],
  Float64Array,
]


def _judged_observations(
  points: dict[int, Float64Array],
  tracks: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
) -> _Judged:
  """Every in-range observation's landmark id, keyframe, pixel, projection,
  residual and survival verdict, plus each keyframe's lens center."""
  lens_poses = [body_pose.compose(lens.body_T_lens) for body_pose in body_poses]
  lens_centers = np.asarray([p.translation() for p in lens_poses], dtype=np.float64)
  n = len(body_poses)
  tids: list[int] = []
  kfs: list[int] = []
  pixels: list[Float64Array] = []
  xyz: list[Float64Array] = []
  for tid, p in points.items():
    for kf, pixel in tracks.get(tid, {}).items():
      if 0 <= kf < n:
        tids.append(tid)
        kfs.append(kf)
        pixels.append(pixel)
        xyz.append(p)
  tid_arr = np.asarray(tids, dtype=np.int64)
  kf_arr = np.asarray(kfs, dtype=np.int64)
  pixel_arr = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
  rotations = np.asarray([p.rotation().matrix() for p in lens_poses])
  lens_points = np.einsum(
    "nji,nj->ni",
    rotations[kf_arr],
    np.asarray(xyz, dtype=np.float64).reshape(-1, 3) - lens_centers[kf_arr],
  )
  in_front = lens_points[:, 2] > _MIN_DEPTH
  projected = np.full((len(tids), 2), np.nan, dtype=np.float64)
  if in_front.any():
    projected[in_front] = mei_project(lens_points[in_front], lens.params)
  residual = np.linalg.norm(projected - pixel_arr, axis=1)
  survives = in_front & np.isfinite(residual) & (residual <= _MAX_RESIDUAL)
  return tid_arr, kf_arr, pixel_arr, projected, residual, survives, lens_centers


def _kept_tracks(
  points: dict[int, Float64Array],
  tids: npt.NDArray[np.int64],
  kfs: npt.NDArray[np.int64],
  pixels: Float64Array,
  survives: npt.NDArray[np.bool_],
  lens_centers: Float64Array,
) -> dict[int, dict[int, Float64Array]]:
  """Surviving observations per surviving landmark, in the caller's landmark order."""
  by_track: dict[int, dict[int, Float64Array]] = {}
  for i in np.flatnonzero(survives):
    by_track.setdefault(int(tids[i]), {})[int(kfs[i])] = pixels[i]
  kept: dict[int, dict[int, Float64Array]] = {}
  for tid in points:
    survivors = by_track.get(tid)
    if survivors is None or len(survivors) < _MIN_OBSERVATIONS:
      continue
    if max_parallax(lens_centers[list(survivors)], points[tid]) < _MIN_PARALLAX:
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
  tids, kfs, pixels, _, _, survives, lens_centers = _judged_observations(
    points, tracks, body_poses, lens
  )
  return _kept_tracks(points, tids, kfs, pixels, survives, lens_centers)


def cull_and_reproject(
  points: dict[int, Float64Array],
  tracks: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
) -> tuple[dict[int, Float64Array], list[Float64Array], list[float]]:
  """Kept landmark points, per-keyframe survivor reprojections and RMSE (px)."""
  tids, kfs, pixels, projected, residual, survives, lens_centers = _judged_observations(
    points, tracks, body_poses, lens
  )
  observations = _kept_tracks(points, tids, kfs, pixels, survives, lens_centers)
  kept = {tid: points[tid] for tid in observations}
  n = len(body_poses)
  shown = survives & np.isin(tids, np.fromiter(observations, dtype=np.int64))
  order = np.argsort(kfs[shown], kind="stable")
  rows = np.flatnonzero(shown)[order]
  bounds = np.searchsorted(kfs[rows], np.arange(n + 1))
  return (
    kept,
    [projected[rows[a:b]] for a, b in zip(bounds[:-1], bounds[1:], strict=True)],
    [
      float(np.sqrt(np.mean(np.square(residual[rows[a:b]])))) if b > a else float("nan")
      for a, b in zip(bounds[:-1], bounds[1:], strict=True)
    ],
  )
