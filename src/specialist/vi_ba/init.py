"""Trajectory seeding: multi-view triangulation and visual-inertial alignment."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from specialist.vi_ba.geometry import translation

Float64Array = npt.NDArray[np.float64]

# Cheirality bound on a unit bearing's z and on a view depth (m).
_MIN_CHEIRALITY_Z = 0.05
# Tikhonov ridge on the alignment's normal equations.
_ALIGN_RIDGE = 1e-12


@dataclass(frozen=True)
class Alignment:
  """Visual-inertial alignment, ``gravity_world`` in the visual trajectory's frame."""

  scale: float
  gravity_world: Float64Array
  velocities: list[Float64Array]


__all__ = [
  "Alignment",
  "apply_alignment_to_poses",
  "body_poses_from_lens_poses",
  "fill_lens_poses",
  "gravity_alignment_rotation",
  "triangulate_tracks_batched",
  "visual_inertial_align",
]


def fill_lens_poses(
  lens_poses: dict[int, gtsam.Pose3], n_keyframes: int
) -> list[gtsam.Pose3]:
  """Every keyframe's world_T_lens, the unregistered ones interpolated along the
  SE(3) geodesic between their registered neighbors and held at the ends."""
  reg = sorted(lens_poses)
  out: list[gtsam.Pose3] = []
  for k in range(n_keyframes):
    after = bisect_left(reg, k)
    lo, hi = reg[max(after - 1, 0)], reg[min(after, len(reg) - 1)]
    pa, pb = lens_poses[lo], lens_poses[hi]
    if hi in (k, lo):
      out.append(pb)
    else:
      f = (k - lo) / (hi - lo)
      out.append(pa.compose(gtsam.Pose3.Expmap(f * gtsam.Pose3.Logmap(pa.between(pb)))))
  return out


def body_poses_from_lens_poses(
  lens_poses: list[gtsam.Pose3],
  body_T_lens: gtsam.Pose3,
) -> list[gtsam.Pose3]:
  """``world_T_body = world_T_lens · body_T_lens⁻¹`` per keyframe."""
  inv = body_T_lens.inverse()
  return [p.compose(inv) for p in lens_poses]


def _solve_each(normal: Float64Array, rhs: Float64Array) -> Float64Array:
  """``np.linalg.solve`` over a stack of systems, NaN where one is singular."""
  out = np.full_like(rhs, np.nan)

  def solve(index: npt.NDArray[np.int64]) -> None:
    if index.size == 0:
      return
    try:
      out[index] = np.linalg.solve(normal[index], rhs[index][:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
      if index.size > 1:
        half = index.size // 2
        solve(index[:half])
        solve(index[half:])

  solve(np.arange(len(normal), dtype=np.int64))
  return out


def triangulate_tracks_batched(
  bearings: Float64Array,
  track: npt.NDArray[np.int64],
  n_tracks: int,
  lens_rotations: Float64Array,
  lens_centers: Float64Array,
  min_landmark_depth: float,
  max_landmark_depth: float,
) -> tuple[Float64Array, npt.NDArray[np.bool_]]:
  """Closest-point triangulation of every ``track`` at once: the ``(n_tracks, 3)``
  points and which passed the view-count, cheirality and depth checks."""
  points = np.full((n_tracks, 3), np.nan)
  passed = np.zeros(n_tracks, dtype=bool)
  valid = bearings[:, 2] > _MIN_CHEIRALITY_Z
  track = track[valid]
  rotations = lens_rotations[valid]
  centers = lens_centers[valid]
  rays = np.einsum("nij,nj->ni", rotations, bearings[valid])
  rays /= np.linalg.norm(rays, axis=1, keepdims=True)
  n_valid = np.bincount(track, minlength=n_tracks)
  # Normal equations per track: (Σ I - dᵢdᵢᵀ) X = Σ (I - dᵢdᵢᵀ) Cᵢ.
  outer = rays[:, :, None] * rays[:, None, :]
  normal = -np.stack(
    [np.bincount(track, outer[:, i, j], n_tracks) for i in range(3) for j in range(3)],
    axis=1,
  ).reshape(n_tracks, 3, 3)
  normal[:, [0, 1, 2], [0, 1, 2]] += n_valid[:, None]
  along = np.einsum("ni,ni->n", rays, centers)
  rhs = np.stack(
    [
      np.bincount(track, centers[:, i] - rays[:, i] * along, n_tracks) for i in range(3)
    ],
    axis=1,
  )
  xyz = _solve_each(normal.astype(np.float64), rhs.astype(np.float64))
  finite = (n_valid >= 2) & np.isfinite(xyz).all(axis=1)
  # Per-view depth, then the count and median of the views in front of the lens.
  depths = np.einsum("ni,ni->n", rotations[:, :, 2], xyz[track] - centers)
  good = finite[track] & (depths >= _MIN_CHEIRALITY_Z)
  n_good = np.bincount(track[good], minlength=n_tracks)
  order = np.lexsort((depths[good], track[good]))
  sorted_depths = depths[good][order]
  start = np.concatenate(([0], np.cumsum(n_good)[:-1]))
  enough = finite & (n_good >= 2)
  lo = start[enough] + (n_good[enough] - 1) // 2
  hi = start[enough] + n_good[enough] // 2
  median = 0.5 * (sorted_depths[lo] + sorted_depths[hi])
  in_range = (median >= min_landmark_depth) & (median <= max_landmark_depth)
  kept = np.flatnonzero(enough)[in_range]
  points[kept] = xyz[kept]
  passed[kept] = True
  return points, passed


def visual_inertial_align(
  body_poses: list[gtsam.Pose3],
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements],
  fixed_gravity: Float64Array | None = None,
  velocity_regularization: float = 1.0e-2,
) -> Alignment:
  """Least-squares velocities (and gravity, unless pinned) of the NavState
  constraints between keyframes, ``velocity_regularization`` a Tikhonov weight."""
  n = len(body_poses)
  if n < 4:
    raise ValueError(f"visual_inertial_align needs ≥4 keyframes, got {n}")
  if len(preintegrated) != n - 1:
    raise ValueError(f"expected {n - 1} pim entries, got {len(preintegrated)}")

  n_unknowns = 3 * n + 4
  n_eq_data = 6 * (n - 1)
  n_eq_reg = 3 * n if velocity_regularization > 0 else 0
  rows: list[npt.NDArray[np.int64]] = []
  cols: list[npt.NDArray[np.int64]] = []
  vals: list[Float64Array] = []
  b = np.zeros(n_eq_data + n_eq_reg + 4)
  axis = np.arange(3, dtype=np.int64)

  def block(row: int, col: int, values: Float64Array) -> None:
    """A 3-vector on the diagonal of the 3x3 block at ``(row, col)``."""
    rows.append(row + axis)
    cols.append(col + axis)
    vals.append(np.broadcast_to(values, (3,)).astype(np.float64))

  for i in range(n - 1):
    dt = float(preintegrated[i].deltaTij())
    rot_i = body_poses[i].rotation().matrix()
    p_visual_delta = np.asarray(body_poses[i + 1].translation()) - np.asarray(
      body_poses[i].translation()
    )
    pim = preintegrated[i]
    delta_p = rot_i @ np.asarray(pim.deltaPij(), dtype=np.float64)
    delta_v = rot_i @ np.asarray(pim.deltaVij(), dtype=np.float64)

    row_p = 6 * i
    row_v = 6 * i + 3
    # Position equation: s·Δp_v − v_i·Δt − 0.5·g·Δt² = R_i·Δp_pim
    rows.append(row_p + axis)
    cols.append(np.zeros(3, dtype=np.int64))
    vals.append(np.asarray(p_visual_delta, dtype=np.float64))
    block(row_p, 1, np.full(3, -0.5 * dt * dt))
    block(row_p, 4 + 3 * i, np.full(3, -dt))
    b[row_p : row_p + 3] = delta_p
    # Velocity equation: v_{i+1} − v_i − g·Δt = R_i·Δv_pim
    block(row_v, 1, np.full(3, -dt))
    block(row_v, 4 + 3 * i, np.full(3, -1.0))
    block(row_v, 4 + 3 * (i + 1), np.ones(3))
    b[row_v : row_v + 3] = delta_v

  if n_eq_reg:
    sqrt_lam = float(np.sqrt(velocity_regularization))
    for i in range(n):
      block(n_eq_data + 3 * i, 4 + 3 * i, np.full(3, sqrt_lam))

  n_rows = n_eq_data + n_eq_reg
  # Gravity and scale are pinned by weight; solved freely they drift on slow motion.
  w = 1.0e3
  if fixed_gravity is not None:
    block(n_rows, 1, np.full(3, w))
    b[n_rows : n_rows + 3] = w * np.asarray(fixed_gravity, dtype=np.float64)
    n_rows += 3
  rows.append(np.array([n_rows], dtype=np.int64))
  cols.append(np.array([0], dtype=np.int64))
  vals.append(np.array([w]))
  b[n_rows] = w  # unit scale
  n_rows += 1

  A = sp.coo_matrix(
    (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
    shape=(n_rows, n_unknowns),
  ).tocsr()
  b = b[:n_rows]
  # The ridge resolves an unobservable unknown to zero.
  normal = (A.T @ A + _ALIGN_RIDGE * sp.identity(n_unknowns)).tocsc()
  x = spsolve(normal, A.T @ b)
  scale = float(x[0])
  gravity = x[1:4].astype(np.float64)
  velocities = [
    np.asarray(x[4 + 3 * i : 4 + 3 * i + 3], dtype=np.float64) for i in range(n)
  ]
  return Alignment(scale=scale, gravity_world=gravity, velocities=velocities)


def gravity_alignment_rotation(gravity_world: Float64Array) -> gtsam.Rot3:
  """The minimum-angle rotation taking ``gravity_world`` onto ``(0, 0, −|g|)``."""
  g = np.asarray(gravity_world, dtype=np.float64)
  g_norm = float(np.linalg.norm(g))
  if g_norm < 1e-10:
    return gtsam.Rot3()
  target = np.array([0.0, 0.0, -1.0], dtype=np.float64)
  src = g / g_norm
  dot = float(np.dot(src, target))
  if dot > 1.0 - 1e-12:
    return gtsam.Rot3()
  if dot < -1.0 + 1e-12:
    # 180° — pick any perpendicular axis.
    axis = np.cross(src, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(axis) < 1e-6:
      axis = np.cross(src, np.array([0.0, 1.0, 0.0]))
    axis = axis / np.linalg.norm(axis)
    return gtsam.Rot3.AxisAngle(gtsam.Point3(*axis), np.pi)
  axis = np.cross(src, target)
  axis = axis / np.linalg.norm(axis)
  angle = float(np.arccos(dot))
  return gtsam.Rot3.AxisAngle(gtsam.Point3(*axis), angle)


def apply_alignment_to_poses(
  body_poses: list[gtsam.Pose3],
  alignment: Alignment,
) -> tuple[list[gtsam.Pose3], gtsam.Rot3]:
  """Body poses scaled and turned so gravity points along −z, plus that rotation."""
  rot = gravity_alignment_rotation(alignment.gravity_world)
  r = rot.matrix()
  return [
    gtsam.Pose3(
      gtsam.Rot3(r @ p.rotation().matrix()), r @ (alignment.scale * translation(p))
    )
    for p in body_poses
  ], rot
