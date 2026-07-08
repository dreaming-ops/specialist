"""Trajectory seeding: multi-view triangulation and visual-inertial alignment."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import gtsam
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as Rscp
from scipy.spatial.transform import Slerp

from specialist.vi_ba.geometry import max_parallax

Float64Array = npt.NDArray[np.float64]

# Incremental-SfM tuning: parallax, PnP inliers, triangulation residual, passes.
SFM_MIN_PARALLAX = float(np.deg2rad(0.6))
# Minimum parallax the two-view init pair (0, K) must span.
SFM_INIT_PAIR_MIN_PARALLAX = float(np.deg2rad(2.0))
SFM_MIN_PNP_POINTS = 6
SFM_MAX_REPROJ = 0.01
SFM_PASSES = 8


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
  "gravity_alignment_rotation",
  "incremental_sfm",
  "interpolate_camera_poses",
  "pick_init_pair",
  "triangulate_track_multiview",
  "two_view_essential",
  "visual_inertial_align",
]


def two_view_essential(
  bearings_a: Float64Array,
  bearings_b: Float64Array,
  threshold: float = 1e-2,
) -> tuple[Float64Array, Float64Array, int]:
  """Essential-matrix ``(R_cw_last, t_cw_last, n_inliers)``, translation unit-norm."""
  if bearings_a.shape[0] < 8 or bearings_b.shape[0] != bearings_a.shape[0]:
    raise RuntimeError(
      f"two_view_essential needs ≥8 correspondences, got {bearings_a.shape[0]}"
    )
  ok = (bearings_a[:, 2] > 1e-3) & (bearings_b[:, 2] > 1e-3)
  if int(ok.sum()) < 8:
    raise RuntimeError(
      f"only {int(ok.sum())} in-forward-hemisphere correspondences — too few"
    )
  norm_a = (bearings_a[ok][:, :2] / bearings_a[ok][:, 2:3]).astype(np.float64)
  norm_b = (bearings_b[ok][:, :2] / bearings_b[ok][:, 2:3]).astype(np.float64)
  E, _ = cv2.findEssentialMat(
    norm_a,
    norm_b,
    focal=1.0,
    pp=(0.0, 0.0),
    method=cv2.RANSAC,
    prob=0.999,
    threshold=threshold,
  )
  if E is None:
    raise RuntimeError("essential matrix failed on the init pair")
  n_inliers, R, t, _ = cv2.recoverPose(E, norm_a, norm_b)
  return R.astype(np.float64), t.flatten().astype(np.float64), int(n_inliers)


def pick_init_pair(
  tracks: dict[int, dict[int, Float64Array]],
  n_keyframes: int,
  min_shared: int,
) -> tuple[int, int, int]:
  """Widest ``(0, K)`` pair keeping ``min_shared`` tracks, else the most-shared pair."""
  first = 0

  def shared_count(last: int) -> int:
    return sum(1 for obs in tracks.values() if first in obs and last in obs)

  for last in range(n_keyframes - 1, 0, -1):
    shared = shared_count(last)
    if shared >= min_shared:
      return first, last, shared
  best_last, best_shared = 1, 0
  for last in range(1, n_keyframes):
    shared = shared_count(last)
    if shared > best_shared:
      best_last, best_shared = last, shared
  return first, best_last, best_shared


def interpolate_camera_poses(
  r_cw_last: Float64Array,
  t_cw_last: Float64Array,
  n_keyframes: int,
  init_first: int = 0,
  init_last: int | None = None,
) -> list[gtsam.Pose3]:
  """Per-keyframe ``world_T_lens`` by SLERP and linear-t through the init pair."""
  if init_last is None:
    init_last = n_keyframes - 1
  if init_last <= init_first:
    raise ValueError(f"init_last ({init_last}) must exceed init_first ({init_first})")
  key_rots = Rscp.from_matrix(np.stack([np.eye(3), r_cw_last]))
  slerp = Slerp([0.0, 1.0], key_rots)
  out: list[gtsam.Pose3] = []
  denom = float(init_last - init_first)
  for i in range(n_keyframes):
    alpha = (i - init_first) / denom
    if 0.0 <= alpha <= 1.0:
      r_cw = slerp([alpha])[0].as_matrix()
    else:
      rotvec = Rscp.from_matrix(r_cw_last).as_rotvec() * alpha
      r_cw = Rscp.from_rotvec(rotvec).as_matrix()
    t_cw = alpha * t_cw_last
    r_wc = r_cw.T
    t_wc = -r_wc @ t_cw
    out.append(gtsam.Pose3(gtsam.Rot3(r_wc), t_wc))
  return out


def body_poses_from_lens_poses(
  lens_poses: list[gtsam.Pose3],
  body_T_lens: gtsam.Pose3,
) -> list[gtsam.Pose3]:
  """``world_T_body = world_T_lens · body_T_lens⁻¹`` per keyframe."""
  inv = body_T_lens.inverse()
  return [p.compose(inv) for p in lens_poses]


def _triangulate_bearings(
  views: list[tuple[Float64Array, Float64Array, Float64Array]],
) -> Float64Array | None:
  """Multi-view DLT for (R, t world→cam, unit bearing) views → world point."""
  rows: list[Float64Array] = []
  for r, t, b in views:
    if b[2] <= 1e-6:
      continue
    x, y = b[0] / b[2], b[1] / b[2]
    p = np.hstack([r, t.reshape(3, 1)])
    rows.append(x * p[2] - p[0])
    rows.append(y * p[2] - p[1])
  if len(rows) < 4:
    return None
  _, _, vh = np.linalg.svd(np.array(rows))
  x_h = vh[-1]
  if abs(x_h[3]) < 1e-12:
    return None
  return np.asarray(x_h[:3] / x_h[3], dtype=np.float64)


def _reproj_bearing(
  r: Float64Array, t: Float64Array, x: Float64Array, b: Float64Array
) -> float:
  p = r @ x + t
  if p[2] <= 1e-6:
    return 1e9
  return float(np.hypot(p[0] / p[2] - b[0] / b[2], p[1] / p[2] - b[1] / b[2]))


def _pnp_bearings(
  pts: Float64Array, bears: Float64Array, min_points: int
) -> tuple[Float64Array, Float64Array] | None:
  """Robust PnP from 3D points + unit bearings → (R, t) world→cam."""
  front = bears[:, 2] > 1e-6
  if int(front.sum()) < min_points:
    return None
  pts, bears = pts[front], bears[front]
  norm = (bears[:, :2] / bears[:, 2:3]).astype(np.float64).reshape(-1, 1, 2)
  obj = pts.astype(np.float64).reshape(-1, 1, 3)
  ok, rvec, tvec, inl = cv2.solvePnPRansac(
    obj,
    norm,
    np.eye(3),
    np.zeros(4),
    reprojectionError=0.02,
    iterationsCount=1000,
    confidence=0.999,
    flags=cv2.SOLVEPNP_EPNP,
  )
  if not ok or inl is None or len(inl) < min_points:
    return None
  inl = inl.ravel()
  rvec, tvec = cv2.solvePnPRefineLM(
    obj[inl], norm[inl], np.eye(3), np.zeros(4), rvec, tvec
  )
  r, _ = cv2.Rodrigues(rvec)
  return r.astype(np.float64), tvec.reshape(3).astype(np.float64)


def incremental_sfm(
  bearings: dict[int, dict[int, Float64Array]],
  min_shared: int = 40,
) -> tuple[dict[int, gtsam.Pose3], dict[int, Float64Array]]:
  """``(lens_poses, landmark_map)`` of incremental monocular SfM, or empty dicts."""
  cv2.setRNGSeed(0)  # deterministic RANSAC for solvePnP/essential
  frames = sorted({k for tr in bearings.values() for k in tr})
  if len(frames) < 2:
    return {}, {}
  poses: dict[int, tuple[Float64Array, Float64Array]] = {}
  landmark_map: dict[int, Float64Array] = {}

  def triangulate(tid: int) -> None:
    views = [
      (poses[k][0], poses[k][1], bearings[tid][k]) for k in bearings[tid] if k in poses
    ]
    if len(views) < 2:
      return
    x = _triangulate_bearings(views)
    if x is None:
      return
    if sum(1 for r, t, _ in views if (r @ x + t)[2] > 0.05) < 2:
      return
    centers = [-r.T @ t for r, t, _ in views]
    if max_parallax(centers, x) < SFM_MIN_PARALLAX:
      return
    if np.median([_reproj_bearing(r, t, x, b) for r, t, b in views]) > SFM_MAX_REPROJ:
      return
    landmark_map[tid] = x

  first = frames[0]
  last: int | None = None
  for candidate in frames[1:]:
    shared = [
      tid for tid in bearings if first in bearings[tid] and candidate in bearings[tid]
    ]
    if len(shared) < min_shared:
      continue
    parallax = float(
      np.median(
        [
          np.arccos(np.clip(bearings[tid][first] @ bearings[tid][candidate], -1.0, 1.0))
          for tid in shared
        ]
      )
    )
    last = candidate
    if parallax >= SFM_INIT_PAIR_MIN_PARALLAX:
      break
  if last is None:
    return {}, {}
  shared = [tid for tid in bearings if first in bearings[tid] and last in bearings[tid]]
  r_cw, t_cw, _ = two_view_essential(
    np.array([bearings[tid][first] for tid in shared]),
    np.array([bearings[tid][last] for tid in shared]),
  )
  poses[first] = (np.eye(3), np.zeros(3))
  poses[last] = (r_cw, t_cw)
  for tid in shared:
    triangulate(tid)

  for _pass in range(SFM_PASSES):
    progress = False
    for k in frames:
      if k in poses:
        continue
      tids = [tid for tid in bearings if k in bearings[tid] and tid in landmark_map]
      if len(tids) < SFM_MIN_PNP_POINTS:
        continue
      pnp_pose = _pnp_bearings(
        np.array([landmark_map[tid] for tid in tids]),
        np.array([bearings[tid][k] for tid in tids]),
        SFM_MIN_PNP_POINTS,
      )
      if pnp_pose is None:
        continue
      poses[k] = pnp_pose
      progress = True
      for tid in bearings:
        if k in bearings[tid]:
          triangulate(tid)
    if not progress:
      break

  lens_poses = {
    k: gtsam.Pose3(gtsam.Rot3(r.T), -r.T @ t) for k, (r, t) in poses.items()
  }
  return lens_poses, landmark_map


# Cheirality bound on a unit bearing's z and on a view depth (m).
_MIN_CHEIRALITY_Z = 0.05


def triangulate_track_multiview(
  bearings: Float64Array,
  lens_rotations: Float64Array,
  lens_centers: Float64Array,
  min_landmark_depth: float = 0.15,
  max_landmark_depth: float = 5.0,
) -> Float64Array | None:
  """Least-squares point of one track's rays, or None when its depth checks fail."""
  if not (len(bearings) == len(lens_rotations) == len(lens_centers)):
    raise ValueError(
      "bearings/rotations/centers length mismatch: "
      f"{len(bearings)}/{len(lens_rotations)}/{len(lens_centers)}"
    )
  valid = bearings[:, 2] > _MIN_CHEIRALITY_Z
  if valid.sum() < 2:
    return None
  valid_rotations = lens_rotations[valid]
  valid_centers = lens_centers[valid]
  rays = np.einsum("nij,nj->ni", valid_rotations, bearings[valid])
  rays /= np.linalg.norm(rays, axis=1, keepdims=True)
  # Normal equations (Σ I - dᵢdᵢᵀ) X = Σ (I - dᵢdᵢᵀ) Cᵢ over centers Cᵢ, unit rays dᵢ.
  normal_matrix = len(rays) * np.eye(3) - rays.T @ rays
  rhs = valid_centers.sum(axis=0) - rays.T @ np.einsum("ni,ni->n", rays, valid_centers)
  try:
    xyz = np.linalg.solve(normal_matrix, rhs).astype(np.float64)
  except np.linalg.LinAlgError:
    return None
  if not np.all(np.isfinite(xyz)):
    return None
  # Per-view depth: the point's z in each lens frame.
  depths = np.einsum("ni,ni->n", valid_rotations[:, :, 2], xyz - valid_centers)
  good_depths = depths[depths >= _MIN_CHEIRALITY_Z]
  # Two views in front suffice; a fisheye landmark can sit near one frame's 90° horizon.
  if len(good_depths) < 2:
    return None
  median_depth = float(np.median(good_depths))
  if not min_landmark_depth <= median_depth <= max_landmark_depth:
    return None
  return xyz


def visual_inertial_align(
  body_poses: list[gtsam.Pose3],
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements],
  velocity_regularization: float = 1.0e-2,
  fixed_gravity: Float64Array | None = None,
  fixed_scale: float | None = None,
) -> Alignment:
  """Least-squares scale, gravity and velocities from keyframe NavState constraints."""
  n = len(body_poses)
  if n < 4:
    raise ValueError(f"visual_inertial_align needs ≥4 keyframes, got {n}")
  if len(preintegrated) != n - 1:
    raise ValueError(f"expected {n - 1} pim entries, got {len(preintegrated)}")
  dts = [float(pim.deltaTij()) for pim in preintegrated]

  n_unknowns = 3 * n + 4
  n_eq_data = 6 * (n - 1)
  n_eq_reg = 3 * n if velocity_regularization > 0 else 0
  A = np.zeros((n_eq_data + n_eq_reg, n_unknowns))
  b = np.zeros(n_eq_data + n_eq_reg)
  eye3 = np.eye(3)

  for i in range(n - 1):
    dt = dts[i]
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
    A[row_p : row_p + 3, 0] = p_visual_delta
    A[row_p : row_p + 3, 1:4] = -0.5 * dt * dt * eye3
    A[row_p : row_p + 3, 4 + 3 * i : 4 + 3 * i + 3] = -dt * eye3
    b[row_p : row_p + 3] = delta_p
    # Velocity equation: v_{i+1} − v_i − g·Δt = R_i·Δv_pim
    A[row_v : row_v + 3, 1:4] = -dt * eye3
    A[row_v : row_v + 3, 4 + 3 * i : 4 + 3 * i + 3] = -eye3
    A[row_v : row_v + 3, 4 + 3 * (i + 1) : 4 + 3 * (i + 1) + 3] = eye3
    b[row_v : row_v + 3] = delta_v

  if n_eq_reg:
    sqrt_lam = float(np.sqrt(velocity_regularization))
    for i in range(n):
      r = n_eq_data + 3 * i
      A[r : r + 3, 4 + 3 * i : 4 + 3 * i + 3] = sqrt_lam * eye3

  if fixed_gravity is not None:
    # A joint scale, gravity and velocity solve lets the scale collapse on slow motion.
    w_g = 1.0e3
    rows_g = np.zeros((3, n_unknowns))
    rows_g[:, 1:4] = w_g * eye3
    A = np.vstack([A, rows_g])
    b = np.concatenate([b, w_g * np.asarray(fixed_gravity, dtype=np.float64)])

  if fixed_scale is not None:
    # Pin the metric scale, for a seed that is already metric.
    w_s = 1.0e3
    row_s = np.zeros((1, n_unknowns))
    row_s[0, 0] = w_s
    A = np.vstack([A, row_s])
    b = np.concatenate([b, [w_s * fixed_scale]])

  x, *_ = np.linalg.lstsq(A, b, rcond=None)
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
      gtsam.Rot3(r @ p.rotation().matrix()),
      r @ (alignment.scale * np.asarray(p.translation(), dtype=np.float64)),
    )
    for p in body_poses
  ], rot
