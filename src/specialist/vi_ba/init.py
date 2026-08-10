"""Trajectory seeding: multi-view triangulation and visual-inertial alignment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import gtsam
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as Rscp
from scipy.spatial.transform import Slerp

from specialist.vi_ba.geometry import max_parallax

Float64Array = npt.NDArray[np.float64]

_SFM_MIN_PARALLAX = float(np.deg2rad(0.6))  # to triangulate a track
_SFM_INIT_PAIR_MIN_PARALLAX = float(np.deg2rad(2.0))  # the bootstrap pair must span
_SFM_MIN_PNP_POINTS = 6  # inliers to register a keyframe by PnP
_SFM_MAX_REPROJ = 0.01  # median triangulation residual on the normalized plane
_SFM_PASSES = 8  # sequential registration passes


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


# Keyframes the init-pair base may slide over when the first has too few tracks.
_INIT_BASE_SLIDE = 32


def _widest_pair_from(
  tracks: dict[int, dict[int, Float64Array]],
  first: int,
  n_keyframes: int,
  min_shared: int,
) -> tuple[int, int]:
  """``(K, shared)`` of the widest ``(first, K)`` pair, else of the one sharing most."""
  shared_by_last: Counter[int] = Counter()
  for obs in tracks.values():
    if first in obs:
      shared_by_last.update(kf for kf in obs if kf != first)

  for last in range(n_keyframes - 1, first, -1):
    if shared_by_last[last] >= min_shared:
      return last, shared_by_last[last]
  best_last, best_shared = min(first + 1, n_keyframes - 1), 0
  for last in range(first + 1, n_keyframes):
    if shared_by_last[last] > best_shared:
      best_last, best_shared = last, shared_by_last[last]
  return best_last, best_shared


def pick_init_pair(
  tracks: dict[int, dict[int, Float64Array]],
  n_keyframes: int,
  min_shared: int,
) -> tuple[int, int, int]:
  """Widest ``(first, K, n_shared)`` with ≥``min_shared`` tracks, else the best seen."""
  best = (0, min(1, n_keyframes - 1), 0)
  for first in range(min(_INIT_BASE_SLIDE, max(1, n_keyframes - 1))):
    last, shared = _widest_pair_from(tracks, first, n_keyframes, min_shared)
    if shared >= min_shared:
      return first, last, shared
    if shared > best[2]:
      best = (first, last, shared)
  return best


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


def _pnp_bearings(
  pts: Float64Array, bears: Float64Array
) -> tuple[Float64Array, Float64Array] | None:
  """Robust PnP from 3D points + unit bearings → (R, t) world→cam."""
  front = bears[:, 2] > 1e-6
  if int(front.sum()) < _SFM_MIN_PNP_POINTS:
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
  if not ok or inl is None or len(inl) < _SFM_MIN_PNP_POINTS:
    return None
  inl = inl.ravel()
  rvec, tvec = cv2.solvePnPRefineLM(
    obj[inl], norm[inl], np.eye(3), np.zeros(4), rvec, tvec
  )
  r, _ = cv2.Rodrigues(rvec)
  return r.astype(np.float64), tvec.reshape(3).astype(np.float64)


def _bootstrap_pair(
  bearings: dict[int, dict[int, Float64Array]],
  tracks_by_kf: dict[int, list[int]],
  frames: list[int],
  min_shared: int,
) -> tuple[int, int] | None:
  """Nearest pair with ``min_shared`` tracks and enough parallax, else the widest."""
  fallback: tuple[int, int] | None = None
  for base, first in enumerate(frames[:_INIT_BASE_SLIDE]):
    widest: int | None = None
    for candidate in frames[base + 1 :]:
      shared = [tid for tid in tracks_by_kf[first] if candidate in bearings[tid]]
      if len(shared) < min_shared:
        continue
      widest = candidate
      parallax = float(
        np.median(
          [
            np.arccos(
              np.clip(bearings[tid][first] @ bearings[tid][candidate], -1.0, 1.0)
            )
            for tid in shared
          ]
        )
      )
      if parallax >= _SFM_INIT_PAIR_MIN_PARALLAX:
        return first, candidate
    if widest is not None and fallback is None:
      fallback = (first, widest)
  return fallback


def incremental_sfm(
  bearings: dict[int, dict[int, Float64Array]],
  min_shared: int = 40,
) -> tuple[dict[int, gtsam.Pose3], dict[int, Float64Array]]:
  """SfM camera poses and landmarks in the bootstrap frame and scale, or empty dicts."""
  cv2.setRNGSeed(0)  # deterministic RANSAC for solvePnP/essential
  frames = sorted({k for tr in bearings.values() for k in tr})
  if len(frames) < 2:
    return {}, {}
  # Inverted index: keyframe -> track ids, in ``bearings`` iteration order.
  tracks_by_kf: dict[int, list[int]] = {k: [] for k in frames}
  for tid, obs in bearings.items():
    for k in obs:
      tracks_by_kf[k].append(tid)
  # Array mirrors of the pose state plus per-track observation arrays.
  frame_index = {k: i for i, k in enumerate(frames)}
  rotations = np.zeros((len(frames), 3, 3))
  translations = np.zeros((len(frames), 3))
  registered = np.zeros(len(frames), dtype=bool)
  track_frame_rows = {
    tid: np.asarray([frame_index[k] for k in obs], dtype=np.int64)
    for tid, obs in bearings.items()
  }
  track_bearing_rows = {
    tid: np.asarray(list(obs.values()), dtype=np.float64)
    for tid, obs in bearings.items()
  }
  poses: dict[int, tuple[Float64Array, Float64Array]] = {}
  landmark_map: dict[int, Float64Array] = {}

  def register(k: int, r_cw: Float64Array, t_cw: Float64Array) -> None:
    poses[k] = (r_cw, t_cw)
    i = frame_index[k]
    rotations[i] = r_cw
    translations[i] = t_cw
    registered[i] = True

  def triangulate(tid: int) -> None:
    rows_idx = track_frame_rows[tid]
    in_map = registered[rows_idx]
    if int(in_map.sum()) < 2:
      return
    r_cw = rotations[rows_idx[in_map]]
    t_cw = translations[rows_idx[in_map]]
    b = track_bearing_rows[tid][in_map]
    # Multi-view DLT over the views whose bearing is safely in front.
    z_ok = b[:, 2] > 1e-6
    if int(z_ok.sum()) < 2:
      return
    x_n = b[z_ok, 0] / b[z_ok, 2]
    y_n = b[z_ok, 1] / b[z_ok, 2]
    p = np.concatenate([r_cw[z_ok], t_cw[z_ok, :, None]], axis=2)
    rows = np.empty((2 * len(x_n), 4))
    rows[0::2] = x_n[:, None] * p[:, 2, :] - p[:, 0, :]
    rows[1::2] = y_n[:, None] * p[:, 2, :] - p[:, 1, :]
    # The SVD fails on non-finite rows from bearings at the MEI valid-domain edge.
    finite = np.isfinite(rows).all(axis=1)
    if int(finite.sum()) < 4:
      return
    rows = rows[finite]
    # Only the last right-singular vector is used; skip the full U factor.
    _, _, vh = np.linalg.svd(rows, full_matrices=False)
    x_h = vh[-1]
    if abs(x_h[3]) < 1e-12:
      return
    x = np.asarray(x_h[:3] / x_h[3], dtype=np.float64)
    p_cam = np.einsum("vij,j->vi", r_cw, x) + t_cw
    if int((p_cam[:, 2] > 0.05).sum()) < 2:
      return
    centers = -np.einsum("vji,vj->vi", r_cw, t_cw)
    if max_parallax(centers, x) < _SFM_MIN_PARALLAX:
      return
    residual = np.full(len(p_cam), 1.0e9)
    front = p_cam[:, 2] > 1e-6
    residual[front] = np.hypot(
      p_cam[front, 0] / p_cam[front, 2] - b[front, 0] / b[front, 2],
      p_cam[front, 1] / p_cam[front, 2] - b[front, 1] / b[front, 2],
    )
    if float(np.median(residual)) > _SFM_MAX_REPROJ:
      return
    landmark_map[tid] = x

  pair = _bootstrap_pair(bearings, tracks_by_kf, frames, min_shared)
  if pair is None:
    return {}, {}
  first, last = pair
  shared = [tid for tid in tracks_by_kf[first] if last in bearings[tid]]
  r_cw, t_cw, _ = two_view_essential(
    np.array([bearings[tid][first] for tid in shared]),
    np.array([bearings[tid][last] for tid in shared]),
  )
  register(first, np.eye(3), np.zeros(3))
  register(last, r_cw, t_cw)
  for tid in shared:
    triangulate(tid)

  for _pass in range(_SFM_PASSES):
    progress = False
    for k in frames:
      if k in poses:
        continue
      tids = [tid for tid in tracks_by_kf[k] if tid in landmark_map]
      if len(tids) < _SFM_MIN_PNP_POINTS:
        continue
      pnp_pose = _pnp_bearings(
        np.array([landmark_map[tid] for tid in tids]),
        np.array([bearings[tid][k] for tid in tids]),
      )
      if pnp_pose is None:
        continue
      register(k, *pnp_pose)
      progress = True
      for tid in tracks_by_kf[k]:
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
