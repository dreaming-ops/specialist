"""A lens's tracks: frontend chaining, revisit matching, triangulation and caps."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import cv2
import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import MeiParams, mei_unproject
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.configs import FrontendCfg
from specialist.vi_ba.features import (
  Frame,
  Rect,
  chain_tracks,
  dis_flow_track,
  essential_ransac_inliers,
  match_pair,
  parallel_map,
)
from specialist.vi_ba.geometry import translation
from specialist.vi_ba.init import triangulate_track_multiview

Float64Array = npt.NDArray[np.float64]

_REVISIT_N_NEIGHBORS = 8  # spatial-neighbor revisit candidates per keyframe
_REVISIT_MATCH_RATIO = 0.8  # Lowe ratio for the revisit ORB matches
_REVISIT_MIN_INLIERS = 10  # matches a revisit pair must keep to count
_REVISIT_MAX_DIST = 0.1  # meters between keyframes for a revisit candidate
_REVISIT_MIN_KF_GAP = 15  # keyframes a revisit must span, or it is just motion
_REVISIT_RANSAC_THRESHOLD = 4.5  # homography RANSAC reprojection tolerance, px
_REVISIT_MIN_VIEWS = 3  # keyframes a union-found track must span to be kept

__all__ = [
  "add_revisit_landmarks",
  "bearings_by_track",
  "cap_landmarks",
  "filter_tracks_by_parallax",
  "revisit_tracks",
  "run_lens_frontend",
  "subsample_obs",
  "triangulate_lens",
  "unproject_tracks",
]


def run_lens_frontend(
  frames: Sequence[np.ndarray],
  params: MeiParams,
  cfg: FrontendCfg,
  masks: list[list[Rect]] | None = None,
  *,
  label: str | None = None,
  band: int | None = None,
) -> tuple[list[Float64Array], dict[int, dict[int, Float64Array]]]:
  """One lens's per-frame keypoints and tracks from DIS flow and essential RANSAC."""
  if masks is not None and len(masks) != len(frames):
    raise ValueError(f"masks length {len(masks)} != frames length {len(frames)}")
  keypoints, matches_between = dis_flow_track(
    frames,
    max_corners=cfg.max_corners,
    masks=masks,
    gpu_flow=cfg.gpu_flow,
    flow_scale=cfg.flow_scale,
    label=label,
    band=band,
  )

  def ransac_pair(k: int) -> npt.NDArray[np.int64]:
    m = matches_between[k - 1]
    if m.shape[0] == 0:
      return np.empty((0, 2), dtype=np.int64)
    # The cv2 RNG is thread-local, so a per-pair seed keeps RANSAC deterministic.
    cv2.setRNGSeed(k)
    inliers = essential_ransac_inliers(
      mei_unproject(keypoints[k - 1][m[:, 0]], params),
      mei_unproject(keypoints[k][m[:, 1]], params),
      threshold_px=cfg.ransac_threshold,
      fx=params.fx,
      max_half_angle=cfg.feature_max_half_angle,
    )
    return np.asarray(m[inliers], dtype=np.int64)

  filtered = parallel_map(ransac_pair, range(1, len(keypoints)))
  tracks = chain_tracks(keypoints, filtered)
  n_kp = sum(len(k) for k in keypoints)
  n_match = sum(int(m.shape[0]) for m in matches_between)
  n_in = sum(int(f.shape[0]) for f in filtered)
  scope = f"{label}/frontend" if label else "frontend"
  print(
    f"  [{scope}] funnel keypoints={n_kp} matched={n_match} "
    f"ransac_inliers={n_in} chained_tracks={len(tracks)}"
  )
  return keypoints, tracks


def unproject_tracks(
  tracks: dict[int, dict[int, Float64Array]],
  params: MeiParams,
  min_obs: int,
) -> Iterator[tuple[int, list[int], Float64Array]]:
  """Yield ``(tid, sorted kfs, bearings)`` per track with at least ``min_obs`` views."""
  spans: list[tuple[int, list[int], int, int]] = []
  flat_pixels: list[Float64Array] = []
  for tid, obs in tracks.items():
    if len(obs) < min_obs:
      continue
    kfs = sorted(obs)
    start = len(flat_pixels)
    flat_pixels.extend(obs[k] for k in kfs)
    spans.append((tid, kfs, start, len(flat_pixels)))
  if not spans:
    return
  bearings = mei_unproject(np.asarray(flat_pixels, dtype=np.float64), params)
  for tid, kfs, start, end in spans:
    yield tid, kfs, bearings[start:end]


def filter_tracks_by_parallax(
  tracks: dict[int, dict[int, Float64Array]],
  params: MeiParams,
  min_parallax: float,
) -> tuple[dict[int, dict[int, Float64Array]], dict[int, dict[int, Float64Array]]]:
  """``(kept, rejected)`` by max bearing-angle change against ``min_parallax``."""
  cos_min = float(np.cos(min_parallax))
  kept: dict[int, dict[int, Float64Array]] = {}
  rejected = {tid: obs for tid, obs in tracks.items() if len(obs) < 2}
  for tid, _, bearings in unproject_tracks(tracks, params, min_obs=2):
    dots = np.clip(bearings @ bearings.T, -1.0, 1.0)
    if float(dots.min()) <= cos_min:
      kept[tid] = tracks[tid]
    else:
      rejected[tid] = tracks[tid]
  return kept, rejected


def triangulate_lens(
  tracks: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  body_T_lens: gtsam.Pose3,
  params: MeiParams,
  min_track_length: int,
  max_landmark_depth: float,
  min_landmark_depth: float = 0.15,
) -> dict[int, Float64Array]:
  """Triangulate tracks; skip any under ``min_track_length`` or out of depth bounds."""
  if not tracks:
    return {}
  lens_poses = [pose.compose(body_T_lens) for pose in body_poses]
  lens_rotations = np.stack([pose.rotation().matrix() for pose in lens_poses])
  lens_centers = np.stack(
    [np.asarray(pose.translation(), dtype=np.float64) for pose in lens_poses]
  )
  out: dict[int, Float64Array] = {}
  for tid, kfs, bearings in unproject_tracks(tracks, params, min_obs=min_track_length):
    xyz = triangulate_track_multiview(
      bearings,
      lens_rotations[kfs],
      lens_centers[kfs],
      min_landmark_depth=min_landmark_depth,
      max_landmark_depth=max_landmark_depth,
    )
    if xyz is not None:
      out[tid] = xyz
  return out


def bearings_by_track(
  tracks: dict[int, dict[int, Float64Array]], params: MeiParams
) -> dict[int, dict[int, Float64Array]]:
  """``{tid: {kf: unit bearing}}`` for the visual-odometry front-end."""
  return {
    tid: dict(zip(kfs, bearings, strict=True))
    for tid, kfs, bearings in unproject_tracks(tracks, params, min_obs=1)
  }


def subsample_obs(
  track: dict[int, Float64Array], max_obs: int
) -> dict[int, Float64Array]:
  """At most ``max_obs`` observations: both endpoints and evenly spaced ones between."""
  kfs = sorted(track)
  if max_obs <= 0 or len(kfs) <= max_obs:
    return track
  idx = np.unique(np.linspace(0, len(kfs) - 1, max_obs).round().astype(int))
  return {kfs[i]: track[kfs[i]] for i in idx}


def cap_landmarks(
  points: dict[int, Float64Array],
  tracks: dict[int, dict[int, Float64Array]],
  cap: int,
) -> dict[int, Float64Array]:
  """Keep at most ``cap`` landmarks, preferring the longest tracks."""
  if cap <= 0 or len(points) <= cap:
    return points
  keep = set(sorted(points, key=lambda t: -len(tracks.get(t, {})))[:cap])
  return {t: p for t, p in points.items() if t in keep}


def revisit_tracks(
  orb: list[Frame],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  base_track_id: int,
  cfg: FrontendCfg,
) -> dict[int, dict[int, Float64Array]]:
  """Tracks that ORB loop closure recovers between close, non-adjacent keyframes."""
  centers = np.asarray([translation(p.compose(lens.body_T_lens)) for p in body_poses])
  n = len(orb)
  parent: dict[tuple[int, int], tuple[int, int]] = {}

  def find(x: tuple[int, int]) -> tuple[int, int]:
    parent.setdefault(x, x)
    root = x
    while parent[root] != root:
      root = parent[root]
    while parent[x] != root:
      parent[x], x = root, parent[x]
    return root

  pairs: list[tuple[int, int]] = []
  for i in range(n):
    if len(orb[i].keypoints) == 0:
      continue
    dist = np.linalg.norm(centers - centers[i], axis=1)
    cand = [
      int(j)
      for j in np.argsort(dist)
      if j > i and (j - i) > _REVISIT_MIN_KF_GAP and dist[j] <= _REVISIT_MAX_DIST
    ][:_REVISIT_N_NEIGHBORS]
    pairs.extend((i, j) for j in cand)

  def verify_pair(i: int, j: int) -> npt.NDArray[np.int64] | None:
    m = match_pair(orb[i], orb[j], _REVISIT_MATCH_RATIO)
    if len(m) < _REVISIT_MIN_INLIERS:
      return None
    pa = orb[i].keypoints[m[:, 0]]
    pb = orb[j].keypoints[m[:, 1]]
    # cv2's RNG is thread-local, so a per-pair seed keeps RANSAC deterministic.
    cv2.setRNGSeed(i * n + j)
    _, mask = cv2.findHomography(pa, pb, cv2.RANSAC, _REVISIT_RANSAC_THRESHOLD)
    if mask is None:
      return None
    inl = mask.ravel().astype(bool)
    if int(inl.sum()) < _REVISIT_MIN_INLIERS:
      return None
    return np.asarray(m[inl], dtype=np.int64)

  verified = parallel_map(verify_pair, [i for i, _ in pairs], [j for _, j in pairs])
  for (i, j), matches in zip(pairs, verified, strict=True):
    if matches is None:
      continue
    for a_idx, b_idx in matches:
      ra = find((i, int(a_idx)))
      rb = find((j, int(b_idx)))
      if ra != rb:
        parent[ra] = rb

  comps: dict[tuple[int, int], dict[int, int]] = {}
  for node in list(parent):
    comps.setdefault(find(node), {})[node[0]] = node[1]
  tracks: dict[int, dict[int, Float64Array]] = {}
  tid = base_track_id + 1
  for by_kf in comps.values():
    if len(by_kf) < _REVISIT_MIN_VIEWS:
      continue
    # Pixel coordinates, as the optical-flow tracks carry.
    tracks[tid] = {
      kf: np.asarray(orb[kf].keypoints[kp], dtype=np.float64)
      for kf, kp in by_kf.items()
    }
    tid += 1
  return tracks


def add_revisit_landmarks(
  points: dict[int, Float64Array],
  kept: dict[int, dict[int, Float64Array]],
  revisits: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  cfg: FrontendCfg,
  *,
  slot: str,
  anchor_kfs: set[int] | None = None,
) -> None:
  """Add ``revisits`` to ``points``, ``kept``; ``anchor_kfs`` gates triangulation."""
  if anchor_kfs:
    tri_tracks = {
      t: {k: v for k, v in obs.items() if k in anchor_kfs}
      for t, obs in revisits.items()
    }
  else:
    tri_tracks = revisits
  revisit_pts = triangulate_lens(
    tri_tracks,
    body_poses,
    lens.body_T_lens,
    lens.params,
    cfg.min_track_length,
    max_landmark_depth=cfg.max_landmark_depth,
  )
  points.update(revisit_pts)
  kept.update(revisits)
  print(f"  [{slot}/revisit] +{len(revisit_pts)} landmarks from {len(revisits)} tracks")
