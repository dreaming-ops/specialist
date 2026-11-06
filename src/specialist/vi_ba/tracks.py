"""A lens's tracks: frontend chaining, revisit matching, triangulation and caps."""

from collections.abc import Callable, Iterator, Sequence

import cv2
import gtsam
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from specialist_camera import MeiParams, mei_unproject

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.configs import FrontendCfg
from specialist.vi_ba.features import (
  Frame,
  chain_tracks,
  dis_flow_track,
  essential_inliers,
  parallel_map,
  prepare_essential_rows,
)
from specialist.vi_ba.geometry import translation
from specialist.vi_ba.gpu_match import match_pairs
from specialist.vi_ba.init import triangulate_tracks_batched

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
  "cap_landmarks",
  "filter_tracks_by_parallax",
  "revisit_tracks",
  "run_lens_frontend",
  "split_tracks_to_window",
  "subsample_obs",
  "triangulate_lens",
  "unproject_tracks",
]


def run_lens_frontend(
  frames: Sequence[np.ndarray],
  params: MeiParams,
  cfg: FrontendCfg,
  *,
  label: str | None = None,
  on_frame: Callable[[int, np.ndarray], None] | None = None,
  band: int | None = None,
) -> tuple[list[Float64Array], dict[int, dict[int, Float64Array]]]:
  """One lens's per-frame keypoints and tracks from DIS flow and essential RANSAC."""
  keypoints, matches_between = dis_flow_track(
    frames,
    max_corners=cfg.max_corners,
    gpu_flow=cfg.gpu_flow,
    flow_scale=cfg.flow_scale,
    label=label,
    on_frame=on_frame,
    band=band,
  )

  sizes = np.asarray([m.shape[0] for m in matches_between], dtype=np.int64)
  bounds = np.concatenate(([0], np.cumsum(sizes)))
  bearings: list[Float64Array] = []
  for side in (0, 1):
    pixels = [
      keypoints[k - 1 + side][m[:, side]]
      for k, m in enumerate(matches_between, start=1)
      if m.shape[0]
    ]
    bearings.append(
      mei_unproject(np.concatenate(pixels), params) if pixels else np.empty((0, 3))
    )

  in_fov, normalized_a, normalized_b = prepare_essential_rows(
    bearings[0], bearings[1], cfg.feature_max_half_angle
  )
  threshold = float(cfg.ransac_threshold / params.fx)

  def ransac_pair(k: int) -> npt.NDArray[np.int64]:
    m = matches_between[k - 1]
    if m.shape[0] == 0:
      return np.empty((0, 2), dtype=np.int64)
    first, last = bounds[k - 1], bounds[k]
    # The cv2 RNG is thread-local, so a per-pair seed keeps RANSAC deterministic.
    cv2.setRNGSeed(k)
    inliers = essential_inliers(
      in_fov[first:last],
      normalized_a[first:last],
      normalized_b[first:last],
      threshold,
    )
    return np.asarray(m[inliers], dtype=np.int64)

  # Each pair's RANSAC is a long native call; the map takes half the cores.
  filtered = parallel_map(ransac_pair, range(1, len(keypoints)), share=2)
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


# Elements of pairwise-dot matrices held at once by the parallax filter.
_PARALLAX_SLAB = 20_000_000


def filter_tracks_by_parallax(
  tracks: dict[int, dict[int, Float64Array]],
  params: MeiParams,
  min_parallax: float,
) -> tuple[dict[int, dict[int, Float64Array]], dict[int, dict[int, Float64Array]]]:
  """``(kept, rejected)`` by max bearing-angle change against ``min_parallax``."""
  cos_min = float(np.cos(min_parallax))
  kept: dict[int, dict[int, Float64Array]] = {}
  rejected = {tid: obs for tid, obs in tracks.items() if len(obs) < 2}
  items = list(unproject_tracks(tracks, params, min_obs=2))
  # Tracks of one length take their pairwise dots in one stacked product.
  by_length: dict[int, list[int]] = {}
  for i, (_, kfs, _) in enumerate(items):
    by_length.setdefault(len(kfs), []).append(i)
  wide = np.zeros(len(items), dtype=bool)
  for n, members in by_length.items():
    slab = max(1, _PARALLAX_SLAB // (n * n))
    for begin in range(0, len(members), slab):
      chunk = members[begin : begin + slab]
      stack = np.stack([items[i][2] for i in chunk])
      dots = np.clip(np.matmul(stack, np.swapaxes(stack, 1, 2)), -1.0, 1.0)
      wide[chunk] = dots.min(axis=(1, 2)) <= cos_min
  for i, (tid, _, _) in enumerate(items):
    (kept if wide[i] else rejected)[tid] = tracks[tid]
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
  """Triangulate every track whose median view depth lies within the depth bounds."""
  if not tracks:
    return {}
  lens_poses = [pose.compose(body_T_lens) for pose in body_poses]
  lens_rotations = np.stack([pose.rotation().matrix() for pose in lens_poses])
  lens_centers = np.stack([translation(pose) for pose in lens_poses])
  tids: list[int] = []
  kf_rows: list[int] = []
  track_rows: list[int] = []
  bearing_rows: list[Float64Array] = []
  for tid, kfs, bearings in unproject_tracks(tracks, params, min_obs=min_track_length):
    track_rows.extend([len(tids)] * len(kfs))
    tids.append(tid)
    kf_rows.extend(kfs)
    bearing_rows.append(bearings)
  if not tids:
    return {}
  kf = np.asarray(kf_rows, dtype=np.int64)
  points, passed = triangulate_tracks_batched(
    np.concatenate(bearing_rows),
    np.asarray(track_rows, dtype=np.int64),
    len(tids),
    lens_rotations[kf],
    lens_centers[kf],
    min_landmark_depth=min_landmark_depth,
    max_landmark_depth=max_landmark_depth,
  )
  return {tids[i]: points[i] for i in np.flatnonzero(passed)}


def split_tracks_to_window(
  points: dict[int, Float64Array],
  tracks: dict[int, dict[int, Float64Array]],
  window: int,
) -> tuple[dict[int, Float64Array], dict[int, dict[int, Float64Array]]]:
  """Cut each track into runs of at most ``window`` keyframes, each its own landmark."""
  if window <= 0:
    return points, tracks
  out_points: dict[int, Float64Array] = {}
  out_tracks: dict[int, dict[int, Float64Array]] = {}
  next_id = max(tracks, default=-1) + 1
  for tid, obs in tracks.items():
    for i, run in enumerate(_runs_within(sorted(obs), window)):
      new_id = tid if i == 0 else next_id
      if i:
        next_id += 1
      out_points[new_id] = points[tid]
      out_tracks[new_id] = {kf: obs[kf] for kf in run}
  return out_points, out_tracks


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


def _revisit_candidates(
  centers: Float64Array, populated: npt.NDArray[np.bool_], max_gap: int
) -> list[tuple[int, int]]:
  """``(i, j)`` pairs of each populated keyframe and its nearest later revisits."""
  pairs: list[tuple[int, int]] = []
  n = len(centers)
  for i in map(int, np.flatnonzero(populated)):
    lo, hi = i + _REVISIT_MIN_KF_GAP + 1, min(n, i + max_gap + 1)
    if lo >= hi:
      continue
    dist = np.linalg.norm(centers[lo:hi] - centers[i], axis=1)
    near = np.flatnonzero(dist <= _REVISIT_MAX_DIST)
    order = near[np.argsort(dist[near], kind="stable")][:_REVISIT_N_NEIGHBORS]
    pairs.extend((i, lo + int(j)) for j in order)
  return pairs


def revisit_tracks(
  orb: list[Frame],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  base_track_id: int,
  window: int = 0,
) -> dict[int, dict[int, Float64Array]]:
  """Revisit tracks between close keyframes at most ``window`` apart, 0 for any."""
  centers = np.asarray([translation(p.compose(lens.body_T_lens)) for p in body_poses])
  n = len(orb)
  max_gap = window if window > 0 else n

  pairs = _revisit_candidates(
    centers, np.asarray([len(f.keypoints) > 0 for f in orb]), max_gap
  )
  matched = match_pairs(orb, pairs, _REVISIT_MATCH_RATIO)

  def verify_pair(p: int) -> npt.NDArray[np.int64] | None:
    matches = matched[p]
    if len(matches) < _REVISIT_MIN_INLIERS:
      return None
    i, j = pairs[p]
    # A per-pair seed keeps the thread-local RANSAC deterministic.
    cv2.setRNGSeed(i * n + j)
    _, mask = cv2.findHomography(
      orb[i].keypoints[matches[:, 0]],
      orb[j].keypoints[matches[:, 1]],
      cv2.USAC_DEFAULT,
      _REVISIT_RANSAC_THRESHOLD,
    )
    if mask is None:
      return None
    inl = mask.ravel().astype(bool)
    if int(inl.sum()) < _REVISIT_MIN_INLIERS:
      return None
    return np.asarray(matches[inl], dtype=np.int64)

  verified = parallel_map(verify_pair, range(len(pairs)))
  edges = [
    (i, j, matches)
    for (i, j), matches in zip(pairs, verified, strict=True)
    if matches is not None
  ]
  tracks: dict[int, dict[int, Float64Array]] = {}
  tid = base_track_id + 1
  components = _matched_components(edges, [len(f.keypoints) for f in orb])
  for by_kf in components:
    # Each run of keyframes within the window is its own track.
    for run in _runs_within(sorted(by_kf), max_gap):
      if len(run) < _REVISIT_MIN_VIEWS:
        continue
      # Pixel coordinates, as the optical-flow tracks carry.
      tracks[tid] = {
        kf: np.asarray(orb[kf].keypoints[by_kf[kf]], dtype=np.float64) for kf in run
      }
      tid += 1
  return tracks


def _matched_components(
  edges: list[tuple[int, int, npt.NDArray[np.int64]]],
  n_keypoints: list[int],
) -> list[dict[int, int]]:
  """Verified-match components as ``{keyframe: keypoint}``, by first appearance."""
  if not edges:
    return []
  first_index = np.concatenate(([0], np.cumsum(n_keypoints)[:-1]))
  a = np.concatenate([first_index[i] + m[:, 0] for i, _, m in edges])
  b = np.concatenate([first_index[j] + m[:, 1] for _, j, m in edges])
  sorted_nodes, appearance = np.unique(
    np.stack([a, b], axis=1).ravel(), return_index=True
  )
  nodes = sorted_nodes[np.argsort(appearance)]
  graph = sp.coo_matrix(
    (
      np.ones(len(a), dtype=np.int8),
      (np.searchsorted(sorted_nodes, a), np.searchsorted(sorted_nodes, b)),
    ),
    shape=(len(sorted_nodes), len(sorted_nodes)),
  ).tocsr()
  _, label = connected_components(graph, directed=False)
  keyframe = np.searchsorted(first_index, nodes, side="right") - 1
  keypoint = nodes - first_index[keyframe]
  by_component: dict[int, dict[int, int]] = {}
  for comp, kf, kp in zip(
    label[np.searchsorted(sorted_nodes, nodes)].tolist(),
    keyframe.tolist(),
    keypoint.tolist(),
    strict=True,
  ):
    by_component.setdefault(comp, {})[kf] = kp
  return list(by_component.values())


def _runs_within(kfs: list[int], span: int) -> list[list[int]]:
  """Ascending ``kfs`` cut into runs whose first and last are at most ``span`` apart."""
  runs: list[list[int]] = []
  for kf in kfs:
    if runs and kf - runs[-1][0] <= span:
      runs[-1].append(kf)
    else:
      runs.append([kf])
  return runs


def add_revisit_landmarks(
  points: dict[int, Float64Array],
  kept: dict[int, dict[int, Float64Array]],
  revisits: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  cfg: FrontendCfg,
  *,
  slot: str,
  anchor_kfs: set[int],
) -> None:
  """Add ``revisits`` to ``kept``, triangulated from ``anchor_kfs`` into ``points``."""
  tri_tracks = {
    t: {k: v for k, v in obs.items() if k in anchor_kfs} for t, obs in revisits.items()
  }
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
