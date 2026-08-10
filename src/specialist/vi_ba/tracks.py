"""A lens's tracks: frontend chaining and multi-view triangulation into landmarks."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import cv2
import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import MeiParams, mei_unproject
from specialist.vi_ba.configs import FrontendCfg
from specialist.vi_ba.features import (
  Frame,
  Rect,
  chain_tracks,
  dis_flow_track,
  essential_ransac_inliers,
  parallel_map,
)
from specialist.vi_ba.init import triangulate_track_multiview

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "filter_tracks_by_parallax",
  "run_lens_frontend",
  "triangulate_lens",
  "unproject_tracks",
]


def run_lens_frontend(
  frames: Sequence[np.ndarray],
  params: MeiParams,
  cfg: FrontendCfg,
  masks: list[list[Rect]] | None = None,
) -> tuple[list[Frame], dict[int, dict[int, Float64Array]]]:
  """One lens's tracks from DIS flow and essential RANSAC, outside ``masks``."""
  if masks is not None and len(masks) != len(frames):
    raise ValueError(f"masks length {len(masks)} != frames length {len(frames)}")
  feature_frames, matches_between = dis_flow_track(
    frames,
    max_corners=cfg.max_corners,
    masks=masks,
    gpu_flow=cfg.gpu_flow,
    flow_scale=cfg.flow_scale,
  )

  def ransac_pair(k: int) -> npt.NDArray[np.int64]:
    m = matches_between[k - 1]
    if m.shape[0] == 0:
      return np.empty((0, 2), dtype=np.int64)
    pts_a = feature_frames[k - 1].keypoints[m[:, 0]]
    pts_b = feature_frames[k].keypoints[m[:, 1]]
    bears_a = mei_unproject(pts_a, params)
    bears_b = mei_unproject(pts_b, params)
    # The cv2 RNG is thread-local, so a per-pair seed keeps RANSAC deterministic.
    cv2.setRNGSeed(k)
    inliers = essential_ransac_inliers(
      pts_a,
      pts_b,
      bears_a,
      bears_b,
      threshold_px=cfg.ransac_threshold,
      fx=params.fx,
      max_half_angle=cfg.feature_max_half_angle,
    )
    return np.asarray(m[inliers], dtype=np.int64)

  filtered = parallel_map(ransac_pair, range(1, len(feature_frames)))
  tracks = chain_tracks(feature_frames, filtered)
  n_kp = sum(int(len(f.keypoints)) for f in feature_frames)
  n_match = sum(int(m.shape[0]) for m in matches_between)
  n_in = sum(int(f.shape[0]) for f in filtered)
  print(
    f"  [frontend funnel] keypoints={n_kp} matched={n_match} "
    f"ransac_inliers={n_in} chained_tracks={len(tracks)}"
  )
  return feature_frames, tracks


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
