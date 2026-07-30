"""Visual frontend: DIS flow tracking, essential RANSAC, and revisit-matching ORB."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

import cv2
import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]
_FlowPair = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
_R = TypeVar("_R")

__all__ = [
  "Frame",
  "chain_tracks",
  "detect_orb",
  "dis_flow_track",
  "essential_ransac_inliers",
  "match_pair",
  "parallel_map",
]


@dataclass(frozen=True)
class Frame:
  """One keyframe's 2D keypoints and their descriptors."""

  keypoints: Float64Array
  descriptors: np.ndarray


def parallel_map(fn: Callable[..., _R], *iterables: Iterable[object]) -> list[_R]:
  """Ordered ``ThreadPoolExecutor.map`` for native work that releases the GIL."""
  columns = [list(it) for it in iterables]
  if len({len(column) for column in columns}) > 1:
    raise ValueError("parallel_map iterables must have equal length")
  if not columns[0]:
    return []
  workers = min(len(columns[0]), max(4, (os.cpu_count() or 8) // 8))
  with ThreadPoolExecutor(max_workers=workers) as executor:
    return list(executor.map(fn, *columns))


def _bilinear_sample_flow(
  flow: npt.NDArray[np.float32], xy: Float64Array
) -> Float64Array:
  """Bilinear-interpolated dense-flow lookup at subpixel positions ``xy``."""
  h, w = flow.shape[:2]
  if xy.shape[0] == 0:
    return np.empty((0, 2))
  x = np.clip(xy[:, 0], 0.0, float(w) - 1.001)
  y = np.clip(xy[:, 1], 0.0, float(h) - 1.001)
  x0 = x.astype(np.int64)
  y0 = y.astype(np.int64)
  fx = (x - x0)[:, None]
  fy = (y - y0)[:, None]
  f00 = flow[y0, x0].astype(np.float64)
  f01 = flow[y0, x0 + 1].astype(np.float64)
  f10 = flow[y0 + 1, x0].astype(np.float64)
  f11 = flow[y0 + 1, x0 + 1].astype(np.float64)
  return (
    (1.0 - fx) * (1.0 - fy) * f00
    + fx * (1.0 - fy) * f01
    + (1.0 - fx) * fy * f10
    + fx * fy * f11
  )


_PRECOMPUTE_CONCURRENCY = 1

# DIS variational-refinement iterations: the subpixel polish and most of DIS runtime.
_DIS_VARIATIONAL_ITERS = 10
# The backward flow feeds only the forward-backward gate, never a track coordinate.
_DIS_BACKWARD_VARIATIONAL_ITERS = 2


def set_precompute_concurrency(n: int) -> None:
  """Set how many cameras prepare at once so the flow precompute shrinks its pool."""
  global _PRECOMPUTE_CONCURRENCY
  _PRECOMPUTE_CONCURRENCY = max(1, n)


def _flow_chunk_size(frames: list[np.ndarray]) -> int:
  """Flow pairs to hold at once, bounded by a fraction of RAM."""
  n = len(frames)
  if n < 2:
    return 1
  h, w = frames[0].shape[:2]
  per_pair_bytes = h * w * 2 * 4 * 2  # (H, W, 2) float32, forward + backward
  total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
  lanes = 2 * _PRECOMPUTE_CONCURRENCY  # two lenses run concurrently per tracker
  chunk = int(total_ram * 0.25 / (per_pair_bytes * lanes))
  return max(1, min(n - 1, chunk))


def _iter_dis_flows(
  frames: list[np.ndarray],
  ks: range,
) -> Iterator[_FlowPair]:
  """``(forward, backward)`` cv2 DIS flow per pair ``(k-1, k)``, computed ahead."""
  pair_ks = list(ks)
  if not pair_ks:
    return

  def calc_pair(k: int) -> _FlowPair:
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)  # type: ignore[attr-defined]
    prev = np.ascontiguousarray(frames[k - 1])
    cur = np.ascontiguousarray(frames[k])
    dis.setVariationalRefinementIterations(_DIS_VARIATIONAL_ITERS)
    forward = dis.calc(prev, cur, None)
    dis.setVariationalRefinementIterations(_DIS_BACKWARD_VARIATIONAL_ITERS)
    backward = dis.calc(cur, prev, None)
    return forward, backward

  # Concurrent DIS calls help only with spare cores; counts never change the output.
  n_cpu = os.cpu_count() or 32
  cv2.setNumThreads(max(n_cpu // 8, min(n_cpu, 32)))
  workers = max(1, n_cpu // 16 // _PRECOMPUTE_CONCURRENCY)
  with ThreadPoolExecutor(max_workers=workers) as executor:
    yield from executor.map(calc_pair, pair_ks)


def _precompute_dis_flows(
  frames: list[np.ndarray],
) -> dict[int, _FlowPair]:
  """Forward+backward DIS flow for every consecutive pair, fully materialized."""
  return dict(enumerate(_iter_dis_flows(frames, range(1, len(frames))), start=1))


def _in_bounds(xy: Float64Array, h: int, w: int) -> npt.NDArray[np.bool_]:
  return (xy[:, 0] >= 1.0) & (xy[:, 0] < w - 1) & (xy[:, 1] >= 1.0) & (xy[:, 1] < h - 1)


def _mask_ok(xy: Float64Array, mask: np.ndarray) -> npt.NDArray[np.bool_]:
  ix = np.round(xy[:, 0]).astype(np.int64)
  iy = np.round(xy[:, 1]).astype(np.int64)
  h, w = mask.shape
  ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
  vals = np.zeros(len(xy), dtype=np.uint8)
  vals[ok] = mask[iy[ok], ix[ok]]
  return vals > 0


def _detect_corners(
  img: np.ndarray,
  mask: np.ndarray | None,
  max_corners: int,
  quality_level: float,
  min_distance: float,
) -> Float64Array:
  if max_corners <= 0:
    return np.empty((0, 2))
  corners = cv2.goodFeaturesToTrack(
    img,
    maxCorners=int(max_corners),
    qualityLevel=quality_level,
    minDistance=float(min_distance),
    mask=mask,
  )
  if corners is None:
    return np.empty((0, 2))
  return corners.reshape(-1, 2).astype(np.float64)


def dis_flow_track(
  frames: list[np.ndarray],
  max_corners: int,
  masks: list[np.ndarray] | None = None,
  fb_error_threshold: float = 1.0,
  min_distance: float = 8.0,
  quality_level: float = 0.01,
  redetect_ratio: float = 0.5,
) -> tuple[list[Frame], list[npt.NDArray[np.int64]]]:
  """Sparse tracks propagated by DIS dense flow, seeded by Shi-Tomasi with redetect."""
  if masks is not None and len(masks) != len(frames):
    raise ValueError(f"masks length {len(masks)} != frames length {len(frames)}")
  if not frames:
    return [], []

  # Stream pair flows from a worker pool in RAM-bounded chunks.
  chunk_size = _flow_chunk_size(frames)
  flow_stream: Iterator[_FlowPair] = iter(())
  chunk_end = 1
  per_frame_pts: list[list[np.ndarray]] = [[] for _ in frames]
  matches_between: list[npt.NDArray[np.int64]] = []

  for k, img in enumerate(frames):
    mask_k = None if masks is None else masks[k]
    if k >= 1:
      if k >= chunk_end:
        chunk_end = min(k + chunk_size, len(frames))
        flow_stream = _iter_dis_flows(frames, range(k, chunk_end))
      flow_fwd, flow_bwd = next(flow_stream)

    if k > 0 and per_frame_pts[k - 1]:
      prev_xy = np.asarray(per_frame_pts[k - 1], dtype=np.float64)
      disp_fwd = _bilinear_sample_flow(flow_fwd, prev_xy)
      nxt_xy = prev_xy + disp_fwd
      disp_bwd = _bilinear_sample_flow(flow_bwd, nxt_xy)
      back_xy = nxt_xy + disp_bwd
      fb = np.linalg.norm(back_xy - prev_xy, axis=1)
      ok = np.isfinite(fb) & (fb < fb_error_threshold)
      h, w = img.shape[:2]
      ok &= _in_bounds(nxt_xy, h, w)
      if mask_k is not None:
        ok &= _mask_ok(nxt_xy, mask_k)
      sel = np.flatnonzero(ok)
      per_frame_pts[k] = list(nxt_xy[sel])
      matches_between.append(
        np.column_stack([sel, np.arange(sel.size)]).astype(np.int64)
      )
    elif k > 0:
      matches_between.append(np.empty((0, 2), dtype=np.int64))

    if per_frame_pts[k]:
      pts_subpix = np.asarray(per_frame_pts[k], dtype=np.float32).reshape(-1, 1, 2)
      cv2.cornerSubPix(
        img,
        pts_subpix,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
      )
      per_frame_pts[k] = list(pts_subpix.reshape(-1, 2).astype(np.float64))

    n_active = len(per_frame_pts[k])
    if k == 0 or n_active < int(round(redetect_ratio * max_corners)):
      if mask_k is None:
        redetect_mask = np.full(img.shape[:2], 255, dtype=np.uint8)
      else:
        redetect_mask = mask_k.copy()
      for p in per_frame_pts[k]:
        cv2.circle(
          redetect_mask,
          (int(round(float(p[0]))), int(round(float(p[1])))),
          int(round(float(min_distance))),
          0,
          thickness=-1,
        )
      new_corners = _detect_corners(
        img,
        redetect_mask,
        max_corners=max_corners - n_active,
        quality_level=quality_level,
        min_distance=min_distance,
      )
      for p in new_corners:
        per_frame_pts[k].append(p.copy())

  frames_out: list[Frame] = []
  for pts in per_frame_pts:
    kps = np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2))
    frames_out.append(
      Frame(
        keypoints=kps,
        descriptors=np.empty((kps.shape[0], 32), dtype=np.uint8),
      )
    )
  return frames_out, matches_between


def detect_orb(
  image: np.ndarray,
  n_features: int,
  fast_threshold: int,
  mask: np.ndarray | None = None,
) -> Frame:
  """Run OpenCV ORB on a single grayscale frame."""
  orb = cv2.ORB_create(  # type: ignore[attr-defined]
    nfeatures=n_features, fastThreshold=fast_threshold
  )
  kps, desc = orb.detectAndCompute(image, mask)
  if kps is None or len(kps) == 0 or desc is None:
    return Frame(
      keypoints=np.empty((0, 2)),
      descriptors=np.empty((0, 32), dtype=np.uint8),
    )
  return Frame(
    keypoints=np.asarray([k.pt for k in kps], dtype=np.float64),
    descriptors=np.asarray(desc, dtype=np.uint8),
  )


def match_pair(a: Frame, b: Frame, ratio: float) -> npt.NDArray[np.int64]:
  """Lowe ratio-test BFMatch as an ``(M, 2)`` array of ``(index_in_a, index_in_b)``."""
  if a.descriptors.size == 0 or b.descriptors.size == 0:
    return np.empty((0, 2), dtype=np.int64)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  knn = bf.knnMatch(a.descriptors, b.descriptors, k=2)
  good: list[tuple[int, int]] = []
  for pair in knn:
    if len(pair) < 2:
      continue
    m0, m1 = pair
    if m0.distance < ratio * m1.distance:
      good.append((int(m0.queryIdx), int(m0.trainIdx)))
  return np.asarray(good, dtype=np.int64) if good else np.empty((0, 2), dtype=np.int64)


def essential_ransac_inliers(
  pts_a: Float64Array,
  pts_b: Float64Array,
  bearings_a: Float64Array,
  bearings_b: Float64Array,
  threshold_px: float,
  fx: float,
  max_half_angle: float = np.pi / 2.0 - 1e-6,
) -> npt.NDArray[np.bool_]:
  """Essential-matrix RANSAC inlier mask; a pair past ``max_half_angle`` stays False."""
  n = pts_a.shape[0]
  if n < 8 or pts_b.shape[0] != n:
    return np.zeros(n, dtype=bool)

  cos_min = float(np.cos(max_half_angle))
  in_fov = (bearings_a[:, 2] > cos_min) & (bearings_b[:, 2] > cos_min)
  if int(in_fov.sum()) < 8:
    return np.zeros(n, dtype=bool)

  ba = bearings_a[in_fov]
  bb = bearings_b[in_fov]
  norm_a = (ba[:, :2] / ba[:, 2:3]).astype(np.float64)
  norm_b = (bb[:, :2] / bb[:, 2:3]).astype(np.float64)
  thresh = float(threshold_px / fx)

  E, mask = cv2.findEssentialMat(
    norm_a,
    norm_b,
    focal=1.0,
    pp=(0.0, 0.0),
    method=cv2.RANSAC,
    prob=0.999,
    threshold=thresh,
  )
  out = np.zeros(n, dtype=bool)
  if E is None or mask is None:
    return out
  in_fov_idx = np.where(in_fov)[0]
  mask_flat = mask.flatten().astype(bool)
  out[in_fov_idx[mask_flat]] = True
  return out


def chain_tracks(
  frames: list[Frame],
  matches_between: list[npt.NDArray[np.int64]],
) -> dict[int, dict[int, Float64Array]]:
  """Chain pairwise matches into ``{track_id: {frame_idx: pt2d}}`` tracks."""
  if len(matches_between) != max(len(frames) - 1, 0):
    raise ValueError(
      f"expected {len(frames) - 1} match arrays, got {len(matches_between)}"
    )
  tracks: dict[int, dict[int, Float64Array]] = {}
  lookup: dict[tuple[int, int], int] = {}
  next_id = 0
  for k, pairs in enumerate(matches_between, start=1):
    if pairs.shape[0] == 0:
      continue
    for a_idx, b_idx in pairs:
      prev_key = (k - 1, int(a_idx))
      if prev_key in lookup:
        tid = lookup[prev_key]
      else:
        tid = next_id
        next_id += 1
        tracks[tid] = {k - 1: frames[k - 1].keypoints[int(a_idx)]}
        lookup[prev_key] = tid
      tracks[tid][k] = frames[k].keypoints[int(b_idx)]
      lookup[(k, int(b_idx))] = tid
  return tracks
