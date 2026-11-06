"""Visual frontend: DIS flow tracking, essential RANSAC, and revisit-matching ORB."""

import os
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import cv2
import gtsam  # noqa: F401  # the extension links the wheel's library
import numpy as np
import numpy.typing as npt

from specialist.vi_ba._mei_factor_cpp import detect_corners as _detect_corners
from specialist.vi_ba._mei_factor_cpp import free_mask, propagate_points

Float64Array = npt.NDArray[np.float64]
_FlowPair = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
_R = TypeVar("_R")
_T = TypeVar("_T")

__all__ = [
  "detect_corners",
  "Frame",
  "available_ram",
  "chain_tracks",
  "cpu_budget",
  "detect_orb",
  "dis_flow_track",
  "essential_inliers",
  "prepare_essential_rows",
  "parallel_map",
  "set_precompute_concurrency",
]


@dataclass(frozen=True)
class Frame:
  """One keyframe's 2D keypoints and their descriptors."""

  keypoints: Float64Array
  descriptors: np.ndarray


def cpu_budget() -> int:
  """Cores this process may actually run on."""
  affinity = getattr(os, "sched_getaffinity", None)
  if affinity is not None:
    return len(affinity(0))
  return os.cpu_count() or 8


def parallel_map(
  fn: Callable[[_T], _R], items: Iterable[_T], share: int = 4
) -> list[_R]:
  """Ordered ``ThreadPoolExecutor.map`` on one ``share``-th of the cores."""
  todo = list(items)
  if not todo:
    return []
  workers = min(len(todo), max(4, cpu_budget() // share))
  with ThreadPoolExecutor(max_workers=workers) as executor:
    return list(executor.map(fn, todo))


_PRECOMPUTE_CONCURRENCY = 1

# DIS variational-refinement iterations: the subpixel polish and most of DIS runtime.
_DIS_VARIATIONAL_ITERS = 10
# The backward flow feeds only the forward-backward gate, never a track coordinate.
_DIS_BACKWARD_VARIATIONAL_ITERS = 2


def set_precompute_concurrency(n: int) -> None:
  """Set how many cameras prepare at once so the flow precompute shrinks its pool."""
  global _PRECOMPUTE_CONCURRENCY
  _PRECOMPUTE_CONCURRENCY = max(1, n)


# cv2's process-global pool, sized for one DIS calc.
_CV2_THREADS = 16

# Flow pairs the cv2 path computes ahead of the walk.
_FLOW_CHUNK_PAIRS = 24


_MEMINFO = Path("/proc/meminfo")


def available_ram() -> int:
  """Bytes obtainable without swapping, reclaimable page cache included."""
  try:
    for line in _MEMINFO.read_text().splitlines():
      field, _, rest = line.partition(":")
      if field == "MemAvailable":
        return int(rest.split()[0]) * 1024
  except OSError, ValueError, IndexError:
    pass
  page = os.sysconf("SC_PAGE_SIZE")
  try:
    return page * os.sysconf("SC_AVPHYS_PAGES")
  except ValueError, OSError:
    return page * os.sysconf("SC_PHYS_PAGES")


def _iter_dis_flows(frames: Sequence[np.ndarray], ks: range) -> Iterator[_FlowPair]:
  """``(forward, backward)`` cv2 DIS flow per pair ``(k-1, k)``, computed ahead."""

  def calc_pair(k: int) -> _FlowPair:
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)  # type: ignore[attr-defined]
    prev = np.ascontiguousarray(frames[k - 1])
    cur = np.ascontiguousarray(frames[k])
    dis.setVariationalRefinementIterations(_DIS_VARIATIONAL_ITERS)
    forward = dis.calc(prev, cur, None)
    dis.setVariationalRefinementIterations(_DIS_BACKWARD_VARIATIONAL_ITERS)
    backward = dis.calc(cur, prev, None)
    return forward, backward

  # Thread and worker counts never change the output.
  cv2.setNumThreads(min(_CV2_THREADS, cpu_budget()))
  workers = max(1, cpu_budget() // (2 * _CV2_THREADS) // _PRECOMPUTE_CONCURRENCY)
  with ThreadPoolExecutor(max_workers=workers) as executor:
    yield from executor.map(calc_pair, ks)


_FlowIter = Callable[[Sequence[np.ndarray], range], Iterator[_FlowPair]]

_SCALED_CACHE = 64


class _ScaledFrames(Sequence[np.ndarray]):
  """A frame sequence resized on access, with a small ring of recent frames."""

  def __init__(self, frames: Sequence[np.ndarray], scale: float):
    self._frames = frames
    self._scale = scale
    self._cache: dict[int, np.ndarray] = {}
    self._order: list[int] = []

  def __len__(self) -> int:
    return len(self._frames)

  def __getitem__(self, index: int) -> np.ndarray:  # type: ignore[override]
    hit = self._cache.get(index)
    if hit is not None:
      return hit
    small = cv2.resize(
      self._frames[index],
      None,
      fx=self._scale,
      fy=self._scale,
      interpolation=cv2.INTER_AREA,
    )
    self._cache[index] = small
    self._order.append(index)
    if len(self._order) > _SCALED_CACHE:
      self._cache.pop(self._order.pop(0), None)
    return small


def _select_flow_iter(
  frames: Sequence[np.ndarray], gpu: bool, label: str | None
) -> _FlowIter:
  """The lens's flow backend: the GPU wavefront or cv2, the same stream either way."""
  if not gpu:
    return _iter_dis_flows
  from specialist.vi_ba import gpu_flow

  begin = time.perf_counter()
  engine = gpu_flow.prime(
    frames, _DIS_VARIATIONAL_ITERS, _DIS_BACKWARD_VARIATIONAL_ITERS
  )
  print(
    f"  [{_flow_scope(label)}] GPU DIS backend: cuda:{engine.device_index}, "
    f"{engine.batch} pairs/window, ready in {time.perf_counter() - begin:.1f}s"
  )
  # On the GPU path every cv2 call already runs inside a pool of its own.
  cv2.setNumThreads(1)
  return engine.iter_flows


def _flow_scope(label: str | None) -> str:
  return f"{label}/flow" if label else "flow"


def _flow_pairs(
  flow_iter: _FlowIter, frames: Sequence[np.ndarray], band: int | None
) -> Iterator[_FlowPair]:
  """``(forward, backward)`` per consecutive pair, lazily."""
  k = 1
  while k < len(frames):
    lookahead = len(frames) if band is None else band - k % band
    if flow_iter is _iter_dis_flows:
      lookahead = min(lookahead, _FLOW_CHUNK_PAIRS)
    end = min(k + lookahead, len(frames))
    yield from flow_iter(frames, range(k, end))
    k = end


# Corner quality fraction, clearance from tracked points, and the reseed loss ratio.
_CORNER_QUALITY_LEVEL = 0.01
_CORNER_MIN_DISTANCE = 8.0
_REDETECT_RATIO = 0.5
_SUBPIX_WINDOW = (5, 5)
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
# Forward-backward round-trip error (px) above which a propagated point is dropped.
_FB_ERROR_THRESHOLD = 1.0


# Rows per corner-response strip, and the stencil's halo beyond it.
_RESPONSE_STRIP = 256
_RESPONSE_HALO = 2
_response_pool = ThreadPoolExecutor(max_workers=8)
# Points per cornerSubPix call.
_SUBPIX_CHUNK = 250
_subpix_pool = ThreadPoolExecutor(max_workers=min(_CV2_THREADS, cpu_budget()))


def _corner_response(img: np.ndarray) -> np.ndarray:
  """``cv2.cornerMinEigenVal`` of ``img``, computed on row strips side by side."""
  h = img.shape[0]
  out = np.empty(img.shape[:2], dtype=np.float32)

  def strip(y0: int) -> None:
    y1 = min(h, y0 + _RESPONSE_STRIP)
    top = max(0, y0 - _RESPONSE_HALO)
    part = cv2.cornerMinEigenVal(
      img[top : min(h, y1 + _RESPONSE_HALO)], blockSize=3, ksize=3
    )
    out[y0:y1] = part[y0 - top : y1 - top]

  list(_response_pool.map(strip, range(0, h, _RESPONSE_STRIP)))
  return out


def detect_corners(
  img: np.ndarray,
  mask: np.ndarray | None,
  max_corners: int,
  quality_level: float,
  min_distance: float,
) -> Float64Array:
  """``cv2.goodFeaturesToTrack`` from the strip response; ``max_corners <= 0`` returns
  no corners and computes no response."""
  if max_corners <= 0:
    return np.empty((0, 2))
  corners: Float64Array = _detect_corners(
    _corner_response(img),
    mask,
    float(quality_level),
    float(min_distance),
    int(max_corners),
  )
  return corners


def _subpix(img: np.ndarray, xy: Float64Array) -> Float64Array:
  """``xy`` refined to the nearest corner of ``img`` at subpixel precision."""
  if xy.shape[0] == 0:
    return xy
  pts = xy.astype(np.float32).reshape(-1, 1, 2)
  chunks = [pts[i : i + _SUBPIX_CHUNK] for i in range(0, pts.shape[0], _SUBPIX_CHUNK)]

  def refine(chunk: np.ndarray) -> None:
    cv2.cornerSubPix(img, chunk, _SUBPIX_WINDOW, (-1, -1), _SUBPIX_CRITERIA)

  list(_subpix_pool.map(refine, chunks))
  return pts.reshape(-1, 2).astype(np.float64)


_disc_offsets: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _disc(radius: int) -> tuple[np.ndarray, np.ndarray]:
  """``(dy, dx)`` offsets of the pixels ``cv2.circle`` fills for a ``radius`` disc."""
  hit = _disc_offsets.get(radius)
  if hit is None:
    center = 2 * radius + 1
    canvas = np.zeros((2 * center + 1, 2 * center + 1), dtype=np.uint8)
    cv2.circle(canvas, (center, center), radius, 255, thickness=-1)
    ys, xs = np.nonzero(canvas)
    hit = _disc_offsets[radius] = (ys - center, xs - center)
  return hit


def _free_mask(
  shape: tuple[int, int], centers: npt.NDArray[np.int64], radius: int
) -> np.ndarray:
  """A 255 mask, zero on ``radius`` discs around the ``(x, y)`` ``centers``."""
  dy, dx = _disc(radius)
  mask: np.ndarray = free_mask(
    int(shape[0]), int(shape[1]), centers.reshape(-1, 2), dy, dx
  )
  return mask


def _seed_corners(
  img: np.ndarray, taken: Float64Array, max_corners: int
) -> Float64Array:
  """Up to ``max_corners`` new Shi-Tomasi corners of ``img`` clear of ``taken``."""
  centers = np.rint(taken).astype(np.int64)
  mask = _free_mask(img.shape[:2], centers, round(_CORNER_MIN_DISTANCE))
  return detect_corners(
    img,
    mask,
    max_corners=max_corners,
    quality_level=_CORNER_QUALITY_LEVEL,
    min_distance=_CORNER_MIN_DISTANCE,
  )


def dis_flow_track(
  frames: Sequence[np.ndarray],
  max_corners: int,
  gpu_flow: bool = False,
  flow_scale: float = 1.0,
  *,
  label: str | None = None,
  on_frame: Callable[[int, np.ndarray], None] | None = None,
  band: int | None = None,
) -> tuple[list[Float64Array], list[npt.NDArray[np.int64]]]:
  """Each frame's ``(N, 2)`` points and the ``(M, 2)`` index pairs to the next frame."""
  if not frames:
    return [], []

  flow_frames = frames if flow_scale == 1.0 else _ScaledFrames(frames, flow_scale)
  flow_iter = _select_flow_iter(flow_frames, gpu_flow and len(frames) > 1, label)
  scale_note = "" if flow_scale == 1.0 else f" @{flow_scale:g}x"
  backend = "cv2" if flow_iter is _iter_dis_flows else "gpu"
  flow_begin = time.perf_counter()
  pairs = _flow_pairs(flow_iter, flow_frames, band)
  # Sized by ``len(frames)``: iterating a banded reader decodes the whole lens.
  per_frame_pts: list[Float64Array] = [np.empty((0, 2)) for _ in range(len(frames))]
  matches_between: list[npt.NDArray[np.int64]] = []
  n_reseeds = 0

  for k in range(len(frames)):
    img = frames[k]
    if on_frame is not None:
      on_frame(k, img)
    if k > 0:
      flow_fwd, flow_bwd = next(pairs)
      h, w = img.shape[:2]
      nxt_xy, ok = propagate_points(
        flow_fwd, flow_bwd, per_frame_pts[k - 1], flow_scale, h, w, _FB_ERROR_THRESHOLD
      )
      sel = np.flatnonzero(ok)
      per_frame_pts[k] = nxt_xy[sel]
      matches_between.append(
        np.column_stack([sel, np.arange(sel.size)]).astype(np.int64)
      )

    per_frame_pts[k] = _subpix(img, per_frame_pts[k])

    n_active = per_frame_pts[k].shape[0]
    if k == 0 or n_active < round(_REDETECT_RATIO * max_corners):
      n_reseeds += 1
      new_corners = _seed_corners(img, per_frame_pts[k], max_corners - n_active)
      per_frame_pts[k] = np.vstack([per_frame_pts[k], new_corners])

  print(
    f"  [{_flow_scope(label)}] {backend}{scale_note} DIS tracked "
    f"{len(frames)} frames in {time.perf_counter() - flow_begin:.1f}s, "
    f"reseeded {n_reseeds}"
  )
  return per_frame_pts, matches_between


def detect_orb(image: np.ndarray, n_features: int, fast_threshold: int) -> Frame:
  """Run OpenCV ORB on a single grayscale frame."""
  orb = cv2.ORB_create(  # type: ignore[attr-defined]
    nfeatures=n_features, fastThreshold=fast_threshold
  )
  kps, desc = orb.detectAndCompute(image, None)
  if kps is None or len(kps) == 0 or desc is None:
    return Frame(
      keypoints=np.empty((0, 2)),
      descriptors=np.empty((0, 32), dtype=np.uint8),
    )
  return Frame(
    keypoints=cv2.KeyPoint_convert(kps).astype(np.float64),  # type: ignore[attr-defined]
    descriptors=np.asarray(desc, dtype=np.uint8),
  )


def prepare_essential_rows(
  bearings_a: Float64Array, bearings_b: Float64Array, max_half_angle: float
) -> tuple[npt.NDArray[np.bool_], Float64Array, Float64Array]:
  """The in-view rows and both sides' normalized coordinates, row for row."""
  if bearings_b.shape[0] != bearings_a.shape[0]:
    raise ValueError("bearings_a and bearings_b must pair row for row")
  cos_min = float(np.cos(max_half_angle))
  in_fov = (bearings_a[:, 2] > cos_min) & (bearings_b[:, 2] > cos_min)
  with np.errstate(divide="ignore", invalid="ignore"):
    normalized_a = bearings_a[:, :2] / bearings_a[:, 2:3]
    normalized_b = bearings_b[:, :2] / bearings_b[:, 2:3]
  return in_fov, normalized_a, normalized_b


def essential_inliers(
  in_fov: npt.NDArray[np.bool_],
  normalized_a: Float64Array,
  normalized_b: Float64Array,
  threshold: float,
) -> npt.NDArray[np.bool_]:
  """USAC essential inliers of one pair's in-view rows; ``threshold`` is normalized."""
  out = np.zeros(in_fov.shape[0], dtype=bool)
  if in_fov.sum() < 8:
    return out
  E, mask = cv2.findEssentialMat(
    normalized_a[in_fov],
    normalized_b[in_fov],
    focal=1.0,
    pp=(0.0, 0.0),
    method=cv2.USAC_DEFAULT,
    prob=0.999,
    threshold=threshold,
  )
  if E is None or mask is None:
    return out
  out[np.flatnonzero(in_fov)[mask.ravel().astype(bool)]] = True
  return out


def chain_tracks(
  keypoints: list[Float64Array],
  matches_between: list[npt.NDArray[np.int64]],
) -> dict[int, dict[int, Float64Array]]:
  """One-to-one matches as ``{track_id: {frame: pixel}}``, ids by first appearance."""
  if len(matches_between) != max(len(keypoints) - 1, 0):
    raise ValueError(
      f"expected {len(keypoints) - 1} match arrays, got {len(matches_between)}"
    )
  if not keypoints:
    return {}
  # Per observation, in the order the tracks acquire them.
  obs_tid: list[npt.NDArray[np.int64]] = []
  obs_kf: list[npt.NDArray[np.int64]] = []
  obs_row: list[npt.NDArray[np.int64]] = []
  frame_start = np.concatenate(([0], np.cumsum([len(k) for k in keypoints])))
  track_of = np.full(len(keypoints[0]), -1, dtype=np.int64)
  next_id = 0
  for k, pairs in enumerate(matches_between, start=1):
    track_of_next = np.full(len(keypoints[k]), -1, dtype=np.int64)
    if pairs.shape[0]:
      a = np.asarray(pairs[:, 0], dtype=np.int64)
      b = np.asarray(pairs[:, 1], dtype=np.int64)
      if np.unique(a).size < a.size or np.unique(b).size < b.size:
        raise ValueError(f"matches into frame {k} are not one to one")
      tid = track_of[a]
      new = tid < 0
      n_new = int(new.sum())
      tid[new] = next_id + np.arange(n_new)
      next_id += n_new
      obs_tid.append(tid[new])
      obs_kf.append(np.full(n_new, k - 1, dtype=np.int64))
      obs_row.append(frame_start[k - 1] + a[new])
      obs_tid.append(tid)
      obs_kf.append(np.full(b.size, k, dtype=np.int64))
      obs_row.append(frame_start[k] + b)
      track_of_next[b] = tid
    track_of = track_of_next
  if next_id == 0:
    return {}
  tid_all = np.concatenate(obs_tid)
  order = np.argsort(tid_all, kind="stable")
  kfs = np.concatenate(obs_kf)[order].tolist()
  points = list(np.concatenate(keypoints)[np.concatenate(obs_row)[order]])
  bounds = np.flatnonzero(np.diff(tid_all[order], prepend=-1, append=next_id))
  return {
    tid: dict(zip(kfs[begin:end], points[begin:end], strict=True))
    for tid, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True))
  }
