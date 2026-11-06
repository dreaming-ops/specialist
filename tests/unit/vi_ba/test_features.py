"""``specialist.vi_ba.features`` on synthetic frames."""

import os
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from specialist.vi_ba import features
from specialist.vi_ba.features import (
  chain_tracks,
  detect_orb,
  essential_inliers,
  prepare_essential_rows,
)

Float64Array = npt.NDArray[np.float64]


def essential_ransac_inliers(
  bearings_a: Float64Array,
  bearings_b: Float64Array,
  threshold_px: float,
  fx: float,
  max_half_angle: float = np.pi / 2.0 - 1e-6,
) -> npt.NDArray[np.bool_]:
  in_fov, normalized_a, normalized_b = prepare_essential_rows(
    bearings_a, bearings_b, max_half_angle
  )
  return essential_inliers(in_fov, normalized_a, normalized_b, float(threshold_px / fx))


def _textured_image(rng: np.random.Generator, h: int = 200, w: int = 300) -> np.ndarray:
  return rng.integers(0, 255, size=(h, w), dtype=np.uint8)


def test_detect_orb_on_solid_image_returns_empty_frame() -> None:
  frame = detect_orb(
    np.zeros((100, 100), dtype=np.uint8), n_features=500, fast_threshold=12
  )
  assert frame.keypoints.shape == (0, 2)
  assert frame.descriptors.shape == (0, 32)


def test_essential_ransac_with_few_points_returns_all_false() -> None:
  bearings = np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]])  # < 8 correspondences
  inliers = essential_ransac_inliers(bearings, bearings, threshold_px=3.0, fx=500.0)
  assert inliers.shape == (2,)
  assert inliers.dtype == bool
  assert not inliers.any()


def _synthetic_correspondence_pair(
  rng: np.random.Generator,
  n_inliers: int,
  n_outliers: int,
) -> tuple[Float64Array, Float64Array]:
  """Return (bearings_a, bearings_b) for a known x-translation."""
  # 3D points in camera A's frame, z > 0.
  pts3d = rng.uniform(-0.5, 0.5, size=(n_inliers, 3))
  pts3d[:, 2] = rng.uniform(1.5, 3.0, size=n_inliers)
  # Camera B is translated along +x by 0.2 m in camera A's frame.
  pts3d_b = pts3d - np.array([0.2, 0.0, 0.0])

  def _unit_rows(p: Float64Array) -> Float64Array:
    bearings: Float64Array = p / np.linalg.norm(p, axis=1, keepdims=True)
    return bearings

  bears_a, bears_b = _unit_rows(pts3d), _unit_rows(pts3d_b)

  if n_outliers > 0:
    bears_a = np.vstack([bears_a, _unit_rows(rng.normal(size=(n_outliers, 3)))])
    bears_b = np.vstack([bears_b, _unit_rows(rng.normal(size=(n_outliers, 3)))])

  return bears_a, bears_b


def test_essential_ransac_selects_inliers_over_outliers() -> None:
  rng = np.random.default_rng(2)
  bears_a, bears_b = _synthetic_correspondence_pair(rng, n_inliers=30, n_outliers=10)
  inliers = essential_ransac_inliers(bears_a, bears_b, threshold_px=1.5, fx=500.0)
  assert inliers.shape == (40,)
  assert inliers[:30].sum() >= 25
  assert inliers[30:].sum() <= 4


def test_chain_tracks_two_frames_one_match() -> None:
  keypoints = [
    np.array([[10.0, 10.0], [20.0, 20.0]]),
    np.array([[11.0, 11.0], [21.0, 19.0]]),
  ]
  matches = [np.array([[0, 1]], dtype=np.int64)]
  tracks = chain_tracks(keypoints, matches)
  assert len(tracks) == 1
  tid = next(iter(tracks))
  assert set(tracks[tid].keys()) == {0, 1}
  assert np.allclose(tracks[tid][0], [10.0, 10.0])
  assert np.allclose(tracks[tid][1], [21.0, 19.0])


def test_chain_tracks_three_frames_chain_through_shared_keypoint() -> None:
  keypoints = [np.array([[0.0, 0.0], [100.0, 100.0]]) for _ in range(3)]
  matches = [
    np.array([[0, 0]], dtype=np.int64),  # f0→f1
    np.array([[0, 0]], dtype=np.int64),  # f1→f2
  ]
  tracks = chain_tracks(keypoints, matches)
  assert len(tracks) == 1
  tid = next(iter(tracks))
  assert set(tracks[tid].keys()) == {0, 1, 2}


def test_chain_tracks_unrelated_pairs_produce_separate_tracks() -> None:
  # f0→f1 pairs kp0 with kp0 and f1→f2 pairs kp1 with kp0, so the tracks do not chain.
  keypoints = [np.array([[0.0, 0.0], [1.0, 1.0]]) for _ in range(3)]
  matches = [
    np.array([[0, 0]], dtype=np.int64),
    np.array([[1, 0]], dtype=np.int64),
  ]
  tracks = chain_tracks(keypoints, matches)
  assert len(tracks) == 2
  lengths = sorted(len(obs) for obs in tracks.values())
  assert lengths == [2, 2]


_MEMINFO_CASES = (
  # A warm page cache leaves little memory free but plenty available.
  (
    "MemTotal:       263773316 kB\nMemFree:         3145728 kB\n"
    "Buffers:          524288 kB\nCached:        226492416 kB\n"
    "MemAvailable:   229638144 kB\n",
    229638144 * 1024,
  ),
  # An old kernel's meminfo lacks MemAvailable, so the sysconf pages answer.
  ("MemTotal:       263773316 kB\nMemFree:         3145728 kB\n", None),
  # No meminfo file at all (macOS) falls back the same way.
  (None, None),
)


@pytest.mark.parametrize(("meminfo", "expected"), _MEMINFO_CASES)
def test_available_ram_follows_the_available_figure(
  meminfo: str | None,
  expected: int | None,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The budget falls back to a positive figure wherever MemAvailable is missing."""
  path = tmp_path / "meminfo"
  if meminfo is not None:
    path.write_text(meminfo)
  monkeypatch.setattr(features, "_MEMINFO", path)
  if expected is None:
    assert features.available_ram() > 0
  else:
    assert features.available_ram() == expected


def test_cpu_budget_follows_the_cores_the_process_may_run_on(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """``sched_getaffinity`` exists only on Linux, so the test patches it in."""
  monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3}, raising=False)
  assert features.cpu_budget() == 4, "the budget ignores the cores it was given"
  monkeypatch.delattr(os, "sched_getaffinity", raising=False)
  assert 0 < features.cpu_budget() <= (os.cpu_count() or 4)


class _CountingFrames(Sequence[np.ndarray]):
  """Counts every ``__getitem__``, since each banded-reader access is a band decode."""

  def __init__(self, frames: list[np.ndarray]) -> None:
    self.frames = frames
    self.counts = [0] * len(frames)

  def __len__(self) -> int:
    return len(self.frames)

  def __getitem__(self, index: int) -> np.ndarray:  # type: ignore[override]
    self.counts[index] += 1
    return self.frames[index]


def test_flow_reads_each_frame_a_bounded_number_of_times() -> None:
  """The camera's walk plus the flow pairs read each frame at most three times."""
  rng = np.random.default_rng(0)
  frames = _CountingFrames(
    [rng.integers(0, 255, (64, 64), dtype=np.uint8) for _ in range(6)]
  )
  features.dis_flow_track(frames, max_corners=30)
  assert max(frames.counts) <= 3, frames.counts


def test_seed_corners_keep_clear_of_taken_points() -> None:
  """Seeds lie in the frame and beyond the minimum distance of every tracked point."""
  rng = np.random.default_rng(3)
  img = _textured_image(rng, 240, 320)
  taken = features._seed_corners(img, np.empty((0, 2)), 50)
  assert taken.shape[0] > 0
  assert np.all(taken[:, 0] >= 0) and np.all(taken[:, 0] < 320)
  assert np.all(taken[:, 1] >= 0) and np.all(taken[:, 1] < 240)
  more = features._seed_corners(img, taken, 50)
  assert more.shape[0] > 0
  gaps = np.linalg.norm(more[:, None, :] - taken[None, :, :], axis=2)
  assert gaps.min() >= features._CORNER_MIN_DISTANCE - 1.0


def test_flow_walk_hands_every_frame_to_the_hook_in_order() -> None:
  rng = np.random.default_rng(0)
  base = _textured_image(rng, 120, 160)
  frames = [np.roll(base, shift, axis=1) for shift in range(5)]
  seen: list[int] = []

  def hook(k: int, frame: np.ndarray) -> None:
    assert frame is frames[k]
    seen.append(k)

  features.dis_flow_track(frames, max_corners=40, on_frame=hook)
  assert seen == [0, 1, 2, 3, 4]


def test_subpix_in_chunks_equals_one_call() -> None:
  rng = np.random.default_rng(9)
  img = _textured_image(rng, 400, 400)
  xy = features._seed_corners(img, np.empty((0, 2)), 3000)
  assert xy.shape[0] > features._SUBPIX_CHUNK
  pts = xy.astype(np.float32).reshape(-1, 1, 2)
  cv2.cornerSubPix(
    img, pts, features._SUBPIX_WINDOW, (-1, -1), features._SUBPIX_CRITERIA
  )
  assert np.array_equal(
    features._subpix(img, xy), pts.reshape(-1, 2).astype(np.float64)
  )


def test_detect_corners_is_good_features_to_track() -> None:
  """With or without a mask, it returns cv2's corners in cv2's order."""
  rng = np.random.default_rng(21)
  img = _textured_image(rng, 700, 900)
  mask = np.full(img.shape, 255, dtype=np.uint8)
  mask[200:260, :] = 0
  mask[:, 500:520] = 0
  for m in (None, mask):
    expected = cv2.goodFeaturesToTrack(img, 3000, 0.01, 8.0, mask=m)
    got = features.detect_corners(img, m, 3000, 0.01, 8.0)
    assert np.array_equal(got, expected.reshape(-1, 2).astype(np.float64))
    assert got.shape[0] > 500


def test_corner_response_is_the_whole_frame_response() -> None:
  rng = np.random.default_rng(22)
  img = _textured_image(rng, 1000, 300)
  assert np.array_equal(
    features._corner_response(img), cv2.cornerMinEigenVal(img, blockSize=3, ksize=3)
  )


def _reference_propagate(
  flow_fwd: np.ndarray,
  flow_bwd: np.ndarray,
  prev_xy: np.ndarray,
  scale: float,
  h: int,
  w: int,
  threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
  """The float64 numpy propagation step with its forward-backward and border gates."""

  def sample(flow: np.ndarray, xy: np.ndarray) -> np.ndarray:
    fh, fw = flow.shape[:2]
    x = np.clip((xy[:, 0] + 0.5) * scale - 0.5, 0.0, float(fw) - 1.001)
    y = np.clip((xy[:, 1] + 0.5) * scale - 0.5, 0.0, float(fh) - 1.001)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    f00 = flow[y0, x0].astype(np.float64)
    f01 = flow[y0, x0 + 1].astype(np.float64)
    f10 = flow[y0 + 1, x0].astype(np.float64)
    f11 = flow[y0 + 1, x0 + 1].astype(np.float64)
    sampled = (
      (1.0 - fx) * (1.0 - fy) * f00
      + fx * (1.0 - fy) * f01
      + (1.0 - fx) * fy * f10
      + fx * fy * f11
    )
    return sampled if scale == 1.0 else sampled / scale

  nxt = prev_xy + sample(flow_fwd, prev_xy)
  back = nxt + sample(flow_bwd, nxt)
  fb = np.linalg.norm(back - prev_xy, axis=1)
  ok = np.isfinite(fb) & (fb < threshold)
  ok &= (
    (nxt[:, 0] >= 1.0) & (nxt[:, 0] < w - 1) & (nxt[:, 1] >= 1.0) & (nxt[:, 1] < h - 1)
  )
  return nxt, ok


@pytest.mark.parametrize("scale", [1.0, 0.25])
def test_propagate_points_is_the_float64_reference(scale: float) -> None:
  """It matches bit for bit, gate included, over points past every frame edge."""
  rng = np.random.default_rng(31)
  h, w = 96, 128
  fh, fw = round(h * scale), round(w * scale)
  flow_fwd = rng.normal(0.0, 3.0, (fh, fw, 2)).astype(np.float32)
  flow_bwd = rng.normal(0.0, 3.0, (fh, fw, 2)).astype(np.float32)
  prev = rng.uniform(-4.0, 4.0, (5000, 2)) + rng.uniform(0.0, 1.0, (5000, 1)) * (w, h)
  nxt, ok = features.propagate_points(flow_fwd, flow_bwd, prev, scale, h, w, 1.0)
  want_nxt, want_ok = _reference_propagate(flow_fwd, flow_bwd, prev, scale, h, w, 1.0)
  assert np.array_equal(nxt, want_nxt)
  assert np.array_equal(ok, want_ok)
  assert 0 < ok.sum() < ok.size


def test_free_mask_is_the_union_of_circle_fills() -> None:
  """The union holds for centers past the frame edge too."""
  rng = np.random.default_rng(32)
  shape = (120, 150)
  centers = rng.integers(-12, 160, (300, 2))
  for radius in (4, 8):
    want = np.full(shape, 255, dtype=np.uint8)
    for x, y in centers:
      cv2.circle(want, (int(x), int(y)), radius, 0, thickness=-1)
    assert np.array_equal(features._free_mask(shape, centers, radius), want)


def test_detect_orb_is_cv2_orb() -> None:
  rng = np.random.default_rng(21)
  img = _textured_image(rng, 240, 320)
  frame = features.detect_orb(img, 300, 20)
  orb = cv2.ORB_create(nfeatures=300, fastThreshold=20)  # type: ignore[attr-defined]
  kps, desc = orb.detectAndCompute(img, None)
  assert np.array_equal(
    frame.keypoints, np.asarray([k.pt for k in kps], dtype=np.float64)
  )
  assert np.array_equal(frame.descriptors, desc)
