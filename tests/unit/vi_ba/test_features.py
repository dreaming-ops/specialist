"""``specialist.vi_ba.features`` on synthetic frames."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from specialist.vi_ba import features
from specialist.vi_ba.features import (
  Frame,
  _mask_ok,
  chain_tracks,
  detect_orb,
  essential_ransac_inliers,
  mask_plane,
  match_pair,
)

Float64Array = npt.NDArray[np.float64]


def _textured_image(rng: np.random.Generator, h: int = 200, w: int = 300) -> np.ndarray:
  return rng.integers(0, 255, size=(h, w), dtype=np.uint8)


def test_detect_orb_on_textured_image_returns_keypoints() -> None:
  rng = np.random.default_rng(0)
  frame = detect_orb(_textured_image(rng), n_features=500, fast_threshold=12)
  assert isinstance(frame, Frame)
  assert frame.keypoints.ndim == 2 and frame.keypoints.shape[1] == 2
  assert frame.keypoints.shape[0] > 10
  assert frame.descriptors.shape == (frame.keypoints.shape[0], 32)
  assert frame.descriptors.dtype == np.uint8


def test_detect_orb_on_solid_image_returns_empty_frame() -> None:
  frame = detect_orb(
    np.zeros((100, 100), dtype=np.uint8), n_features=500, fast_threshold=12
  )
  assert frame.keypoints.shape == (0, 2)
  assert frame.descriptors.shape == (0, 32)


def test_detect_orb_respects_mask() -> None:
  """Keypoints inside the masked-out region must be suppressed."""
  rng = np.random.default_rng(0)
  img = _textured_image(rng, h=200, w=300)
  # Mask out the right half of the frame: 0 = ignore, 255 = keep.
  mask = np.full(img.shape, 255, dtype=np.uint8)
  mask[:, 150:] = 0
  frame = detect_orb(img, n_features=500, fast_threshold=12, mask=mask)
  assert frame.keypoints.shape[0] > 0
  # No keypoint should land inside the masked-out region.
  assert np.all(frame.keypoints[:, 0] < 150), (
    f"masked region produced {(frame.keypoints[:, 0] >= 150).sum()} keypoints"
  )


def test_match_pair_identical_frames_returns_well_formed_matches() -> None:
  rng = np.random.default_rng(1)
  img = _textured_image(rng)
  a = detect_orb(img, n_features=200, fast_threshold=12)
  # The ratio test can reject identical descriptors whose second-best is equally close.
  matches = match_pair(a, a, ratio=0.99)
  assert matches.ndim == 2 and matches.shape[1] == 2
  assert matches.shape[0] > 0


def test_match_pair_empty_descriptors_returns_empty() -> None:
  empty = Frame(
    keypoints=np.empty((0, 2), dtype=np.float64),
    descriptors=np.empty((0, 32), dtype=np.uint8),
  )
  assert match_pair(empty, empty, ratio=0.7).shape == (0, 2)


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


def test_rect_mask_matches_the_plane_it_replaces() -> None:
  """The blocked-rectangle point test agrees exactly with the full-resolution plane."""
  shape = (40, 60)
  blocked = [(5, 5, 20, 15), (30, 25, 55, 38)]
  plane = mask_plane(shape, blocked)

  ys, xs = np.mgrid[-3 : shape[0] + 3, -3 : shape[1] + 3]
  xy = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

  ix = np.round(xy[:, 0]).astype(np.int64)
  iy = np.round(xy[:, 1]).astype(np.int64)
  inside = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
  from_plane = np.zeros(len(xy), dtype=bool)
  from_plane[inside] = plane[iy[inside], ix[inside]] > 0

  assert np.array_equal(_mask_ok(xy, blocked, shape), from_plane)


def test_rect_mask_with_no_rectangles_keeps_everything_in_frame() -> None:
  xy = np.array([[0.0, 0.0], [9.0, 9.0], [-1.0, 5.0], [5.0, 99.0]])
  assert list(_mask_ok(xy, [], (10, 10))) == [True, True, False, False]


def test_flow_chunk_budget_tracks_available_memory() -> None:
  """Total RAM over-commits: decoded keyframes and co-tenants already hold memory."""
  frames = [np.zeros((2880, 2880), dtype=np.uint8) for _ in range(400)]
  per_pair = 2880 * 2880 * 2 * 4 * 2
  cached = features._flow_chunk_size(frames) * per_pair
  assert 0 < cached <= max(features.available_ram() * 0.25, per_pair)
  assert features.available_ram() > 0


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


def test_flow_chunk_is_capped_regardless_of_memory() -> None:
  """The cap is ``_FLOW_CHUNK_MAX_PAIRS`` however much memory the host has."""
  frames = [np.zeros((2880, 2880), dtype=np.uint8) for _ in range(4000)]
  assert features._flow_chunk_size(frames) <= features._FLOW_CHUNK_MAX_PAIRS


def test_cpu_budget_follows_the_cores_the_process_may_run_on() -> None:
  """``os.cpu_count`` reports the machine, not the cores this process may use."""
  budget = features.cpu_budget()
  assert 0 < budget <= (os.cpu_count() or budget)
  if (affinity := getattr(os, "sched_getaffinity", None)) is not None:
    assert budget == len(affinity(0))


def test_flow_falls_back_to_cv2_when_the_gpu_stream_fails_mid_lens(
  capsys: pytest.CaptureFixture[str],
) -> None:
  """Losing the GPU part-way through a lens costs accuracy, not the run."""
  rng = np.random.default_rng(0)
  base = _textured_image(rng, 120, 160)
  frames = [np.roll(base, shift, axis=1) for shift in range(6)]

  def dies_on_the_third_pair(
    frames_in: object, ks: range
  ) -> object:  # matches _FlowIter
    for k in ks:
      if k >= 3:
        raise RuntimeError("CUDA out of memory")
      yield from features._iter_dis_flows(frames_in, range(k, k + 1))

  expected, _ = features.dis_flow_track(frames, max_corners=40)
  original = features._select_flow_iter
  features._select_flow_iter = lambda _frames, _gpu: dies_on_the_third_pair
  try:
    recovered, _ = features.dis_flow_track(frames, max_corners=40, gpu_flow=True)
  finally:
    features._select_flow_iter = original

  assert "GPU DIS lost" in capsys.readouterr().out
  assert len(recovered) == len(expected)
  assert sum(pts.shape[0] for pts in recovered) > 0


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
