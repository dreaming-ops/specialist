"""The GPU revisit matcher returns cv2's brute-force answer."""

from __future__ import annotations

import numpy as np
import pytest

from specialist.vi_ba import gpu_match
from specialist.vi_ba.features import detect_orb
from tests.frame_fixtures import cuda_available, match_pair


def _block_frames(n: int) -> list[np.ndarray]:
  """Random 8 px blocks shifted 3 px a frame: many FAST corners and descriptor ties."""
  rng = np.random.default_rng(5)
  cells = rng.integers(0, 2, (64, 70), dtype=np.uint8) * 255
  base = np.kron(cells, np.ones((8, 8), dtype=np.uint8))
  return [np.ascontiguousarray(base[:, 3 * k : 3 * k + 512]) for k in range(n)]


@pytest.mark.skipif(not cuda_available(), reason="needs CUDA")
def test_gpu_matches_equal_cv2_with_uneven_and_tied_trains() -> None:
  """Uneven train frames share one product, and repeated descriptors tie everywhere."""
  frames = [detect_orb(f, 2000, 20) for f in _block_frames(5)]
  rng = np.random.default_rng(11)
  for k, keep in enumerate((1700, 900, 300, 2, 1200)):
    order = rng.permutation(len(frames[k].keypoints))[:keep]
    frames[k] = type(frames[k])(
      keypoints=frames[k].keypoints[order], descriptors=frames[k].descriptors[order]
    )
  tied = frames[0].descriptors[rng.integers(0, 40, 600)]
  frames.append(type(frames[0])(keypoints=np.zeros((600, 2)), descriptors=tied))
  n = len(frames)
  pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
  out = gpu_match.match_pairs(frames, pairs, 0.8)
  for (i, j), got in zip(pairs, out, strict=True):
    assert np.array_equal(got, match_pair(frames[i], frames[j], 0.8)), (i, j)
  assert sum(len(m) for m in out) > 100
