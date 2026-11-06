"""Synthetic frame sequences shared by the vi_ba frontend unit tests."""

import cv2
import numpy as np
import numpy.typing as npt

from specialist.vi_ba.features import Frame

# Wider than every crop, so the np.roll wrap-around seam stays out of each frame.
_BASE_WIDTH = 320


def translating_frames(
  n: int, shift: int, seed: int, height: int = 240, width: int = 240
) -> list[np.ndarray]:
  """Frames shifted ``shift`` px each, so every DIS and RANSAC match is an inlier."""
  rng = np.random.default_rng(seed)
  base = rng.integers(0, 255, size=(height, _BASE_WIDTH), dtype=np.uint8)
  base = cv2.GaussianBlur(base, (0, 0), sigmaX=2.0)
  lo = (_BASE_WIDTH - width) // 2
  return [np.roll(base, shift * k, axis=1)[:, lo : lo + width].copy() for k in range(n)]


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


def cuda_available() -> bool:
  """True when torch imports and sees a CUDA device."""
  try:
    import torch

    return bool(torch.cuda.is_available())
  except Exception:  # noqa: BLE001 - no torch, no GPU tests
    return False
