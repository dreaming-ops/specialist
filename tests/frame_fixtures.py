"""Synthetic frame sequences shared by the vi_ba frontend unit tests."""

from __future__ import annotations

import cv2
import numpy as np

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
