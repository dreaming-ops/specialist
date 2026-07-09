"""``_precompute_dis_flows`` matches a serial shared-instance DIS loop byte for byte."""

from __future__ import annotations

import cv2
import numpy as np

from specialist.vi_ba.features import (
  _DIS_BACKWARD_VARIATIONAL_ITERS,
  _DIS_VARIATIONAL_ITERS,
  _precompute_dis_flows,
)


def _translating_frames(n: int, shift: int, seed: int) -> list[np.ndarray]:
  """A textured 256x256 frame translated by ``shift`` px/frame (pure +x)."""
  rng = np.random.default_rng(seed)
  base = rng.integers(0, 255, size=(256, 320), dtype=np.uint8)
  base = cv2.GaussianBlur(base, (0, 0), sigmaX=2.0)
  return [np.roll(base, shift * k, axis=1)[:, 32:288].copy() for k in range(n)]


def test_precompute_matches_shared_serial_dis() -> None:
  """Parallel fresh-per-pair flow == the original shared-DIS serial flow, exactly."""
  frames = _translating_frames(5, shift=3, seed=2)

  # Serial reference: one DISOpticalFlow reused across pairs (the original path).
  dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
  ref: dict[int, tuple[np.ndarray, np.ndarray]] = {}
  for k in range(1, len(frames)):
    a = np.ascontiguousarray(frames[k - 1])
    b = np.ascontiguousarray(frames[k])
    dis.setVariationalRefinementIterations(_DIS_VARIATIONAL_ITERS)
    fwd = dis.calc(a, b, None)
    dis.setVariationalRefinementIterations(_DIS_BACKWARD_VARIATIONAL_ITERS)
    ref[k] = (fwd, dis.calc(b, a, None))

  par = _precompute_dis_flows(frames)

  assert set(par) == set(ref)
  for k in ref:
    assert np.array_equal(par[k][0], ref[k][0]), f"fwd flow differs at pair {k}"
    assert np.array_equal(par[k][1], ref[k][1]), f"bwd flow differs at pair {k}"
