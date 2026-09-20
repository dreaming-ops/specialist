"""``_iter_dis_flows`` against a serial cv2 DIS loop and the GPU DIS stream."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from specialist.vi_ba import gpu_flow
from specialist.vi_ba.features import (
  _DIS_BACKWARD_VARIATIONAL_ITERS,
  _DIS_VARIATIONAL_ITERS,
  _iter_dis_flows,
)
from tests.frame_fixtures import cuda_available, translating_frames

# About twice the median per-pixel gap (px) between the GPU port and cv2 DIS here.
_GPU_FLOW_TOL = 0.015


def test_precompute_matches_shared_serial_dis() -> None:
  frames = translating_frames(5, shift=3, seed=2, height=256, width=256)

  # Serial reference: one DISOpticalFlow reused across pairs.
  dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
  ref: dict[int, tuple[np.ndarray, np.ndarray]] = {}
  for k in range(1, len(frames)):
    a = np.ascontiguousarray(frames[k - 1])
    b = np.ascontiguousarray(frames[k])
    dis.setVariationalRefinementIterations(_DIS_VARIATIONAL_ITERS)
    fwd = dis.calc(a, b, None)
    dis.setVariationalRefinementIterations(_DIS_BACKWARD_VARIATIONAL_ITERS)
    ref[k] = (fwd, dis.calc(b, a, None))

  par = dict(enumerate(_iter_dis_flows(frames, range(1, len(frames))), start=1))

  assert set(par) == set(ref)
  for k in ref:
    assert np.array_equal(par[k][0], ref[k][0]), f"fwd flow differs at pair {k}"
    assert np.array_equal(par[k][1], ref[k][1]), f"bwd flow differs at pair {k}"


@pytest.mark.skipif(not cuda_available(), reason="needs CUDA")
def test_gpu_flow_tracks_cv2_dis() -> None:
  frames = translating_frames(6, shift=3, seed=2, height=256, width=256)
  engine = gpu_flow.prime(
    frames, _DIS_VARIATIONAL_ITERS, _DIS_BACKWARD_VARIATIONAL_ITERS
  )
  try:
    gpu = list(engine.iter_flows(frames, range(1, 6)))
  finally:
    gpu_flow.shutdown()
  cpu = list(_iter_dis_flows(frames, range(1, 6)))
  for g, c in zip(gpu, cpu, strict=True):
    for g_flow, c_flow in zip(g, c, strict=True):
      assert np.median(np.abs(g_flow - c_flow)) < _GPU_FLOW_TOL
