"""``_iter_dis_flows`` matches a serial shared-instance DIS loop byte for byte."""

from __future__ import annotations

import cv2
import numpy as np

from specialist.vi_ba.features import (
  _DIS_BACKWARD_VARIATIONAL_ITERS,
  _DIS_VARIATIONAL_ITERS,
  _iter_dis_flows,
)
from tests.frame_fixtures import translating_frames


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
