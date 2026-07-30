"""Decoding an Insta360 ``offset_v3`` array into two body-to-lens ``Pose3`` values."""

from __future__ import annotations

import json
import math
from pathlib import Path

import gtsam
import numpy as np
import numpy.typing as npt
import pytest

from specialist.record.offset_v3 import (
  LensOffset,
  OffsetV3,
  parse_offset_v3,
)

_FIXTURE_JSON = (
  Path(__file__).resolve().parents[2] / "fixtures" / "vi_ba" / "offset_v3_x3.json"
)


def _block(
  xi: float = 1.948,
  fx: float = 4625.0,
  fy: float = 4625.0,
  cx: float = 2968.0,
  cy: float = 2994.0,
  ypr_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
  t: tuple[float, float, float] = (0.0, 0.0, 0.0),
  k: tuple[float, float, float] = (0.0, 0.0, 0.0),
  p: tuple[float, float] = (0.0, 0.0),
  w: int = 11904,
  h: int = 5952,
) -> list[float]:
  yaw, pitch, roll = ypr_deg
  tx, ty, tz = t
  k1, k2, k3 = k
  p1, p2 = p
  return [
    xi,
    fx,
    fy,
    cx,
    cy,
    yaw,
    pitch,
    roll,
    tx,
    ty,
    tz,
    k1,
    k2,
    k3,
    p1,
    p2,
    float(w),
    float(h),
    71.0,
  ]


def _make_offset_v3(
  front: list[float] | None = None,
  rear: list[float] | None = None,
) -> npt.NDArray[np.float64]:
  front = front if front is not None else _block()
  rear = rear if rear is not None else _block(cx=8931.88, t=(0.0, 0.0, -0.026248))
  return np.asarray([2.0, *front, *rear, 199424.0], dtype=np.float64)


def _load_real_offset_v3() -> npt.NDArray[np.float64]:
  payload = json.loads(_FIXTURE_JSON.read_text())
  return np.asarray(payload["offset_v3"], dtype=np.float64)


def test_parse_offset_v3_returns_front_and_rear() -> None:
  ex = parse_offset_v3(_make_offset_v3())
  assert isinstance(ex, OffsetV3)
  assert isinstance(ex.front, LensOffset)
  assert isinstance(ex.rear, LensOffset)


def test_parse_offset_v3_field_round_trip() -> None:
  front = _block(
    xi=1.95,
    fx=4625.45,
    fy=4623.58,
    cx=2968.61,
    cy=2994.87,
    ypr_deg=(0.581, 0.714, 90.712),
    t=(0.0, 0.0, 0.0),
    k=(0.39252487, 1.23661351, -3.86204076),
    p=(-1.075e-05, -0.00215951),
  )
  rear = _block(
    xi=1.95,
    fx=4627.0,
    fy=4627.39,
    cx=8931.88,
    cy=2989.66,
    ypr_deg=(-0.518, 0.41, 90.417),
    t=(-0.001542, -1e-05, -0.026248),
    k=(0.39135978, 1.2525574, -3.91095424),
    p=(-0.00138846, -0.00263395),
  )
  arr = _make_offset_v3(front=front, rear=rear)
  ex = parse_offset_v3(arr)
  names = (
    "xi fx fy cx cy yaw pitch roll tx ty tz k1 k2 k3 p1 p2 frame_width frame_height"
  ).split()
  for lens, block in [(ex.front, front), (ex.rear, rear)]:
    expected = [*block[:5], *map(math.radians, block[5:8]), *block[8:18]]
    assert {n: getattr(lens, n) for n in names} == dict(
      zip(names, expected, strict=True)
    )


def test_parse_offset_v3_wrong_size_raises() -> None:
  with pytest.raises(ValueError, match="40"):
    parse_offset_v3(np.zeros(16, dtype=np.float64))
  with pytest.raises(ValueError, match="40"):
    parse_offset_v3(np.zeros(39, dtype=np.float64))


def test_parse_offset_v3_wrong_lens_count_raises() -> None:
  arr = _make_offset_v3()
  arr[0] = 1.0
  with pytest.raises(ValueError, match="lens count"):
    parse_offset_v3(arr)


def test_lens_offset_to_pose3_identity_rpy() -> None:
  arr = _make_offset_v3(
    front=_block(ypr_deg=(0.0, 0.0, 0.0), t=(0.1, -0.2, 0.3)),
  )
  front = parse_offset_v3(arr).front
  pose = front.to_pose3()
  assert np.allclose(pose.rotation().matrix(), np.eye(3), atol=1e-12)
  assert np.allclose(pose.translation(), (-0.1, 0.2, -0.3), atol=1e-12)


@pytest.mark.parametrize("ypr_deg", [(0.0, 0.0, 90.0), (10.0, 20.0, 30.0)])
def test_lens_offset_to_pose3_matches_ypr_zyx_intrinsic(
  ypr_deg: tuple[float, float, float],
) -> None:
  yaw, pitch, roll = ypr_deg
  arr = _make_offset_v3(
    front=_block(ypr_deg=ypr_deg),
  )
  front = parse_offset_v3(arr).front
  expected = gtsam.Rot3.Ypr(
    math.radians(yaw), math.radians(pitch), math.radians(roll)
  ).matrix()
  assert np.allclose(front.to_pose3().rotation().matrix(), expected, atol=1e-12)


# Real-capture assertions on 40 floats from a real .insv, under tests/fixtures/vi_ba/.
def test_real_offset_v3_fixture_is_forty_floats() -> None:
  arr = _load_real_offset_v3()
  assert arr.shape == (40,)
  assert arr[0] == 2.0


def test_real_offset_v3_front_translation_is_zero() -> None:
  ex = parse_offset_v3(_load_real_offset_v3())
  # Front lens defines body origin.
  assert np.allclose(ex.body_T_front_lens.translation(), (0.0, 0.0, 0.0), atol=1e-6)


def test_real_offset_v3_rear_translation_along_negative_y() -> None:
  ex = parse_offset_v3(_load_real_offset_v3())
  t = ex.body_T_rear_lens.translation()
  # Body frame is the front lens; the rear lens sits ~26 mm along -y.
  assert abs(t[0]) < 5e-3
  assert -0.030 < t[1] < -0.020
  assert abs(t[2]) < 5e-3


def test_real_offset_v3_roll_near_90_degrees_for_both_lenses() -> None:
  ex = parse_offset_v3(_load_real_offset_v3())
  assert abs(math.degrees(ex.front.roll) - 90.0) < 2.0
  assert abs(math.degrees(ex.rear.roll) - 90.0) < 2.0


def test_real_offset_v3_xi_is_super_fisheye() -> None:
  ex = parse_offset_v3(_load_real_offset_v3())
  assert 1.5 < ex.front.xi < 2.5
  assert 1.5 < ex.rear.xi < 2.5
