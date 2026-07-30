"""Cross-tracker MEI tag factors: zero residual, Jacobians and the bearing guard."""

from __future__ import annotations

from dataclasses import astuple

import gtsam
import numpy as np
import numpy.typing as npt
from specialist.vi_ba._mei_factor_cpp import (
  make_mei_cross_finger_factor_cpp,
  make_mei_cross_tag_factor_cpp,
)

from specialist.loader.mei_model import MeiParams, mei_project
from tests.mei_fixtures import distorted_mei_params, fd_pose_columns, unit_pixel_noise

_OBS, _OWN, _BT, _THETA = 0, 1, 2, 3


def _params() -> MeiParams:
  # Light distortion with a moderate super-fisheye xi.
  return distorted_mei_params(xi=1.6)


def _ext() -> gtsam.Pose3:
  return gtsam.Pose3(
    gtsam.Rot3.RzRyRx(0.05, -0.03, 0.02), gtsam.Point3(0.01, -0.02, 0.015)
  )


def _x_o() -> gtsam.Pose3:
  return gtsam.Pose3(
    gtsam.Rot3.RzRyRx(0.1, -0.05, 0.08), gtsam.Point3(0.05, -0.02, 0.0)
  )


def _x_w() -> gtsam.Pose3:
  # Owner ~0.6 m in front of the observer so its tag is visible.
  return gtsam.Pose3(gtsam.Rot3.RzRyRx(-0.1, 0.15, 0.05), gtsam.Point3(0.1, 0.05, 0.6))


def _base_tag() -> gtsam.Pose3:
  return gtsam.Pose3(gtsam.Rot3.RzRyRx(0.1, 0.2, -0.15), gtsam.Point3(0.0, 0.02, 0.03))


def _corner() -> npt.NDArray[np.float64]:
  return np.array([0.004, -0.003, 0.0])


def _p_lens(
  x_o: gtsam.Pose3,
  x_w: gtsam.Pose3,
  base_tag: gtsam.Pose3,
  ext: gtsam.Pose3,
  c: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
  m = base_tag.transformFrom(gtsam.Point3(*c))
  p_world = x_w.transformFrom(gtsam.Point3(*m))
  p_body = x_o.transformTo(gtsam.Point3(*p_world))
  return np.asarray(ext.transformTo(gtsam.Point3(*p_body)), dtype=np.float64)


def _factor(
  pixel: npt.NDArray[np.float64],
  ext: gtsam.Pose3,
  c: npt.NDArray[np.float64],
  params: MeiParams,
  obs_bearing: npt.NDArray[np.float64] | None = None,
) -> gtsam.NonlinearFactor:
  bearing = np.zeros(3) if obs_bearing is None else np.asarray(obs_bearing, float)
  return make_mei_cross_tag_factor_cpp(
    _OBS,
    _OWN,
    _BT,
    np.asarray(pixel, dtype=np.float64).reshape(2),
    unit_pixel_noise(),
    ext,
    np.asarray(c, dtype=np.float64).reshape(3),
    np.array(astuple(params)),
    bearing,
  )


def _values(x_o: gtsam.Pose3, x_w: gtsam.Pose3, bt: gtsam.Pose3) -> gtsam.Values:
  v = gtsam.Values()
  v.insert(_OBS, x_o)
  v.insert(_OWN, x_w)
  v.insert(_BT, bt)
  return v


def test_cross_zero_residual_at_reprojecting_config() -> None:
  params = _params()
  ext, x_o, x_w, bt, c = _ext(), _x_o(), _x_w(), _base_tag(), _corner()
  p_lens = _p_lens(x_o, x_w, bt, ext, c)
  assert p_lens[2] > 0.0
  pixel = mei_project(p_lens[None, :], params)[0]
  f = _factor(pixel, ext, c, params)
  assert np.allclose(f.unwhitenedError(_values(x_o, x_w, bt)), 0.0, atol=1e-6)


def test_cross_jacobians_match_finite_difference() -> None:
  params = _params()
  ext, x_o, x_w, bt, c = _ext(), _x_o(), _x_w(), _base_tag(), _corner()
  pixel = mei_project(_p_lens(x_o, x_w, bt, ext, c)[None, :], params)[0]
  pixel = pixel + np.array([2.0, -1.5])
  f = _factor(pixel, ext, c, params)

  a, _ = f.linearize(_values(x_o, x_w, bt)).jacobian()
  assert a.shape == (2, 18), a.shape
  h1, h2, h3 = a[:, 0:6], a[:, 6:12], a[:, 12:18]
  fd1 = fd_pose_columns(lambda p: f.unwhitenedError(_values(p, x_w, bt)), x_o)
  fd2 = fd_pose_columns(lambda p: f.unwhitenedError(_values(x_o, p, bt)), x_w)
  fd3 = fd_pose_columns(lambda p: f.unwhitenedError(_values(x_o, x_w, p)), bt)
  assert np.allclose(h1, fd1, atol=1e-4), (h1, fd1)
  assert np.allclose(h2, fd2, atol=1e-4), (h2, fd2)
  assert np.allclose(h3, fd3, atol=1e-4), (h3, fd3)


def test_cross_obs_bearing_deactivates_opposite_hemisphere() -> None:
  params = _params()
  ext, x_o, x_w, bt, c = _ext(), _x_o(), _x_w(), _base_tag(), _corner()
  p_lens = _p_lens(x_o, x_w, bt, ext, c)
  pixel = mei_project(p_lens[None, :], params)[0] + np.array([5.0, -5.0])
  opposite = -p_lens / np.linalg.norm(p_lens)
  on = _factor(pixel, ext, c, params, obs_bearing=opposite)
  v = _values(x_o, x_w, bt)
  assert np.allclose(on.unwhitenedError(v), 0.0)
  a, _ = on.linearize(v).jacobian()
  assert np.allclose(a, 0.0)


# --- quaternary cross-finger factor ---------------------------------------


def _base_t_joint() -> gtsam.Pose3:
  return gtsam.Pose3(
    gtsam.Rot3.RzRyRx(-0.08, 0.05, 0.12), gtsam.Point3(0.04, 0.02, 0.0)
  )


def _finger_corner() -> npt.NDArray[np.float64]:
  # Real distance from the joint axis (URDF tag offset + corner half-width).
  return np.array([0.045, 0.025, 0.0])


def _cross_finger_p_lens(
  x_o: gtsam.Pose3,
  x_w: gtsam.Pose3,
  base_tag: gtsam.Pose3,
  theta: float,
  ext: gtsam.Pose3,
  btj: gtsam.Pose3,
  c: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
  ct, st = np.cos(theta), np.sin(theta)
  p_pivot = np.array([c[0] * ct - c[1] * st, c[0] * st + c[1] * ct, c[2]])
  # The joint rotation is just an extra prefix on the corner; reuse _p_lens.
  q = np.asarray(btj.transformFrom(gtsam.Point3(*p_pivot)), dtype=np.float64)
  return _p_lens(x_o, x_w, base_tag, ext, q)


def _cross_finger_factor(
  pixel: npt.NDArray[np.float64],
  ext: gtsam.Pose3,
  btj: gtsam.Pose3,
  c: npt.NDArray[np.float64],
  params: MeiParams,
  obs_bearing: npt.NDArray[np.float64] | None = None,
) -> gtsam.NonlinearFactor:
  bearing = np.zeros(3) if obs_bearing is None else np.asarray(obs_bearing, float)
  return make_mei_cross_finger_factor_cpp(
    _OBS,
    _OWN,
    _BT,
    _THETA,
    np.asarray(pixel, dtype=np.float64).reshape(2),
    unit_pixel_noise(),
    ext,
    btj,
    np.asarray(c, dtype=np.float64).reshape(3),
    np.array(astuple(params)),
    bearing,
  )


def _values4(
  x_o: gtsam.Pose3, x_w: gtsam.Pose3, bt: gtsam.Pose3, theta: float
) -> gtsam.Values:
  v = _values(x_o, x_w, bt)
  v.insert(_THETA, float(theta))
  return v


def test_cross_finger_zero_residual_at_reprojecting_config() -> None:
  params = _params()
  ext, x_o, x_w, bt = _ext(), _x_o(), _x_w(), _base_tag()
  btj, c, theta = _base_t_joint(), _finger_corner(), 0.3
  p_lens = _cross_finger_p_lens(x_o, x_w, bt, theta, ext, btj, c)
  assert p_lens[2] > 0.0
  pixel = mei_project(p_lens[None, :], params)[0]
  f = _cross_finger_factor(pixel, ext, btj, c, params)
  assert np.allclose(f.unwhitenedError(_values4(x_o, x_w, bt, theta)), 0.0, atol=1e-6)


def test_cross_finger_jacobians_match_finite_difference() -> None:
  params = _params()
  ext, x_o, x_w, bt = _ext(), _x_o(), _x_w(), _base_tag()
  btj, c, theta = _base_t_joint(), _finger_corner(), 0.3
  pixel = mei_project(
    _cross_finger_p_lens(x_o, x_w, bt, theta, ext, btj, c)[None, :], params
  )[0] + np.array([2.0, -1.5])
  f = _cross_finger_factor(pixel, ext, btj, c, params)

  a, _ = f.linearize(_values4(x_o, x_w, bt, theta)).jacobian()
  assert a.shape == (2, 19), a.shape
  h1, h2, h3, h4 = a[:, 0:6], a[:, 6:12], a[:, 12:18], a[:, 18:19]
  fd1 = fd_pose_columns(lambda p: f.unwhitenedError(_values4(p, x_w, bt, theta)), x_o)
  fd2 = fd_pose_columns(lambda p: f.unwhitenedError(_values4(x_o, p, bt, theta)), x_w)
  fd3 = fd_pose_columns(lambda p: f.unwhitenedError(_values4(x_o, x_w, p, theta)), bt)
  hh = 1e-6
  fd4 = (
    (
      f.unwhitenedError(_values4(x_o, x_w, bt, theta + hh))
      - f.unwhitenedError(_values4(x_o, x_w, bt, theta - hh))
    )
    / (2 * hh)
  ).reshape(2, 1)
  assert np.allclose(h1, fd1, atol=1e-4), (h1, fd1)
  assert np.allclose(h2, fd2, atol=1e-4), (h2, fd2)
  assert np.allclose(h3, fd3, atol=1e-4), (h3, fd3)
  assert np.allclose(h4, fd4, atol=1e-4), (h4, fd4)
