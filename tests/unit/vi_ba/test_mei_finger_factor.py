"""MEI finger and base-tag factors: zero residual, Jacobians and the bearing guard."""

from __future__ import annotations

import math
from dataclasses import astuple

import gtsam
import numpy as np
import numpy.typing as npt
from specialist.vi_ba._mei_factor_cpp import (
  make_mei_base_tag_factor_cpp,
  make_mei_finger_factor_cpp,
)

from specialist.loader.mei_model import MeiParams, mei_project

_THETA_KEY = 0
_BASE_TAG_KEY = 1


def _params() -> MeiParams:
  # MEI with light distortion + a super-fisheye xi.
  return MeiParams(
    fx=800.0,
    fy=800.0,
    cx=960.0,
    cy=960.0,
    k1=0.02,
    k2=-0.01,
    k3=0.003,
    p1=0.001,
    p2=-0.002,
    xi=1.95,
  )


def _noise() -> gtsam.noiseModel.Base:
  # Isotropic σ=1 px → whitened == unwhitened; makes Jacobian introspection easy.
  return gtsam.noiseModel.Isotropic.Sigma(2, 1.0)


def _body_T_lens() -> gtsam.Pose3:
  # A small, non-trivial body_T_lens (not identity) to exercise R_lb.
  return gtsam.Pose3(
    gtsam.Rot3.RzRyRx(0.05, -0.03, 0.02), gtsam.Point3(0.01, -0.02, 0.015)
  )


def _body_T_base_tag() -> gtsam.Pose3:
  # The base tag sits ~0.5 m in front of the body, rotated (camera sees it).
  return gtsam.Pose3(gtsam.Rot3.RzRyRx(0.1, 0.2, -0.15), gtsam.Point3(0.0, 0.0, 0.5))


def _base_T_joint() -> gtsam.Pose3:
  # The finger joint origin offset from the base tag, slightly rotated.
  return gtsam.Pose3(
    gtsam.Rot3.RzRyRx(-0.08, 0.05, 0.12), gtsam.Point3(0.04, 0.02, 0.0)
  )


def _corner() -> npt.NDArray[np.float64]:
  # Tag corner in the joint frame at theta=0: URDF tag offset plus corner half-width.
  return np.array([0.045, 0.025, 0.0])


def _finger_p_lens(
  theta: float,
  body_T_lens: gtsam.Pose3,
  body_T_base_tag: gtsam.Pose3,
  base_T_joint: gtsam.Pose3,
  c: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
  """Replicate the binary finger factor's forward kinematics: corner -> lens."""
  ct, st = math.cos(theta), math.sin(theta)
  p_pivot = np.array([c[0] * ct - c[1] * st, c[0] * st + c[1] * ct, c[2]])
  p_joint = base_T_joint.transformFrom(gtsam.Point3(*p_pivot))
  p_body = body_T_base_tag.transformFrom(gtsam.Point3(*p_joint))
  return np.asarray(body_T_lens.transformTo(gtsam.Point3(*p_body)), dtype=np.float64)


def _finger_factor(
  pixel: npt.NDArray[np.float64],
  body_T_lens: gtsam.Pose3,
  base_T_joint: gtsam.Pose3,
  c: npt.NDArray[np.float64],
  params: MeiParams,
  obs_bearing: npt.NDArray[np.float64] | None = None,
) -> gtsam.NonlinearFactor:
  bearing = np.zeros(3) if obs_bearing is None else np.asarray(obs_bearing, float)
  return make_mei_finger_factor_cpp(
    _THETA_KEY,
    _BASE_TAG_KEY,
    np.asarray(pixel, dtype=np.float64).reshape(2),
    _noise(),
    body_T_lens,
    base_T_joint,
    np.asarray(c, dtype=np.float64).reshape(3),
    np.array(astuple(params)),
    bearing,
  )


def _base_tag_factor(
  pixel: npt.NDArray[np.float64],
  body_T_lens: gtsam.Pose3,
  c: npt.NDArray[np.float64],
  params: MeiParams,
  obs_bearing: npt.NDArray[np.float64] | None = None,
) -> gtsam.NonlinearFactor:
  bearing = np.zeros(3) if obs_bearing is None else np.asarray(obs_bearing, float)
  return make_mei_base_tag_factor_cpp(
    _BASE_TAG_KEY,
    np.asarray(pixel, dtype=np.float64).reshape(2),
    _noise(),
    body_T_lens,
    np.asarray(c, dtype=np.float64).reshape(3),
    np.array(astuple(params)),
    bearing,
  )


def _finger_values(theta: float, pose: gtsam.Pose3) -> gtsam.Values:
  v = gtsam.Values()
  v.insert(_THETA_KEY, float(theta))
  v.insert(_BASE_TAG_KEY, pose)
  return v


def _pose_values(pose: gtsam.Pose3) -> gtsam.Values:
  v = gtsam.Values()
  v.insert(_BASE_TAG_KEY, pose)
  return v


def _fd_pose_columns(
  residual_at, pose: gtsam.Pose3, h: float = 1e-6
) -> npt.NDArray[np.float64]:
  """Central finite difference of a residual w.r.t. the Pose3 retract tangent."""
  cols = []
  for i in range(6):
    xi = np.zeros(6)
    xi[i] = h
    r_plus = residual_at(pose.retract(xi))
    r_minus = residual_at(pose.retract(-xi))
    cols.append((r_plus - r_minus) / (2.0 * h))
  return np.column_stack(cols)  # 2x6


# --- binary finger factor -------------------------------------------------


def test_finger_zero_residual_at_reprojecting_config() -> None:
  params = _params()
  ext, btag, btj, c = _body_T_lens(), _body_T_base_tag(), _base_T_joint(), _corner()
  theta0 = 0.3
  p_lens = _finger_p_lens(theta0, ext, btag, btj, c)
  assert p_lens[2] > 0.0
  pixel = mei_project(p_lens[None, :], params)[0]
  factor = _finger_factor(pixel, ext, btj, c, params)
  err = factor.unwhitenedError(_finger_values(theta0, btag))
  assert np.allclose(err, 0.0, atol=1e-7), err


def test_finger_nonzero_residual_when_perturbed() -> None:
  params = _params()
  ext, btag, btj, c = _body_T_lens(), _body_T_base_tag(), _base_T_joint(), _corner()
  theta0 = 0.3
  pixel = mei_project(_finger_p_lens(theta0, ext, btag, btj, c)[None, :], params)[0]
  factor = _finger_factor(pixel, ext, btj, c, params)
  err = factor.unwhitenedError(_finger_values(theta0 + 0.2, btag))
  assert np.linalg.norm(err) > 1.0


def test_finger_jacobians_match_finite_difference() -> None:
  """Analytic H1 (theta) and H2 (base-tag pose) match a central finite diff."""
  params = _params()
  ext, btag, btj, c = _body_T_lens(), _body_T_base_tag(), _base_T_joint(), _corner()
  # Shifted pixel so the residual is non-zero away from any optimum.
  pixel = mei_project(_finger_p_lens(0.0, ext, btag, btj, c)[None, :], params)[0]
  pixel = pixel + np.array([3.0, -2.0])
  factor = _finger_factor(pixel, ext, btj, c, params)

  h = 1e-6
  for theta in (-0.6, -0.25, 0.0, 0.15, 0.4, 0.75):
    a, _ = factor.linearize(_finger_values(theta, btag)).jacobian()
    assert a.shape == (2, 7), f"theta={theta}: unexpected shape {a.shape}"
    h1, h2 = a[:, 0:1], a[:, 1:7]

    r_plus = factor.unwhitenedError(_finger_values(theta + h, btag))
    r_minus = factor.unwhitenedError(_finger_values(theta - h, btag))
    fd_theta = (r_plus - r_minus) / (2.0 * h)
    assert np.allclose(h1[:, 0], fd_theta, atol=1e-4), (
      f"theta={theta}: H1={h1[:, 0]}, fd={fd_theta}"
    )

    fd_pose = _fd_pose_columns(
      lambda p, th=theta: factor.unwhitenedError(_finger_values(th, p)), btag
    )
    assert np.allclose(h2, fd_pose, atol=1e-4), f"theta={theta}: H2={h2}, fd={fd_pose}"


def test_finger_obs_bearing_deactivates_opposite_hemisphere() -> None:
  params = _params()
  ext, btag, btj, c = _body_T_lens(), _body_T_base_tag(), _base_T_joint(), _corner()
  theta0 = 0.3
  p_lens = _finger_p_lens(theta0, ext, btag, btj, c)
  pixel = mei_project(p_lens[None, :], params)[0] + np.array([5.0, -5.0])
  opposite = -p_lens / np.linalg.norm(p_lens)

  off = _finger_factor(pixel, ext, btj, c, params)
  assert np.linalg.norm(off.unwhitenedError(_finger_values(theta0, btag))) > 0.0

  on = _finger_factor(pixel, ext, btj, c, params, obs_bearing=opposite)
  v = _finger_values(theta0, btag)
  assert np.allclose(on.unwhitenedError(v), 0.0)
  a, _ = on.linearize(v).jacobian()
  assert np.allclose(a, 0.0)


# --- unary base-tag factor ------------------------------------------------


def test_base_tag_zero_residual_and_jacobian() -> None:
  params = _params()
  ext, btag = _body_T_lens(), _body_T_base_tag()
  c = np.array([0.004, -0.003, 0.0])  # base tag corner in the base-tag frame
  p_body = btag.transformFrom(gtsam.Point3(*c))
  p_lens = np.asarray(ext.transformTo(gtsam.Point3(*p_body)), dtype=np.float64)
  assert p_lens[2] > 0.0
  pixel = mei_project(p_lens[None, :], params)[0]

  factor = _base_tag_factor(pixel, ext, c, params)
  assert np.allclose(factor.unwhitenedError(_pose_values(btag)), 0.0, atol=1e-7)

  # Shifted pixel for a non-zero residual; analytic H1 == finite diff.
  factor = _base_tag_factor(pixel + np.array([2.5, -1.5]), ext, c, params)
  a, _ = factor.linearize(_pose_values(btag)).jacobian()
  assert a.shape == (2, 6)
  fd = _fd_pose_columns(lambda p: factor.unwhitenedError(_pose_values(p)), btag)
  assert np.allclose(a, fd, atol=1e-4), f"H1={a}, fd={fd}"
