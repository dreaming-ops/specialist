"""``_mei_factor_cpp``: zero residual, Jacobians, past-90° cheirality, bearing guard."""

from __future__ import annotations

import math
from dataclasses import astuple

import gtsam
import numpy as np
import numpy.typing as npt
from specialist.vi_ba._mei_factor_cpp import make_mei_factor_cpp

from specialist.loader.mei_model import MeiParams, mei_project, mei_unproject
from tests.mei_fixtures import unit_pixel_noise

_POSE_KEY = 0
_LM_KEY = 1


def _params() -> MeiParams:
  # Realistic values (from the committed offset_v3 fixture).
  return MeiParams(
    fx=4625.45,
    fy=4623.58,
    cx=2968.61,
    cy=2994.87,
    k1=0.39252487,
    k2=1.23661351,
    k3=-3.86204076,
    p1=-1.075e-05,
    p2=-0.00215951,
    xi=1.94817,
  )


def _factor(
  pixel: npt.NDArray[np.float64],
  body_T_lens: gtsam.Pose3,
  params: MeiParams,
  obs_bearing: npt.NDArray[np.float64] | None = None,
) -> gtsam.NonlinearFactor:
  bearing = np.zeros(3) if obs_bearing is None else np.asarray(obs_bearing, float)
  return make_mei_factor_cpp(
    _POSE_KEY,
    _LM_KEY,
    np.asarray(pixel, dtype=np.float64).reshape(2),
    unit_pixel_noise(),
    body_T_lens,
    np.array(astuple(params)),
    bearing,
  )


def _rear_body_T_lens() -> gtsam.Pose3:
  return gtsam.Pose3(
    gtsam.Rot3.Ypr(0.0, 0.0, math.pi / 2), gtsam.Point3(0.0, 0.0, -0.026248)
  )


def _values(pose: gtsam.Pose3, lm: npt.NDArray[np.float64]) -> gtsam.Values:
  v = gtsam.Values()
  v.insert(_POSE_KEY, pose)
  v.insert(_LM_KEY, gtsam.Point3(float(lm[0]), float(lm[1]), float(lm[2])))
  return v


def _project_through(
  body_pose: gtsam.Pose3,
  landmark_world: npt.NDArray[np.float64],
  body_T_lens: gtsam.Pose3,
  params: MeiParams,
) -> npt.NDArray[np.float64]:
  T_world_lens = body_pose.compose(body_T_lens)
  p_lens = T_world_lens.transformTo(
    gtsam.Point3(
      float(landmark_world[0]), float(landmark_world[1]), float(landmark_world[2])
    )
  )
  return mei_project(np.asarray(p_lens)[None, :], params)[0]


def test_zero_residual_front_identity_extrinsic() -> None:
  params = _params()
  body_T_lens = gtsam.Pose3()
  body_pose = gtsam.Pose3()
  landmark = np.array([0.3, 0.1, 2.0])  # in front of lens
  pixel = _project_through(body_pose, landmark, body_T_lens, params)

  factor = _factor(pixel, body_T_lens, params)
  err = factor.unwhitenedError(_values(body_pose, landmark))
  assert np.allclose(err, 0.0, atol=1e-10)


def test_zero_residual_rear_extrinsic() -> None:
  params = _params()
  body_T_lens = _rear_body_T_lens()
  body_pose = gtsam.Pose3()
  # A landmark in front of the rear lens, along its +z, then carried into world.
  p_in_lens = np.array([0.1, -0.1, 1.5])
  T_world_lens = body_T_lens  # body == world here
  landmark_world = np.asarray(T_world_lens.transformFrom(gtsam.Point3(*p_in_lens)))
  pixel = _project_through(body_pose, landmark_world, body_T_lens, params)

  factor = _factor(pixel, body_T_lens, params)
  err = factor.unwhitenedError(_values(body_pose, landmark_world))
  assert np.allclose(err, 0.0, atol=1e-10)


def test_nonzero_residual_when_pose_perturbed() -> None:
  params = _params()
  body_T_lens = gtsam.Pose3()
  body_pose = gtsam.Pose3()
  landmark = np.array([0.3, 0.1, 2.0])
  pixel = _project_through(body_pose, landmark, body_T_lens, params)

  factor = _factor(pixel, body_T_lens, params)

  perturbed = body_pose.retract(np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0]))
  err = factor.unwhitenedError(_values(perturbed, landmark))
  assert np.linalg.norm(err) > 1.0  # 5 cm translation ≫ 1 px noise


_FOV_LANDMARKS: tuple[tuple[npt.NDArray[np.float64], str], ...] = (
  (np.array([0.01, 0.01, 2.0]), "near-axial"),
  (np.array([0.5, 0.2, 1.5]), "~20° off axis"),
  (np.array([1.0, 0.3, 0.9]), "~50° off axis"),
  (np.array([1.6, 0.0, 0.3]), "~80° off axis"),
  (np.array([1.0, 0.0, -0.1]), "past 90° half-angle (super-fisheye)"),
)


def _assert_fd_jacobian(
  factor: gtsam.NonlinearFactor,
  body_pose: gtsam.Pose3,
  landmark: npt.NDArray[np.float64],
  jac: npt.NDArray[np.float64],
  kind: str,
  label: str,
) -> None:
  """Assert ``jac · δ ≈ r(x ⊞ δ) − r(x)`` along each perturbation axis."""
  base = factor.unwhitenedError(_values(body_pose, landmark))
  for axis in range(jac.shape[1]):
    delta = np.zeros(jac.shape[1])
    delta[axis] = 1e-5
    stepped = (
      _values(body_pose.retract(delta), landmark)
      if kind == "pose"
      else _values(body_pose, landmark + delta)
    )
    observed = factor.unwhitenedError(stepped) - base
    assert np.allclose(jac @ delta, observed, atol=1e-4), (
      f"{label}: {kind} axis {axis}: pred={jac @ delta}, obs={observed}"
    )


def test_jacobian_self_consistency_across_fov() -> None:
  """At ≥5 FOV-spanning landmarks, J·δ ≈ r(x ⊞ δ) − r(x) for small δ."""
  params = _params()
  body_T_lens = gtsam.Pose3()
  body_pose = gtsam.Pose3()

  for landmark, label in _FOV_LANDMARKS:
    pixel = _project_through(body_pose, landmark, body_T_lens, params)
    # A shifted pixel puts the Jacobian at a non-zero residual, as LM evaluates it.
    factor = _factor(pixel + np.array([2.0, -3.0]), body_T_lens, params)
    a, _ = factor.linearize(_values(body_pose, landmark)).jacobian()
    assert a.shape == (2, 9), f"{label}: unexpected Jacobian shape {a.shape}"
    _assert_fd_jacobian(factor, body_pose, landmark, a[:, :6], "pose", label)
    _assert_fd_jacobian(factor, body_pose, landmark, a[:, 6:], "landmark", label)


def test_mei_cheirality_past_90_degrees_preserves_hemisphere() -> None:
  """A 95° half-angle ray plus ±1 px noise unprojects into its own hemisphere."""
  params = _params()
  rng = np.random.default_rng(seed=42)

  half_angle = math.radians(95.0)
  base_ray = np.array([math.sin(half_angle), 0.0, math.cos(half_angle)])
  assert base_ray[2] < 0.0  # past horizon; sanity.

  depth = 1.5
  cam_points = (base_ray * depth)[None, :]
  pixel_true = mei_project(cam_points, params)[0]
  assert np.all(np.isfinite(pixel_true)), "MEI must represent 95° half-angle"

  for _ in range(100):
    noise = rng.normal(scale=1.0, size=2)
    pixel_noisy = pixel_true + noise
    bearing = mei_unproject(pixel_noisy[None, :], params)[0]
    # Hemisphere check: recovered z should be close to base z, NOT flipped sign.
    assert bearing[2] < 0.2, (
      f"cheirality flip: base z={base_ray[2]:.3f}, recovered z={bearing[2]:.3f}"
    )
    assert np.linalg.norm(bearing - base_ray) < 0.05


def test_obs_bearing_deactivates_opposite_hemisphere() -> None:
  """A point opposite the observed ray zeroes the residual and the Jacobian."""
  params = _params()
  body_T_lens = gtsam.Pose3()
  body_pose = gtsam.Pose3()
  behind = np.array([1.0, 0.1, -0.36])  # 110 deg off-axis, short of the 121 deg fold
  pixel = mei_project(behind[None, :], params)[0]
  assert np.all(np.isfinite(pixel)), "ξ>1 omni model projects behind-camera points"
  forward = np.array([0.0, 0.0, 1.0])  # the tag was observed in front

  # Without a bearing the factor is active even behind the camera (the basin).
  off = _factor(pixel + np.array([4.0, -4.0]), body_T_lens, params)
  assert np.linalg.norm(off.unwhitenedError(_values(body_pose, behind))) > 0.0

  # With the forward bearing the opposite-side point deactivates entirely.
  on = _factor(pixel + np.array([4.0, -4.0]), body_T_lens, params, obs_bearing=forward)
  v = _values(body_pose, behind)
  assert np.allclose(on.unwhitenedError(v), 0.0)
  a, _ = on.linearize(v).jacobian()
  assert np.allclose(a, 0.0)


def test_obs_bearing_keeps_same_hemisphere_active() -> None:
  """A peripheral past-90° (z<0) detection on the same side stays fully active."""
  params = _params()
  body_T_lens = gtsam.Pose3()
  body_pose = gtsam.Pose3()
  periph = np.array([1.0, 0.0, -0.1])  # ~95° half-angle, z<0 but in-FOV
  pixel = mei_project(periph[None, :], params)[0] + np.array([2.0, -3.0])
  bearing = periph / np.linalg.norm(periph)  # observed along the same ray
  v = _values(body_pose, periph)
  off = _factor(pixel, body_T_lens, params).unwhitenedError(v)
  on = _factor(pixel, body_T_lens, params, obs_bearing=bearing).unwhitenedError(v)
  assert np.linalg.norm(on) > 0.0  # active, not deactivated
  assert np.allclose(off, on)  # bearing on the observed side changes nothing
