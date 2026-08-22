"""Native MEI factors: zero residual, numeric Jacobians and the bearing guard."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import astuple, dataclass, field

import gtsam
import numpy as np
import numpy.typing as npt
import pytest
from specialist_camera import MeiParams, mei_project, mei_unproject

from specialist.vi_ba import _mei_factor_cpp
from tests.mei_fixtures import distorted_mei_params

Float64Array = npt.NDArray[np.float64]

# Distinct keys per variable role, so one Values serves any factor arity.
_POSE, _LM, _OBS, _OWN, _BT, _THETA, _SEAT = 0, 1, 2, 3, 4, 5, 6
_NOISE = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)


def _real_params() -> MeiParams:
  """The intrinsics of the offset_v3 fixture."""
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


def _rear_body_T_lens() -> gtsam.Pose3:
  return gtsam.Pose3(
    gtsam.Rot3.Ypr(0.0, 0.0, math.pi / 2), gtsam.Point3(0.0, 0.0, -0.026248)
  )


def _small_body_T_lens() -> gtsam.Pose3:
  return gtsam.Pose3(
    gtsam.Rot3.RzRyRx(0.05, -0.03, 0.02), gtsam.Point3(0.01, -0.02, 0.015)
  )


def _body_T_base_tag() -> gtsam.Pose3:
  """The base tag ~0.5 m in front of the body, rotated so the camera sees it."""
  return gtsam.Pose3(gtsam.Rot3.RzRyRx(0.1, 0.2, -0.15), gtsam.Point3(0.0, 0.0, 0.5))


def _base_T_joint() -> gtsam.Pose3:
  """The finger joint origin offset from the base tag, slightly rotated."""
  return gtsam.Pose3(
    gtsam.Rot3.RzRyRx(-0.08, 0.05, 0.12), gtsam.Point3(0.04, 0.02, 0.0)
  )


def _tag_corner() -> Float64Array:
  return np.array([0.004, -0.003, 0.0])


def _finger_corner() -> Float64Array:
  """The joint-frame tag corner at theta=0, at the real distance from the joint axis."""
  return np.array([0.045, 0.025, 0.0])


def _maker(
  factory: Callable[..., gtsam.NonlinearFactor],
  keys: tuple[int, ...],
  geometry: tuple[object, ...],
  params: MeiParams,
  **extra: int,
) -> Callable[[Float64Array, Float64Array], gtsam.NonlinearFactor]:
  """Native factories take keys, pixel, noise, geometry, intrinsics, then bearing."""
  return lambda pixel, bearing: factory(
    *keys, pixel, _NOISE, *geometry, np.array(astuple(params)), bearing, **extra
  )


def _values(**variables: object) -> gtsam.Values:
  out = gtsam.Values()
  for key, value in variables.items():
    out.insert(globals()[f"_{key.upper()}"], value)
  return out


def _pivoted(c: Float64Array, theta: float) -> Float64Array:
  """The corner after the revolute joint has turned by ``theta`` about +z."""
  ct, st = math.cos(theta), math.sin(theta)
  return np.array([c[0] * ct - c[1] * st, c[0] * st + c[1] * ct, c[2]])


def _into_lens(body_T_lens: gtsam.Pose3, p_body: Float64Array) -> Float64Array:
  return np.asarray(body_T_lens.transformTo(gtsam.Point3(*p_body)), dtype=np.float64)


def _through(pose: gtsam.Pose3, point: Float64Array) -> Float64Array:
  return np.asarray(pose.transformFrom(gtsam.Point3(*point)), dtype=np.float64)


@dataclass(frozen=True)
class _Case:
  """One factor, its linearization variables, and the configurations that stress it."""

  name: str
  params: MeiParams
  # (pixel, obs_bearing) -> the factor under test
  make: Callable[[Float64Array, Float64Array], gtsam.NonlinearFactor]
  # (key, kind, dimension) per variable, in Jacobian column order
  blocks: tuple[tuple[int, str, int], ...]
  # (label, values, the point those values put in the lens frame)
  configurations: tuple[tuple[str, gtsam.Values, Float64Array], ...]
  residual_atol: float
  # A pixel shift that moves the Jacobian check off the optimum.
  offset: Float64Array = field(default_factory=lambda: np.array([2.0, -3.0]))
  step: float = 1e-4

  @property
  def columns(self) -> int:
    return sum(dim for _key, _kind, dim in self.blocks)


def _mei_case(name: str, body_T_lens: gtsam.Pose3, params: MeiParams) -> _Case:
  """``MeiFactorCpp`` at a body pose of identity, so body == world."""
  body_pose = gtsam.Pose3()
  landmarks: tuple[tuple[str, Float64Array], ...] = (
    ("near-axial", np.array([0.01, 0.01, 2.0])),
    ("~20° off axis", np.array([0.5, 0.2, 1.5])),
    ("~50° off axis", np.array([1.0, 0.3, 0.9])),
    ("~80° off axis", np.array([1.6, 0.0, 0.3])),
    ("past 90° half-angle (super-fisheye)", np.array([1.0, 0.0, -0.1])),
  )
  return _Case(
    name=name,
    params=params,
    make=_maker(
      _mei_factor_cpp.make_mei_factor_cpp, (_POSE, _LM), (body_T_lens,), params
    ),
    blocks=((_POSE, "pose", 6), (_LM, "point", 3)),
    configurations=tuple(
      (
        label,
        _values(pose=body_pose, lm=gtsam.Point3(*_through(body_T_lens, lens_xyz))),
        lens_xyz,
      )
      for label, lens_xyz in landmarks
    ),
    residual_atol=1e-10,
  )


def _cross_lens_point(
  world_T_observer: gtsam.Pose3,
  world_T_owner: gtsam.Pose3,
  body_T_base_tag: gtsam.Pose3,
  body_T_lens: gtsam.Pose3,
  c: Float64Array,
) -> Float64Array:
  """The owner's tag corner, carried to the world and back into the observer's lens."""
  p_world = _through(world_T_owner, _through(body_T_base_tag, c))
  p_body = np.asarray(
    world_T_observer.transformTo(gtsam.Point3(*p_world)), dtype=np.float64
  )
  return _into_lens(body_T_lens, p_body)


def _anchor_tag_case() -> _Case:
  """``MeiAnchorTagFactorCpp``, with the wall tag's world pose as a variable."""
  params = distorted_mei_params(xi=1.9)
  body_T_lens, c = _small_body_T_lens(), _tag_corner()
  body = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.07, -0.04, 0.11), gtsam.Point3(0.2, 0.1, 0.0))
  anchor = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.2, 0.1, -0.05), gtsam.Point3(0.3, 0.1, 1.0))
  return _Case(
    name="anchor tag",
    params=params,
    make=_maker(
      _mei_factor_cpp.make_mei_anchor_tag_factor_cpp,
      (_POSE, _OWN),
      (body_T_lens, c),
      params,
    ),
    blocks=((_POSE, "pose", 6), (_OWN, "pose", 6)),
    # Swept out to the rim, where the corner noise is priced widest.
    configurations=tuple(
      (
        label,
        _values(pose=body, own=gtsam.Pose3(anchor.rotation(), gtsam.Point3(*xyz))),
        _into_lens(
          body_T_lens,
          np.asarray(
            body.transformTo(
              gtsam.Point3(
                *_through(gtsam.Pose3(anchor.rotation(), gtsam.Point3(*xyz)), c)
              )
            )
          ),
        ),
      )
      for label, xyz in (
        ("frontal", (0.3, 0.1, 1.0)),
        ("~50° off axis", (1.2, 0.3, 0.9)),
        ("~80° off axis", (1.8, 0.1, 0.3)),
        ("past 90° half-angle", (1.2, 0.0, -0.1)),
      )
    ),
    residual_atol=1e-6,
    offset=np.array([2.0, -1.5]),
  )


def _seated_cases() -> tuple[_Case, ...]:
  """The seated variants, each with a 3-column Rot3 seat block as its last variable."""
  params = distorted_mei_params(xi=1.95)
  seat, body_T_base_tag, base_T_joint = (
    _small_body_T_lens(),
    _body_T_base_tag(),
    _base_T_joint(),
  )
  seat_rot = seat.rotation()
  body = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.07, -0.04, 0.11), gtsam.Point3(0.2, 0.1, 0.0))
  anchor = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.2, 0.1, -0.05), gtsam.Point3(0.3, 0.1, 1.0))
  world_T_observer = gtsam.Pose3(
    gtsam.Rot3.RzRyRx(0.1, -0.05, 0.08), gtsam.Point3(0.05, -0.02, 0.0)
  )
  world_T_owner = gtsam.Pose3(
    gtsam.Rot3.RzRyRx(-0.1, 0.15, 0.05), gtsam.Point3(0.1, 0.05, 0.6)
  )
  theta = 0.3
  offset = np.array([2.0, -1.5])

  # The stitched-convention seat; the base tag reprojects through its 90-degree roll.
  rear = _rear_body_T_lens()
  rear_lens_xyz = np.array([0.2, -0.1, 0.7])
  body_R_base_tag = gtsam.Rot3.RzRyRx(0.1, 0.2, -0.15)
  rear_body_T_base_tag = gtsam.Pose3(
    body_R_base_tag,
    gtsam.Point3(
      *(_through(rear, rear_lens_xyz) - body_R_base_tag.matrix() @ _tag_corner())
    ),
  )

  return (
    _Case(
      name="seated anchor tag",
      params=params,
      make=_maker(
        _mei_factor_cpp.make_mei_anchor_tag_factor_cpp,
        (_POSE, _OWN),
        (seat, _tag_corner()),
        params,
        seat_key=_SEAT,
      ),
      blocks=((_POSE, "pose", 6), (_OWN, "pose", 6), (_SEAT, "rot", 3)),
      configurations=(
        (
          "frontal",
          _values(pose=body, own=anchor, seat=seat_rot),
          _into_lens(
            seat,
            np.asarray(
              body.transformTo(gtsam.Point3(*_through(anchor, _tag_corner())))
            ),
          ),
        ),
      ),
      residual_atol=1e-6,
      offset=offset,
    ),
    _Case(
      name="seated base tag",
      params=params,
      make=_maker(
        _mei_factor_cpp.make_mei_seated_base_tag_factor_cpp,
        (_BT, _SEAT),
        (seat, _tag_corner()),
        params,
      ),
      blocks=((_BT, "pose", 6), (_SEAT, "rot", 3)),
      configurations=(
        (
          "frontal",
          _values(bt=body_T_base_tag, seat=seat_rot),
          _into_lens(seat, _through(body_T_base_tag, _tag_corner())),
        ),
      ),
      residual_atol=1e-7,
      offset=offset,
    ),
    _Case(
      name="seated base tag rear seat",
      params=params,
      make=_maker(
        _mei_factor_cpp.make_mei_seated_base_tag_factor_cpp,
        (_BT, _SEAT),
        (rear, _tag_corner()),
        params,
      ),
      blocks=((_BT, "pose", 6), (_SEAT, "rot", 3)),
      configurations=(
        (
          "frontal",
          _values(bt=rear_body_T_base_tag, seat=rear.rotation()),
          rear_lens_xyz,
        ),
      ),
      residual_atol=1e-7,
      offset=offset,
    ),
    _Case(
      name="seated finger",
      params=params,
      make=_maker(
        _mei_factor_cpp.make_mei_seated_finger_factor_cpp,
        (_THETA, _BT, _SEAT),
        (seat, base_T_joint, _finger_corner()),
        params,
      ),
      blocks=((_THETA, "scalar", 1), (_BT, "pose", 6), (_SEAT, "rot", 3)),
      configurations=tuple(
        (
          f"theta={sweep}",
          _values(theta=float(sweep), bt=body_T_base_tag, seat=seat_rot),
          _into_lens(
            seat,
            _through(
              body_T_base_tag, _through(base_T_joint, _pivoted(_finger_corner(), sweep))
            ),
          ),
        )
        for sweep in (-0.6, -0.25, 0.0, 0.15, 0.3, 0.4, 0.75)
      ),
      residual_atol=1e-7,
      offset=offset,
    ),
    _Case(
      name="seated cross tag",
      params=params,
      make=_maker(
        _mei_factor_cpp.make_mei_seated_cross_tag_factor_cpp,
        (_OBS, _OWN, _BT, _SEAT),
        (seat, _tag_corner()),
        params,
      ),
      blocks=(
        (_OBS, "pose", 6),
        (_OWN, "pose", 6),
        (_BT, "pose", 6),
        (_SEAT, "rot", 3),
      ),
      configurations=(
        (
          "frontal",
          _values(
            obs=world_T_observer, own=world_T_owner, bt=body_T_base_tag, seat=seat_rot
          ),
          _cross_lens_point(
            world_T_observer, world_T_owner, body_T_base_tag, seat, _tag_corner()
          ),
        ),
      ),
      residual_atol=1e-6,
      offset=offset,
    ),
    _Case(
      name="seated cross finger",
      params=params,
      make=_maker(
        _mei_factor_cpp.make_mei_seated_cross_finger_factor_cpp,
        (_OBS, _OWN, _BT, _THETA, _SEAT),
        (seat, base_T_joint, _finger_corner()),
        params,
      ),
      blocks=(
        (_OBS, "pose", 6),
        (_OWN, "pose", 6),
        (_BT, "pose", 6),
        (_THETA, "scalar", 1),
        (_SEAT, "rot", 3),
      ),
      configurations=(
        (
          "frontal",
          _values(
            obs=world_T_observer,
            own=world_T_owner,
            bt=body_T_base_tag,
            theta=theta,
            seat=seat_rot,
          ),
          _cross_lens_point(
            world_T_observer,
            world_T_owner,
            body_T_base_tag,
            seat,
            _through(base_T_joint, _pivoted(_finger_corner(), theta)),
          ),
        ),
      ),
      residual_atol=1e-6,
      offset=offset,
    ),
  )


def _cases() -> tuple[_Case, ...]:
  return (
    _mei_case("mei front", gtsam.Pose3(), _real_params()),
    _mei_case("mei rear body_T_lens", _rear_body_T_lens(), _real_params()),
    _anchor_tag_case(),
    *_seated_cases(),
  )


_CASES = _cases()
_BY_NAME = {case.name: case for case in _CASES}


def _parametrized(test: Callable[..., None]) -> Callable[..., None]:
  marked: Callable[..., None] = pytest.mark.parametrize(
    "case", _CASES, ids=lambda c: c.name
  )(test)
  return marked


def _stepped(
  values: gtsam.Values, key: int, kind: str, axis: int, h: float
) -> gtsam.Values:
  """``values`` with one variable moved by ``h`` along one of its axes."""
  out = gtsam.Values(values)
  if kind == "pose":
    delta = np.zeros(6)
    delta[axis] = h
    out.update(key, values.atPose3(key).retract(delta))
  elif kind == "rot":
    delta = np.zeros(3)
    delta[axis] = h
    out.update(key, values.atRot3(key).retract(delta))
  elif kind == "point":
    delta = np.zeros(3)
    delta[axis] = h
    out.update(key, gtsam.Point3(*(np.asarray(values.atPoint3(key)) + delta)))
  else:
    out.update(key, float(values.atDouble(key)) + h)
  return out


def _finite_difference(
  factor: gtsam.NonlinearFactor, case: _Case, values: gtsam.Values
) -> Float64Array:
  """Central finite difference of the residual in the factor's Jacobian column order."""
  columns = []
  for key, kind, dim in case.blocks:
    for axis in range(dim):
      plus = factor.unwhitenedError(_stepped(values, key, kind, axis, case.step))
      minus = factor.unwhitenedError(_stepped(values, key, kind, axis, -case.step))
      columns.append((plus - minus) / (2.0 * case.step))
  return np.column_stack(columns)


def _pixel_for(case: _Case, p_lens: Float64Array) -> Float64Array:
  return np.asarray(mei_project(p_lens[None, :], case.params)[0], dtype=np.float64)


@_parametrized
def test_zero_residual_at_the_reprojecting_configuration(case: _Case) -> None:
  for label, values, p_lens in case.configurations:
    factor = case.make(_pixel_for(case, p_lens), np.zeros(3))
    err = factor.unwhitenedError(values)
    assert np.allclose(err, 0.0, atol=case.residual_atol), (label, err)


@_parametrized
def test_analytic_jacobians_match_a_central_finite_difference(case: _Case) -> None:
  for label, values, p_lens in case.configurations:
    pixel = _pixel_for(case, p_lens) + case.offset
    factor = case.make(pixel, np.zeros(3))
    a, _ = factor.linearize(values).jacobian()
    assert a.shape == (2, case.columns), f"{label}: unexpected shape {a.shape}"
    fd = _finite_difference(factor, case, values)
    assert np.allclose(a, fd, atol=1e-6), f"{label}: analytic={a}, fd={fd}"


@_parametrized
def test_the_observed_bearing_deactivates_the_opposite_hemisphere(case: _Case) -> None:
  """Residual and Jacobian both go to zero."""
  label, values, p_lens = case.configurations[0]
  pixel = _pixel_for(case, p_lens) + np.array([5.0, -5.0])
  opposite = -p_lens / np.linalg.norm(p_lens)

  off = case.make(pixel, np.zeros(3))
  assert np.linalg.norm(off.unwhitenedError(values)) > 0.0, label

  on = case.make(pixel, opposite)
  assert np.allclose(on.unwhitenedError(values), 0.0), label
  a, _ = on.linearize(values).jacobian()
  assert np.allclose(a, 0.0), label


def test_a_perturbed_pose_moves_the_residual_far_past_the_pixel_noise() -> None:
  """A 5 cm translation answers with a residual far past the 1 px noise model."""
  case = _BY_NAME["mei front"]
  _label, values, p_lens = case.configurations[0]
  factor = case.make(_pixel_for(case, p_lens), np.zeros(3))
  moved = _stepped(values, _POSE, "pose", 3, 0.05)
  assert np.linalg.norm(factor.unwhitenedError(moved)) > 1.0


def test_a_perturbed_finger_angle_moves_the_residual_far_past_the_pixel_noise() -> None:
  case = _BY_NAME["seated finger"]
  _label, values, p_lens = case.configurations[4]  # theta = 0.3
  factor = case.make(_pixel_for(case, p_lens), np.zeros(3))
  moved = _stepped(values, _THETA, "scalar", 0, 0.2)
  assert np.linalg.norm(factor.unwhitenedError(moved)) > 1.0


def test_mei_cheirality_past_90_degrees_preserves_hemisphere() -> None:
  """Project a 95°-half-angle ray + ±1 px noise + unproject ⇒ same hemisphere."""
  params = _real_params()
  rng = np.random.default_rng(seed=42)

  half_angle = math.radians(95.0)
  base_ray = np.array([math.sin(half_angle), 0.0, math.cos(half_angle)])
  assert base_ray[2] < 0.0  # past the horizon

  depth = 1.5
  cam_points = (base_ray * depth)[None, :]
  pixel_true = mei_project(cam_points, params)[0]
  assert np.all(np.isfinite(pixel_true)), "MEI must represent 95° half-angle"

  for _ in range(100):
    noise = rng.normal(scale=1.0, size=2)
    pixel_noisy = pixel_true + noise
    bearing = mei_unproject(pixel_noisy[None, :], params)[0]
    assert bearing[2] < 0.2, (
      f"cheirality flip: base z={base_ray[2]:.3f}, recovered z={bearing[2]:.3f}"
    )
    assert np.linalg.norm(bearing - base_ray) < 0.05


def test_the_behind_camera_basin_the_bearing_guard_closes() -> None:
  """Without a bearing, a ξ>1 factor stays active on a behind-camera landmark."""
  case = _BY_NAME["mei front"]
  behind = np.array([1.0, 0.1, -0.36])  # 110 deg off-axis, short of the 121 deg fold
  pixel = mei_project(behind[None, :], case.params)[0]
  assert np.all(np.isfinite(pixel)), "ξ>1 omni model projects behind-lens points"
  values = _values(pose=gtsam.Pose3(), lm=gtsam.Point3(*behind))
  shifted = pixel + np.array([4.0, -4.0])

  off = case.make(shifted, np.zeros(3))
  assert np.linalg.norm(off.unwhitenedError(values)) > 0.0


def test_a_landmark_past_the_fold_deactivates_the_factor() -> None:
  """Past 121 deg the radius shrinks, so the pixel would repeat one nearer the axis."""
  case = _BY_NAME["mei front"]
  folded = np.array([0.3, 0.1, -0.8])  # 159 deg off-axis
  assert np.isnan(mei_project(folded[None, :], case.params)).all()
  values = _values(pose=gtsam.Pose3(), lm=gtsam.Point3(*folded))
  factor = case.make(np.array([3000.0, 3000.0]), np.zeros(3))
  assert np.allclose(factor.unwhitenedError(values), 0.0)
  a, _ = factor.linearize(values).jacobian()
  assert np.allclose(a, 0.0)


def test_obs_bearing_keeps_same_hemisphere_active() -> None:
  """A peripheral past-90° detection with a same-side bearing stays fully active."""
  case = _BY_NAME["mei front"]
  periph = np.array([1.0, 0.0, -0.1])  # ~95° half-angle, z<0 but in-FOV
  pixel = mei_project(periph[None, :], case.params)[0] + np.array([2.0, -3.0])
  values = _values(pose=gtsam.Pose3(), lm=gtsam.Point3(*periph))
  off = case.make(pixel, np.zeros(3)).unwhitenedError(values)
  on = case.make(pixel, periph / np.linalg.norm(periph)).unwhitenedError(values)
  assert np.linalg.norm(on) > 0.0
  assert np.allclose(off, on)  # bearing on the observed side changes nothing
