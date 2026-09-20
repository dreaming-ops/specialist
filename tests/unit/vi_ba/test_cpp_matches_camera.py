"""``mei_factor.cpp`` and ``mei_project`` agree across the dataset's ``xi`` range."""

from __future__ import annotations

from dataclasses import astuple

import gtsam
import numpy as np
import numpy.typing as npt
import pytest
from specialist_camera import MeiParams, mei_project

from specialist.vi_ba import _mei_factor_cpp
from tests.mei_fixtures import distorted_mei_params

Float64Array = npt.NDArray[np.float64]

_POSE, _LM = 1, 2

_LENS_POINTS = (
  np.array([0.01, 0.01, 2.0]),
  np.array([0.5, 0.2, 1.5]),
  np.array([1.0, 0.3, 0.9]),
  np.array([1.6, 0.0, 0.3]),
  np.array([-0.7, 0.4, 1.1]),
)

_BODY_T_LENSES = (
  gtsam.Pose3(),
  gtsam.Pose3(gtsam.Rot3.Rz(np.pi), gtsam.Point3(0.0, 0.0, -0.03)),
)


def _cpp_projection(
  params: MeiParams, body_T_lens: gtsam.Pose3, lens_xyz: Float64Array
) -> Float64Array:
  """The native factor's error against a zero measurement, which is its projection."""
  landmark = body_T_lens.transformFrom(gtsam.Point3(*lens_xyz))
  factor = _mei_factor_cpp.make_mei_factor_cpp(
    _POSE,
    _LM,
    np.zeros(2),
    gtsam.noiseModel.Isotropic.Sigma(2, 1.0),
    body_T_lens,
    np.array(astuple(params)),
  )
  values = gtsam.Values()
  values.insert(_POSE, gtsam.Pose3())
  values.insert(_LM, landmark)
  return np.asarray(factor.unwhitenedError(values), dtype=np.float64)


@pytest.mark.parametrize("xi", [0.0, 0.8, 1.6, 1.948], ids=lambda x: f"xi={x}")
@pytest.mark.parametrize("body_T_lens", _BODY_T_LENSES, ids=["identity", "rear"])
@pytest.mark.parametrize("lens_xyz", _LENS_POINTS, ids=lambda p: str(p.tolist()))
def test_native_factor_projects_where_mei_project_does(
  xi: float, body_T_lens: gtsam.Pose3, lens_xyz: Float64Array
) -> None:
  params = distorted_mei_params(xi)
  expected = mei_project(lens_xyz[None, :], params)[0]
  got = _cpp_projection(params, body_T_lens, lens_xyz)
  np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-9)
