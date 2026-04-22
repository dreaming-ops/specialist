"""The MEI projection round-trips."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from specialist.loader.mei_model import (
  MeiParams,
  mei_project,
  mei_project_with_jacobian,
  mei_unproject,
)

# Plausible Insta360 intrinsics in 1920x1920 per-lens coords.
_X3 = MeiParams(
  fx=540.0, fy=540.0, cx=960.0, cy=960.0,
  k1=0.39, k2=1.24, k3=-3.86, p1=-1.0e-5, p2=-2.16e-3, xi=1.948,
)  # fmt: skip
_PINHOLE = MeiParams(
  fx=400.0, fy=400.0, cx=320.0, cy=240.0,
  k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0, xi=0.0,
)  # fmt: skip


def test_mei_round_trip_pinhole_centered() -> None:
  params = _PINHOLE
  pixels = np.array([[320.0, 240.0]])
  bearing = mei_unproject(pixels, params)
  assert bearing.shape == (1, 3)
  assert np.allclose(bearing, [[0.0, 0.0, 1.0]], atol=1e-9)


def test_mei_round_trip_pinhole_offcenter() -> None:
  params = _PINHOLE
  rng = np.random.default_rng(0)
  pixels = rng.uniform([100.0, 80.0], [540.0, 400.0], size=(50, 2))
  bearing = mei_unproject(pixels, params)
  reprojected = mei_project(bearing, params)
  assert np.allclose(reprojected, pixels, atol=1e-6)


def _forward_rays(rng: np.random.Generator, n: int, half_angle: float) -> np.ndarray:
  """Sample unit rays inside a forward cone of given half-angle."""
  cos_min = np.cos(half_angle)
  cos_t = rng.uniform(cos_min, 1.0, size=n)
  sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
  phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
  return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)


def test_mei_round_trip_with_distortion() -> None:
  """ray -> pixel -> ray within tight tolerance, staying inside the camera FOV."""
  params = _X3
  rng = np.random.default_rng(42)
  rays = _forward_rays(rng, n=80, half_angle=np.deg2rad(85.0))
  pixels = mei_project(rays, params)
  assert np.isfinite(pixels).all()
  recovered = mei_unproject(pixels, params)
  assert np.isfinite(recovered).all()
  assert np.max(np.abs(recovered - rays)) < 1e-6


@pytest.mark.parametrize("xi", [-0.5, 0.0, 0.8, 1.948])
def test_mei_project_is_nan_exactly_past_the_fold(xi: float) -> None:
  """NaN exactly where ``cos(angle) <= -xi``, or ``<= -1/xi`` once ``xi > 1``."""
  params = replace(_X3, xi=xi)
  fold = np.arccos(-(xi if xi <= 1.0 else 1.0 / xi))
  angles = np.array([fold - 1e-3, fold + 1e-3, np.pi - 1e-3])
  rays = np.stack([np.sin(angles), np.zeros(3), np.cos(angles)], axis=1)
  pixels = mei_project(rays, params)
  assert np.isfinite(pixels[0]).all()
  assert np.isnan(pixels[1:]).all()


def test_mei_rejects_bad_shape() -> None:
  with pytest.raises(ValueError):
    mei_project(np.zeros((3,)), _PINHOLE)
  with pytest.raises(ValueError):
    mei_unproject(np.zeros((3,)), _PINHOLE)


def test_mei_project_with_jacobian_pixel_matches_batch_project() -> None:
  """The single-point projector matches ``mei_project`` for X3-like params."""
  params = _X3
  for p in [
    np.array([0.05, 0.05, 1.0]),
    np.array([0.5, 0.2, 1.5]),
    np.array([1.0, -0.3, 0.8]),
    np.array([1.5, 0.0, 0.2]),
  ]:
    pixel_a, _ = mei_project_with_jacobian(p, params)
    pixel_b = mei_project(p[None, :], params)[0]
    assert np.allclose(pixel_a, pixel_b, atol=1e-12)


def test_mei_project_jacobian_matches_central_finite_differences() -> None:
  """The match holds to 1e-4 relative, even past the 90° half-angle."""
  params = _X3
  test_points = [
    np.array([0.01, 0.02, 1.5]),  # near-axial
    np.array([0.4, 0.3, 1.0]),
    np.array([1.0, -0.2, 0.6]),
    np.array([1.6, 0.0, 0.3]),
    np.array([1.0, 0.0, -0.1]),  # past 90° half-angle, super-fisheye
  ]
  eps = 1e-6
  for p in test_points:
    _, jac = mei_project_with_jacobian(p, params)
    fd = np.zeros((2, 3), dtype=np.float64)
    for i in range(3):
      d = np.zeros(3)
      d[i] = eps
      pix_p = mei_project((p + d)[None, :], params)[0]
      pix_m = mei_project((p - d)[None, :], params)[0]
      fd[:, i] = (pix_p - pix_m) / (2.0 * eps)
    rel = np.linalg.norm(jac - fd) / max(np.linalg.norm(fd), 1.0)
    assert rel < 1e-4, f"point {p}: |jac - fd|/|fd| = {rel:.2e}"


def test_mei_project_with_jacobian_returns_nan_behind_mirror() -> None:
  # At ξ = 0 (pinhole) a point with z < 0 fails the cheirality check.
  params = _PINHOLE  # xi=0
  pixel, jac = mei_project_with_jacobian(np.array([0.1, 0.0, -1.0]), params)
  assert np.all(np.isnan(pixel))
  assert np.all(np.isnan(jac))


def test_mei_from_camera_matrix_validates_shapes() -> None:
  with pytest.raises(ValueError):
    MeiParams.from_camera_matrix(np.zeros((2, 2)), np.zeros((6,)))
  with pytest.raises(ValueError):
    MeiParams.from_camera_matrix(np.eye(3), np.zeros((5,)))
