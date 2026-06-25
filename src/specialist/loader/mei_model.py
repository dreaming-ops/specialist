"""Pure-numpy MEI unified projection; distortion is ``[k1, k2, k3, p1, p2, xi]``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MeiParams:
  """MEI intrinsics: pinhole + radial/tangential distortion + ``xi``."""

  fx: float
  fy: float
  cx: float
  cy: float
  k1: float
  k2: float
  k3: float
  p1: float
  p2: float
  xi: float

  @classmethod
  def from_camera_matrix(
    cls, camera_matrix: Float64Array, distortion_coeffs: Float64Array
  ) -> MeiParams:
    """Build from the ``LensIntrinsics`` arrays parsed out of a ``.insv``."""
    if camera_matrix.shape != (3, 3):
      raise ValueError(f"camera_matrix must be 3x3, got {camera_matrix.shape}")
    if distortion_coeffs.shape != (6,):
      raise ValueError(
        f"distortion_coeffs must be (6,) [k1,k2,k3,p1,p2,xi], "
        f"got {distortion_coeffs.shape}"
      )
    return cls(
      fx=float(camera_matrix[0, 0]),
      fy=float(camera_matrix[1, 1]),
      cx=float(camera_matrix[0, 2]),
      cy=float(camera_matrix[1, 2]),
      k1=float(distortion_coeffs[0]),
      k2=float(distortion_coeffs[1]),
      k3=float(distortion_coeffs[2]),
      p1=float(distortion_coeffs[3]),
      p2=float(distortion_coeffs[4]),
      xi=float(distortion_coeffs[5]),
    )


def unit_rows(v: Float64Array) -> Float64Array:
  """Normalize each row of ``v`` to unit length, leaving zero rows unchanged."""
  norms = np.linalg.norm(v, axis=1, keepdims=True)
  norms = np.where(norms == 0.0, 1.0, norms)
  return v / norms


def require_2d(arr: Float64Array, cols: int, name: str) -> None:
  """Validate that ``arr`` is ``(N, cols)``; raise ValueError otherwise."""
  if arr.ndim != 2 or arr.shape[1] != cols:
    raise ValueError(f"{name} must be (N, {cols}), got {arr.shape}")


def mei_project(points_cam: Float64Array, params: MeiParams) -> Float64Array:
  """``(N, 2)`` pixels of ``(N, 3)`` lens-frame points, NaN past the fold."""
  require_2d(points_cam, 3, "points_cam")
  sphere = unit_rows(points_cam)

  denom = sphere[:, 2] + params.xi
  # Pixels need z > -xi, and z > -1/xi for xi > 1, where the image radius folds back.
  fold = params.xi if params.xi <= 1.0 else 1.0 / params.xi
  valid = sphere[:, 2] > -fold
  x = np.where(valid, sphere[:, 0] / denom, np.nan)
  y = np.where(valid, sphere[:, 1] / denom, np.nan)

  r2 = x * x + y * y
  radial = 1.0 + params.k1 * r2 + params.k2 * r2 * r2 + params.k3 * r2**3
  x_d = x * radial + 2.0 * params.p1 * x * y + params.p2 * (r2 + 2.0 * x * x)
  y_d = y * radial + params.p1 * (r2 + 2.0 * y * y) + 2.0 * params.p2 * x * y

  u = params.fx * x_d + params.cx
  v = params.fy * y_d + params.cy
  return np.stack([u, v], axis=1)


def mei_unproject(pixels: Float64Array, params: MeiParams) -> Float64Array:
  """``(N, 3)`` unit bearings of ``(N, 2)`` pixels, NaN outside the model."""
  require_2d(pixels, 2, "pixels")

  x_d = (pixels[:, 0] - params.cx) / params.fx
  y_d = (pixels[:, 1] - params.cy) / params.fy

  x = x_d.copy()
  y = y_d.copy()
  for _ in range(20):
    r2 = x * x + y * y
    radial = 1.0 + params.k1 * r2 + params.k2 * r2 * r2 + params.k3 * r2**3
    dx_t = 2.0 * params.p1 * x * y + params.p2 * (r2 + 2.0 * x * x)
    dy_t = params.p1 * (r2 + 2.0 * y * y) + 2.0 * params.p2 * x * y
    x_new = (x_d - dx_t) / np.where(radial == 0.0, 1.0, radial)
    y_new = (y_d - dy_t) / np.where(radial == 0.0, 1.0, radial)
    if np.max(np.abs(x_new - x)) < 1e-12 and np.max(np.abs(y_new - y)) < 1e-12:
      x, y = x_new, y_new
      break
    x, y = x_new, y_new

  r2 = x * x + y * y
  # Invert the sphere shift in closed form, valid for xi >= 0.
  disc = 1.0 + (1.0 - params.xi * params.xi) * r2
  disc = np.where(disc < 0.0, np.nan, disc)
  factor = (params.xi + np.sqrt(disc)) / (r2 + 1.0)
  bearing = np.stack([factor * x, factor * y, factor - params.xi], axis=1)
  return unit_rows(bearing)
