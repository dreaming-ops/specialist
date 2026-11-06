"""MEI unified omnidirectional projection with Brown-Conrady distortion."""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]

__all__ = ["LENSES", "LensName", "MeiParams", "mei_project", "mei_unproject", "rescale"]

LensName = Literal["front", "rear"]
LENSES: tuple[LensName, LensName] = ("front", "rear")


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


def unit_rows(v: Float64Array) -> Float64Array:
  """Normalize each row of ``v`` to unit length, leaving zero rows unchanged."""
  norms = np.linalg.norm(v, axis=1, keepdims=True)
  norms = np.where(norms <= 1e-12, 1.0, norms)
  return v / norms


def require_2d(arr: Float64Array, cols: int, name: str) -> None:
  """Raise ValueError unless ``arr`` has shape ``(N, cols)``."""
  if arr.ndim != 2 or arr.shape[1] != cols:
    raise ValueError(f"{name} must be (N, {cols}), got {arr.shape}")


def rescale(
  raw: MeiParams, frame_width: float, decoded_width: int, lens: LensName
) -> MeiParams:
  """Rescale ``raw`` from the two-lens frame to one lens at ``decoded_width``."""
  scale = 2.0 * decoded_width / frame_width
  left_edge = frame_width / 2.0 if lens == "rear" else 0.0
  return replace(
    raw,
    fx=raw.fx * scale,
    fy=raw.fy * scale,
    cx=(raw.cx - left_edge) * scale,
    cy=raw.cy * scale,
  )


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

  x, y = x_d.copy(), y_d.copy()
  rows = np.arange(len(pixels))
  for _ in range(20):
    if rows.size == 0:
      break
    xs, ys = x[rows], y[rows]
    r2 = xs * xs + ys * ys
    radial = 1.0 + params.k1 * r2 + params.k2 * r2 * r2 + params.k3 * r2**3
    dx_t = 2.0 * params.p1 * xs * ys + params.p2 * (r2 + 2.0 * xs * xs)
    dy_t = params.p1 * (r2 + 2.0 * ys * ys) + 2.0 * params.p2 * xs * ys
    safe_radial = np.where(radial == 0.0, 1.0, radial)
    x_new = (x_d[rows] - dx_t) / safe_radial
    y_new = (y_d[rows] - dy_t) / safe_radial
    x[rows] = x_new
    y[rows] = y_new
    rows = rows[np.maximum(np.abs(x_new - xs), np.abs(y_new - ys)) >= 1e-12]

  r2 = x * x + y * y
  # factor is the larger root of factor**2 (r2 + 1) - 2 factor xi + xi**2 - 1 = 0.
  disc = 1.0 + (1.0 - params.xi * params.xi) * r2
  disc = np.where(disc < 0.0, np.nan, disc)
  factor = (params.xi + np.sqrt(disc)) / (r2 + 1.0)
  bearing = np.stack([factor * x, factor * y, factor - params.xi], axis=1)
  return unit_rows(bearing)
