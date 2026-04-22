"""MEI projection factor on body pose and landmark; a cheirality failure zeros it."""

from __future__ import annotations

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import MeiParams, mei_project_with_jacobian

__all__ = ["make_mei_factor"]


def _skew3(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
  return np.array(
    [
      [0.0, -v[2], v[1]],
      [v[2], 0.0, -v[0]],
      [-v[1], v[0], 0.0],
    ],
    dtype=np.float64,
  )


def make_mei_factor(
  pose_key: int,
  landmark_key: int,
  pixel: npt.NDArray[np.float64],
  noise: gtsam.noiseModel.Base,
  body_T_lens: gtsam.Pose3,
  params: MeiParams,
) -> gtsam.CustomFactor:
  """Binary MEI projection factor on ``(body_pose, world_landmark)`` for one lens."""
  pixel_arr = np.asarray(pixel, dtype=np.float64).reshape(2)
  # The body → lens pieces stay constant for the factor's lifetime.
  r_bl = body_T_lens.rotation().matrix()
  r_lb = r_bl.T
  t_bl = np.asarray(body_T_lens.translation(), dtype=np.float64)

  def error_func(
    this: gtsam.CustomFactor,
    values: gtsam.Values,
    H: list[np.ndarray] | None,
  ) -> npt.NDArray[np.float64]:
    keys = this.keys()
    body_pose = values.atPose3(keys[0])
    landmark = np.asarray(values.atPoint3(keys[1]), dtype=np.float64)

    # Compose world → lens in raw numpy, without gtsam pybind round-trips.
    r_wb = body_pose.rotation().matrix()
    t_wb = np.asarray(body_pose.translation(), dtype=np.float64)
    r_bw = r_wb.T
    p_b = r_bw @ (landmark - t_wb)
    p_lens = r_lb @ (p_b - t_bl)

    predicted, j_mei = mei_project_with_jacobian(p_lens, params)

    if not np.all(np.isfinite(predicted)):
      # Cheirality — zero out so LM sees an inactive factor this iteration.
      if H is not None and len(H) >= 2:
        H[0] = np.zeros((2, 6), dtype=np.float64)
        H[1] = np.zeros((2, 3), dtype=np.float64)
      return np.zeros(2, dtype=np.float64)

    residual = predicted - pixel_arr

    if H is not None and len(H) >= 2:
      j_mei_r_lb = j_mei @ r_lb  # 2×3, reused on both blocks

      hp = np.empty((2, 6), dtype=np.float64)
      hp[:, :3] = j_mei_r_lb @ _skew3(p_b)  # rotation columns
      hp[:, 3:] = -j_mei_r_lb  # translation columns
      hl = j_mei_r_lb @ r_bw  # = j_mei · R_lw

      H[0] = hp
      H[1] = hl

    return np.asarray(residual, dtype=np.float64)

  return gtsam.CustomFactor(noise, [pose_key, landmark_key], error_func)
