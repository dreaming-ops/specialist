"""Dual-lens VI-BA graph: MEI and IMU factors, gauge pinned at the first pose."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import astuple, dataclass

import gtsam
import numpy as np
import numpy.typing as npt
from gtsam.symbol_shorthand import B, L, V, X

from specialist.loader.mei_model import MeiParams

try:
  # Native C++ MEI reprojection factor on (body pose, world landmark).
  from specialist.vi_ba._mei_factor_cpp import make_mei_factor_cpp
except ImportError as _exc:  # pragma: no cover - extension is built by uv sync
  raise ImportError(
    "The native MEI factor extension (specialist.vi_ba._mei_factor_cpp) is not "
    "built. It is compiled automatically by `uv sync`; to (re)build it manually "
    "run `uv run build-mei-factor`."
  ) from _exc

Float64Array = npt.NDArray[np.float64]

_REAR_LANDMARK_OFFSET = 1_000_000

__all__ = [
  "LensSetup",
  "add_mei_factor",
  "build_vi_graph",
  "front_landmark_key",
  "huber_pixel_noise",
  "rear_landmark_key",
]


def _as_pixel(v: Float64Array) -> Float64Array:
  return np.asarray(v, dtype=np.float64).reshape(2)


def _as_vec3(v: Float64Array) -> Float64Array:
  return np.asarray(v, dtype=np.float64).reshape(3)


def _gauge_noise() -> gtsam.noiseModel.Base:
  """Tight gauge prior (σ=1e-4 on all six pose DoF) that pins the frame."""
  return gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-4, dtype=np.float64))


@dataclass(frozen=True)
class LensSetup:
  """Fixed per-lens inputs: body_T_lens + MEI intrinsics."""

  body_T_lens: gtsam.Pose3
  params: MeiParams


def front_landmark_key(track_id: int) -> int:
  return int(L(track_id))


def rear_landmark_key(track_id: int) -> int:
  return int(L(_REAR_LANDMARK_OFFSET + track_id))


def huber_pixel_noise(sigma: float) -> gtsam.noiseModel.Base:
  """Robust isotropic pixel noise (Huber, k=1.345) for a reprojection factor."""
  return gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(1.345),
    gtsam.noiseModel.Isotropic.Sigma(2, sigma),
  )


def add_mei_factor(
  graph: gtsam.NonlinearFactorGraph,
  pose_key: int,
  point_key: int,
  pixel: Float64Array,
  noise: gtsam.noiseModel.Base,
  lens: LensSetup,
  obs_bearing: Float64Array | None = None,
) -> None:
  """Push one MEI reprojection factor; ``obs_bearing`` arms the cheirality guard."""
  bearing = np.zeros(3) if obs_bearing is None else _as_vec3(obs_bearing)
  graph.push_back(
    make_mei_factor_cpp(
      pose_key,
      point_key,
      _as_pixel(pixel),
      noise,
      lens.body_T_lens,
      np.array(astuple(lens.params)),
      bearing,
    )
  )


def _add_projection_factors(
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
  tracks: Mapping[int, Mapping[int, Float64Array]],
  points: Mapping[int, Float64Array],
  lens: LensSetup,
  landmark_key_fn: Callable[[int], int],
  pixel_noise: gtsam.noiseModel.Base,
  pose_key_fn: Callable[[int], int],
) -> None:
  for tid, xyz in points.items():
    lm_key = landmark_key_fn(tid)
    if not values.exists(lm_key):
      values.insert(lm_key, gtsam.Point3(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    for kf, pt in tracks[tid].items():
      add_mei_factor(graph, pose_key_fn(kf), lm_key, pt, pixel_noise, lens)


def _add_both_lenses(
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
  front: LensSetup,
  front_tracks: Mapping[int, Mapping[int, Float64Array]],
  front_points: Mapping[int, Float64Array],
  front_landmark_key_fn: Callable[[int], int],
  rear: LensSetup | None,
  rear_tracks: Mapping[int, Mapping[int, Float64Array]] | None,
  rear_points: Mapping[int, Float64Array] | None,
  rear_landmark_key_fn: Callable[[int], int],
  pixel_noise: gtsam.noiseModel.Base,
  pose_key_fn: Callable[[int], int],
) -> None:
  """Add front projection factors, and rear ones when every rear input is present."""
  _add_projection_factors(
    graph,
    values,
    front_tracks,
    front_points,
    front,
    front_landmark_key_fn,
    pixel_noise,
    pose_key_fn,
  )
  if rear is not None and rear_tracks is not None and rear_points is not None:
    _add_projection_factors(
      graph,
      values,
      rear_tracks,
      rear_points,
      rear,
      rear_landmark_key_fn,
      pixel_noise,
      pose_key_fn,
    )


def build_vi_graph(
  body_poses: list[gtsam.Pose3],
  *,
  velocities: list[Float64Array] | None = None,
  biases: list[gtsam.imuBias.ConstantBias] | None = None,
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements] | None = None,
  front: LensSetup,
  front_tracks: Mapping[int, Mapping[int, Float64Array]],
  front_points: Mapping[int, Float64Array],
  rear: LensSetup | None,
  rear_tracks: Mapping[int, Mapping[int, Float64Array]] | None,
  rear_points: Mapping[int, Float64Array] | None,
  sigma: float,
  velocity_prior_sigma: float = 0.1,
  bias_prior_accel_sigma: float = 1.0e-2,
  bias_prior_gyro_sigma: float = 1.0e-3,
  anchor_last_pose: bool = False,
  pose_key_fn: Callable[[int], int] | None = None,
  velocity_key_fn: Callable[[int], int] | None = None,
  bias_key_fn: Callable[[int], int] | None = None,
  front_landmark_key_fn: Callable[[int], int] | None = None,
  rear_landmark_key_fn: Callable[[int], int] | None = None,
  graph: gtsam.NonlinearFactorGraph | None = None,
  values: gtsam.Values | None = None,
  weak_pose_sigmas: tuple[float, ...] | None = None,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """Dual-lens VI graph; ``preintegrated=None`` drops the IMU for weak pose priors."""
  n = len(body_poses)

  if graph is None:
    graph = gtsam.NonlinearFactorGraph()
  if values is None:
    values = gtsam.Values()
  pose_key = pose_key_fn or (lambda i: int(X(i)))
  front_lm = front_landmark_key_fn or front_landmark_key
  rear_lm = rear_landmark_key_fn or rear_landmark_key
  pixel_noise = huber_pixel_noise(sigma)
  gauge = _gauge_noise()

  if preintegrated is None:
    # Visual-only: priors on X(0) and optionally X(N-1); weak pose priors on the rest.
    weak = gtsam.noiseModel.Diagonal.Sigmas(
      np.asarray(weak_pose_sigmas, dtype=np.float64)
    )
    for i, pose in enumerate(body_poses):
      values.insert(pose_key(i), pose)
    graph.push_back(gtsam.PriorFactorPose3(pose_key(0), body_poses[0], gauge))
    if anchor_last_pose and n > 1:
      graph.push_back(gtsam.PriorFactorPose3(pose_key(n - 1), body_poses[-1], gauge))
    for i in range(1, n):
      if anchor_last_pose and i == n - 1:
        continue
      graph.push_back(gtsam.PriorFactorPose3(pose_key(i), body_poses[i], weak))
  else:
    # Visual-inertial: CombinedImuFactor between keyframes, priors on X0, V0 and B0.
    assert velocities is not None and biases is not None
    vel_key = velocity_key_fn or (lambda i: int(V(i)))
    bias_key = bias_key_fn or (lambda i: int(B(i)))
    vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, velocity_prior_sigma)
    bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
      np.asarray(
        [bias_prior_accel_sigma] * 3 + [bias_prior_gyro_sigma] * 3,
        dtype=np.float64,
      )
    )
    for i in range(n):
      values.insert(pose_key(i), body_poses[i])
      values.insert(vel_key(i), np.asarray(velocities[i], dtype=np.float64))
      values.insert(bias_key(i), biases[i])
    graph.push_back(gtsam.PriorFactorPose3(pose_key(0), body_poses[0], gauge))
    if anchor_last_pose:
      graph.push_back(gtsam.PriorFactorPose3(pose_key(n - 1), body_poses[-1], gauge))
    graph.push_back(
      gtsam.PriorFactorVector(
        vel_key(0), np.asarray(velocities[0], dtype=np.float64), vel_noise
      )
    )
    graph.push_back(gtsam.PriorFactorConstantBias(bias_key(0), biases[0], bias_noise))
    for i, pim in enumerate(preintegrated):
      graph.push_back(
        gtsam.CombinedImuFactor(
          pose_key(i),
          vel_key(i),
          pose_key(i + 1),
          vel_key(i + 1),
          bias_key(i),
          bias_key(i + 1),
          pim,
        )
      )

  _add_both_lenses(
    graph,
    values,
    front,
    front_tracks,
    front_points,
    front_lm,
    rear,
    rear_tracks,
    rear_points,
    rear_lm,
    pixel_noise,
    pose_key,
  )
  return graph, values
