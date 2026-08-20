"""Dual-lens VI-BA graph: MEI and IMU factors, gauge pinned at the first pose."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import astuple, dataclass, field

import gtsam
import numpy as np
import numpy.typing as npt
from gtsam.symbol_shorthand import B, L, V, X

from specialist.loader.mei_model import MeiParams
from specialist.vi_ba.features import Rect
from specialist.vi_ba.tag_detection import TagDetection

try:
  from specialist.vi_ba._mei_factor_cpp import make_mei_factor_cpp
except ImportError as _exc:  # pragma: no cover - extension is built by uv sync
  raise ImportError(
    "The native MEI factor extension (specialist.vi_ba._mei_factor_cpp) is not "
    "built. It is compiled automatically by `uv sync`; to (re)build it manually "
    "run `uv run build-mei-factor`."
  ) from _exc

Float64Array = npt.NDArray[np.float64]

# Prior on the first bias state, in gtsam's (accel, gyro) tangent order.
_BIAS_PRIOR_SIGMA_ACCEL = 1.0e-2
_BIAS_PRIOR_SIGMA_GYRO = 1.0e-3

_REAR_LANDMARK_OFFSET = 1_000_000

__all__ = [
  "add_mei_factor",
  "LensChannel",
  "LensSetup",
  "add_mei_tag_factor",
  "build_vi_graph",
  "front_landmark_key",
  "huber_pixel_noise",
  "rear_landmark_key",
]


def _as_pixel(v: Float64Array) -> Float64Array:
  return np.asarray(v, dtype=np.float64).reshape(2)


def _as_vec3(v: Float64Array) -> Float64Array:
  return np.asarray(v, dtype=np.float64).reshape(3)


def _as_point3(v: Float64Array) -> gtsam.Point3:
  return gtsam.Point3(*np.asarray(v, dtype=np.float64).reshape(3))


def _gauge_noise() -> gtsam.noiseModel.Base:
  """Tight gauge prior (σ=1e-4 on all six pose DoF) that pins the frame."""
  return gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-4, dtype=np.float64))


@dataclass(frozen=True)
class LensSetup:
  """Fixed per-lens inputs: body_T_lens + MEI intrinsics."""

  body_T_lens: gtsam.Pose3
  params: MeiParams


@dataclass
class LensChannel:
  """One camera lens's setup, tag detections, keypoints, tracks and landmarks."""

  lens: str  # "front" or "rear", the key anchor observations come back under
  setup: LensSetup
  tags: list[list[TagDetection]] = field(default_factory=list)
  keypoints: list[Float64Array] = field(default_factory=list)
  tracks: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  points: dict[int, Float64Array] = field(default_factory=dict)
  rejected: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  # Rectangles masked from this lens at every keyframe; the front lens has none.
  static_mask_rects: list[Rect] = field(default_factory=list)


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
  graph.push_back(
    make_mei_factor_cpp(
      pose_key,
      point_key,
      _as_pixel(pixel),
      noise,
      lens.body_T_lens,
      np.array(astuple(lens.params)),
      np.zeros(3) if obs_bearing is None else _as_vec3(obs_bearing),
    )
  )


def add_mei_tag_factor(
  graph: gtsam.NonlinearFactorGraph,
  make: Callable[..., gtsam.NonlinearFactor],
  keys: Sequence[int],
  pixel: Float64Array,
  noise: gtsam.noiseModel.Base,
  lens: LensSetup,
  corner: Float64Array,
  obs_bearing: Float64Array,
  base_T_joint: gtsam.Pose3 | None = None,
) -> None:
  """Push one native ``make`` factor on ``keys`` for a tag ``corner`` through ``lens``."""
  joint = () if base_T_joint is None else (base_T_joint,)
  graph.push_back(
    make(
      *keys,
      _as_pixel(pixel),
      noise,
      lens.body_T_lens,
      *joint,
      _as_point3(corner),
      np.array(astuple(lens.params)),
      _as_vec3(obs_bearing),
    )
  )


def _add_projection_factors(
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
  channel: LensChannel,
  landmark_key_fn: Callable[[int], int],
  pixel_noise: gtsam.noiseModel.Base,
  pose_key_fn: Callable[[int], int],
) -> None:
  for tid, xyz in channel.points.items():
    lm_key = landmark_key_fn(tid)
    if not values.exists(lm_key):
      values.insert(lm_key, _as_point3(xyz))
    for kf, pt in channel.tracks[tid].items():
      add_mei_factor(graph, pose_key_fn(kf), lm_key, pt, pixel_noise, channel.setup)


def build_vi_graph(
  body_poses: list[gtsam.Pose3],
  *,
  velocities: list[Float64Array] | None = None,
  biases: list[gtsam.imuBias.ConstantBias] | None = None,
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements] | None = None,
  front: LensChannel,
  rear: LensChannel,
  sigma: float,
  velocity_prior_sigma: float = 0.1,
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
    weak_end = n - 1 if anchor_last_pose else n
    for i in range(1, weak_end):
      graph.push_back(gtsam.PriorFactorPose3(pose_key(i), body_poses[i], weak))
  else:
    # Visual-inertial: CombinedImuFactor between keyframes, priors on X0, V0 and B0.
    assert velocities is not None and biases is not None
    vel_key = velocity_key_fn or (lambda i: int(V(i)))
    bias_key = bias_key_fn or (lambda i: int(B(i)))
    vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, velocity_prior_sigma)
    bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
      np.asarray(
        [_BIAS_PRIOR_SIGMA_ACCEL] * 3 + [_BIAS_PRIOR_SIGMA_GYRO] * 3,
        dtype=np.float64,
      )
    )
    for i in range(n):
      values.insert(pose_key(i), body_poses[i])
      values.insert(vel_key(i), np.asarray(velocities[i], dtype=np.float64))
      values.insert(bias_key(i), biases[i])
    graph.push_back(gtsam.PriorFactorPose3(pose_key(0), body_poses[0], gauge))
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

  for channel, landmark_key_fn in ((front, front_lm), (rear, rear_lm)):
    _add_projection_factors(
      graph, values, channel, landmark_key_fn, pixel_noise, pose_key
    )
  return graph, values
