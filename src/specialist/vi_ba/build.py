"""The VI-BA factor graph: a pose per keyframe, a point per landmark, a MEI factor
per observation and IMU factors between keyframes."""

from collections.abc import Callable, Sequence
from dataclasses import astuple, dataclass, field

import gtsam
import numpy as np
import numpy.typing as npt
from specialist_camera import MeiParams
from specialist_dataset import schema

from specialist.vi_ba.tag_detection import TagDetection

try:
  from specialist.vi_ba._mei_factor_cpp import add_mei_factors_cpp
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

__all__ = [
  "LensChannel",
  "LensSetup",
  "add_mei_tag_factor",
  "build_vi_graph",
  "huber_pixel_noise",
]


def _as_pixel(v: Float64Array) -> Float64Array:
  return np.asarray(v, dtype=np.float64).reshape(2)


def _as_vec3(v: Float64Array) -> Float64Array:
  return np.asarray(v, dtype=np.float64).reshape(3)


def _as_point3(v: Float64Array) -> gtsam.Point3:
  return gtsam.Point3(*np.asarray(v, dtype=np.float64).reshape(3))


@dataclass(frozen=True)
class LensSetup:
  """Fixed per-lens inputs: ``body_T_lens``, MEI intrinsics and the radius (px) past
  which corner noise grows."""

  body_T_lens: gtsam.Pose3
  params: MeiParams
  calibrated_radius: float


@dataclass
class LensChannel:
  """One camera lens's setup, tag detections, keypoints, tracks and landmarks."""

  lens: schema.LensName
  setup: LensSetup
  tags: list[list[TagDetection]] = field(default_factory=list)
  keypoints: list[Float64Array] = field(default_factory=list)
  tracks: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  points: dict[int, Float64Array] = field(default_factory=dict)
  rejected: dict[int, dict[int, Float64Array]] = field(default_factory=dict)


def huber_pixel_noise(sigma: float) -> gtsam.noiseModel.Base:
  """Robust isotropic pixel noise (Huber, k=1.345) for a reprojection factor."""
  return gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(1.345),
    gtsam.noiseModel.Isotropic.Sigma(2, sigma),
  )


def add_mei_factors(
  graph: gtsam.NonlinearFactorGraph,
  pose_keys: Sequence[int],
  point_keys: Sequence[int],
  pixels: Sequence[Float64Array],
  noise: gtsam.noiseModel.Base,
  lens: LensSetup,
) -> None:
  """One MEI factor per row: ``point_keys`` seen from ``pose_keys`` through ``lens``."""
  add_mei_factors_cpp(
    graph,
    np.asarray(pose_keys, dtype=np.int64),
    np.asarray(point_keys, dtype=np.int64),
    np.asarray(pixels, dtype=np.float64).reshape(-1, 2),
    noise,
    lens.body_T_lens,
    np.array(astuple(lens.params)),
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
  **kwargs: int | None,
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
      **kwargs,
    )
  )


def build_vi_graph(
  body_poses: list[gtsam.Pose3],
  *,
  velocities: list[Float64Array],
  biases: list[gtsam.imuBias.ConstantBias],
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements],
  front: LensChannel,
  sigma: float,
  velocity_prior_sigma: float,
  pose_key_fn: Callable[[int], int],
  velocity_key_fn: Callable[[int], int],
  bias_key_fn: Callable[[int], int],
  front_landmark_key_fn: Callable[[int], int],
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """The front lens's MEI factors plus the IMU factors between consecutive
  keyframes, with priors on ``V(0)`` and ``B(0)``."""
  n = len(body_poses)

  vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, velocity_prior_sigma)
  bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.asarray(
      [_BIAS_PRIOR_SIGMA_ACCEL] * 3 + [_BIAS_PRIOR_SIGMA_GYRO] * 3,
      dtype=np.float64,
    )
  )
  for i in range(n):
    values.insert(pose_key_fn(i), body_poses[i])
    values.insert(velocity_key_fn(i), np.asarray(velocities[i], dtype=np.float64))
    values.insert(bias_key_fn(i), biases[i])
  graph.push_back(
    gtsam.PriorFactorVector(
      velocity_key_fn(0), np.asarray(velocities[0], dtype=np.float64), vel_noise
    )
  )
  graph.push_back(gtsam.PriorFactorConstantBias(bias_key_fn(0), biases[0], bias_noise))
  for i, pim in enumerate(preintegrated):
    graph.push_back(
      gtsam.CombinedImuFactor(
        pose_key_fn(i),
        velocity_key_fn(i),
        pose_key_fn(i + 1),
        velocity_key_fn(i + 1),
        bias_key_fn(i),
        bias_key_fn(i + 1),
        pim,
      )
    )

  pose_keys: list[int] = []
  point_keys: list[int] = []
  pixels: list[Float64Array] = []
  for tid, xyz in front.points.items():
    lm_key = front_landmark_key_fn(tid)
    values.insert(lm_key, _as_point3(xyz))
    for kf, pt in front.tracks[tid].items():
      pose_keys.append(pose_key_fn(kf))
      point_keys.append(lm_key)
      pixels.append(pt)
  add_mei_factors(
    graph, pose_keys, point_keys, pixels, huber_pixel_noise(sigma), front.setup
  )
  return graph, values
