"""``vi_ba.init``: pose filling, triangulation and visual-inertial alignment."""

from __future__ import annotations

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.vi_ba.imu import (
  ImuSamples,
  integrate_samples,
  make_preintegration_params,
)
from specialist.vi_ba.init import (
  Alignment,
  apply_alignment_to_poses,
  gravity_alignment_rotation,
  visual_inertial_align,
)

Float64Array = npt.NDArray[np.float64]


def _pim_with_motion(
  dt_total: float,
  gravity_world: np.ndarray,
  motion_accel_world: np.ndarray,
) -> gtsam.PreintegratedCombinedMeasurements:
  """Preintegrate a level body reading ``motion_accel_world - gravity``."""
  n = max(2, int(dt_total * 1000))
  t = np.linspace(0.0, dt_total, n, endpoint=False, dtype=np.float64)
  reading = motion_accel_world - gravity_world
  samples = ImuSamples(
    t=t,
    gyro=np.zeros((n, 3), dtype=np.float64),
    accel=np.tile(reading, (n, 1)),
  )
  params = make_preintegration_params(
    gyro_noise_sigma=1.0e-4,
    accel_noise_sigma=1.0e-4,
    gyro_bias_rw_sigma=1.0e-5,
    accel_bias_rw_sigma=1.0e-5,
    gravity_world=(
      float(gravity_world[0]),
      float(gravity_world[1]),
      float(gravity_world[2]),
    ),
  )
  bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
  return integrate_samples(params, samples, bias)


_ACCEL_PHASES = [
  np.array([0.4, 0.3, 0.2]),
  np.array([-0.2, -0.1, 0.0]),
  np.array([0.1, -0.3, 0.2]),
  np.array([0.3, 0.1, -0.2]),
  np.array([-0.3, 0.4, 0.1]),
]


def _integrate_metric_trajectory(
  accel_phases: list[np.ndarray], dts: list[float]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
  """Per-keyframe metric positions and velocities of constant-accel phases from rest."""
  positions, velocities = [np.zeros(3)], [np.zeros(3)]
  for a, dt in zip(accel_phases, dts, strict=True):
    positions.append(positions[-1] + velocities[-1] * dt + 0.5 * a * dt * dt)
    velocities.append(velocities[-1] + a * dt)
  return positions, velocities


def test_gravity_alignment_rotation_maps_gravity_to_minus_z() -> None:
  g = np.array([0.3, 0.2, -9.78])
  r = gravity_alignment_rotation(g)
  rotated = r.matrix() @ g
  assert rotated[0] ** 2 + rotated[1] ** 2 < 1e-10
  assert rotated[2] < 0.0
  assert abs(abs(rotated[2]) - np.linalg.norm(g)) < 1e-10


def test_apply_alignment_rescales_and_rotates() -> None:
  # Two poses along +x at 0.1 m (visual scale), gravity tipped in visual frame.
  body_poses = [
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 0.0)),
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.1, 0.0, 0.0)),
  ]
  align = Alignment(
    scale=3.0,
    gravity_world=np.array([0.0, 9.81, 0.0]),  # gravity points along +y in visual frame
    velocities=[np.zeros(3), np.zeros(3)],
  )
  aligned, rot = apply_alignment_to_poses(body_poses, align)
  # Scale 3 makes the baseline 0.3 m, and gravity rotates the original +y to world -z.
  np.testing.assert_allclose(aligned[1].translation(), [0.3, 0.0, 0.0], atol=1e-10)
  np.testing.assert_allclose(
    rot.matrix() @ [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], atol=1e-10
  )


def _lens_arrays(poses: list[gtsam.Pose3]) -> tuple[Float64Array, Float64Array]:
  """Each pose's world_R_lens and lens center, as ``triangulate_lens`` builds them."""
  rotations = np.stack([pose.rotation().matrix() for pose in poses])
  centers = np.stack(
    [np.asarray(pose.translation(), dtype=np.float64) for pose in poses]
  )
  return rotations, centers


def test_visual_inertial_align_pins_the_unit_scale() -> None:
  """The velocities follow the unit scale, whatever the data alone implies."""
  gravity = np.array([0.0, 0.0, -9.81])
  dt = 0.1
  pims = [_pim_with_motion(dt, gravity, a) for a in _ACCEL_PHASES]
  dts = [float(pim.deltaTij()) for pim in pims]
  metric_positions, metric_velocities = _integrate_metric_trajectory(_ACCEL_PHASES, dts)

  # A metric seed returns scale ~1.0 and recovers the velocities.
  metric_poses = [gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(*p)) for p in metric_positions]
  a1 = visual_inertial_align(metric_poses, pims, gravity, velocity_regularization=0.0)
  assert abs(a1.scale - 1.0) < 1e-2
  assert np.allclose(a1.velocities[-1], metric_velocities[-1], atol=0.05)

  # A 1/3-scaled seed (true scale 3) stays near 1.0.
  third_poses = [
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(*(p / 3.0))) for p in metric_positions
  ]
  a2 = visual_inertial_align(third_poses, pims, gravity, velocity_regularization=0.0)
  assert abs(a2.scale - 1.0) < 1e-2


def test_batched_triangulation_recovers_points_and_rejects_degenerate_tracks() -> None:
  from specialist.vi_ba.init import triangulate_tracks_batched

  rng = np.random.default_rng(11)
  poses = [
    gtsam.Pose3(
      gtsam.Rot3.Ypr(0.05 * i, 0.02 * i, 0.0), gtsam.Point3(0.15 * i, 0.02 * i, 0)
    )
    for i in range(12)
  ]
  rotations, centers = _lens_arrays(poses)
  truth = np.column_stack(
    [rng.uniform(-1, 1, 60), rng.uniform(-1, 1, 60), rng.uniform(0.3, 4.0, 60)]
  )
  views = [
    sorted(rng.choice(12, size=rng.integers(1, 6), replace=False)) for _ in range(60)
  ]
  kf = np.concatenate(views).astype(np.int64)
  track = np.repeat(np.arange(60), [len(v) for v in views])
  rays = np.einsum("nji,nj->ni", rotations[kf], truth[track] - centers[kf])
  rays /= np.linalg.norm(rays, axis=1, keepdims=True)
  rays[track % 7 == 0, 2] *= -1.0  # behind the lens
  points, passed = triangulate_tracks_batched(
    rays,
    track,
    60,
    rotations[kf],
    centers[kf],
    min_landmark_depth=0.15,
    max_landmark_depth=5.0,
  )
  single_view = np.array([len(v) < 2 for v in views])
  assert not passed[single_view].any() and not passed[np.arange(60) % 7 == 0].any()
  assert passed.sum() > 20
  np.testing.assert_allclose(points[passed], truth[passed], atol=1e-6)
