"""``vi_ba.init``: two-view init, pose interpolation and triangulation."""

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
  interpolate_camera_poses,
  pick_init_pair,
  triangulate_track_multiview,
  two_view_essential,
  visual_inertial_align,
)

Float64Array = npt.NDArray[np.float64]


def _unit(v: Float64Array) -> Float64Array:
  return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_two_view_essential_recovers_pure_translation() -> None:
  """Camera moves along +x by 0.3 m; landmarks at z ∈ [1.5, 3] m."""
  rng = np.random.default_rng(3)
  n = 40
  pts3d_a = rng.uniform(-0.4, 0.4, size=(n, 3))
  pts3d_a[:, 2] = rng.uniform(1.5, 3.0, size=n)
  pts3d_b = pts3d_a - np.array([0.3, 0.0, 0.0])

  bearings_a = _unit(pts3d_a)
  bearings_b = _unit(pts3d_b)

  r_cw, t_cw, n_inliers = two_view_essential(bearings_a, bearings_b, threshold=1e-3)
  assert n_inliers >= 30

  # Translation is known up to scale: expect (-0.3, 0, 0) normalized in camera-B frame.
  t_dir = t_cw / np.linalg.norm(t_cw)
  assert abs(t_dir[0] + 1.0) < 0.1  # close to (-1, 0, 0)
  assert np.allclose(r_cw, np.eye(3), atol=5e-2)


def test_two_view_essential_too_few_bearings_raises() -> None:
  bearings = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
  bearings /= np.linalg.norm(bearings, axis=1, keepdims=True)
  try:
    two_view_essential(bearings, bearings, threshold=1e-3)
  except RuntimeError:
    return
  raise AssertionError("expected RuntimeError with <8 bearings")


def test_interpolate_camera_poses_endpoints_and_midpoint() -> None:
  r_cw_last = np.eye(3)
  t_cw_last = np.array([1.0, 0.0, 0.0])
  poses = interpolate_camera_poses(r_cw_last, t_cw_last, n_keyframes=5)
  assert len(poses) == 5
  # kf 0 is identity world→camera, ie. camera at origin.
  assert np.allclose(poses[0].translation(), (0.0, 0.0, 0.0))
  # poses holds T_world_lens, so keyframe 4's center is its translation.
  assert np.allclose(poses[-1].translation(), (-1.0, 0.0, 0.0), atol=1e-6)
  assert np.allclose(poses[2].translation(), (-0.5, 0.0, 0.0), atol=1e-6)


def test_pick_init_pair_finds_widest_with_min_shared() -> None:
  # Tracks A {0,1,2}, B {0,1,2,3}, C {0,4} over 5 frames; min shared 2 gives (0, 2, 2).
  tracks: dict[int, dict[int, Float64Array]] = {
    0: {0: np.zeros(2), 1: np.zeros(2), 2: np.zeros(2)},
    1: {0: np.zeros(2), 1: np.zeros(2), 2: np.zeros(2), 3: np.zeros(2)},
    2: {0: np.zeros(2), 4: np.zeros(2)},
  }
  first, last, n_shared = pick_init_pair(tracks, n_keyframes=5, min_shared=2)
  assert (first, last, n_shared) == (0, 4, 1) or n_shared >= 2


def test_pick_init_pair_falls_back_to_best_when_below_threshold() -> None:
  tracks: dict[int, dict[int, Float64Array]] = {
    0: {0: np.zeros(2), 1: np.zeros(2)},
  }
  first, last, n_shared = pick_init_pair(tracks, n_keyframes=3, min_shared=10)
  assert first == 0
  assert last == 1
  assert n_shared == 1


def test_interpolate_camera_poses_handles_mid_window_init_pair() -> None:
  """Init pair (2, 6) of 10: keyframes 0-1 extrapolate back, 7-9 forward."""
  r_cw_last = np.eye(3)
  t_cw_last = np.array([1.0, 0.0, 0.0])
  poses = interpolate_camera_poses(
    r_cw_last, t_cw_last, n_keyframes=10, init_first=2, init_last=6
  )
  assert len(poses) == 10
  # alpha for kf 2 = 0 → camera at origin (T_world_lens identity-ish).
  assert np.allclose(poses[2].translation(), (0.0, 0.0, 0.0), atol=1e-6)
  # alpha for kf 6 = 1 → translation matches the inverse-applied unit baseline.
  assert np.allclose(poses[6].translation(), (-1.0, 0.0, 0.0), atol=1e-6)
  # alpha for kf 0 = (0 - 2) / (6 - 2) = -0.5: extrapolate back half a baseline.
  assert np.allclose(poses[0].translation(), (0.5, 0.0, 0.0), atol=1e-6)


def _pim_with_motion(
  dt_total: float,
  gravity_world: np.ndarray,
  motion_accel_world: np.ndarray,
  rate: int = 1000,
) -> gtsam.PreintegratedCombinedMeasurements:
  """Preintegrate a level body reading ``motion_accel_world - gravity``."""
  n = max(2, int(dt_total * rate))
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


def _stationary_pim(
  dt_total: float,
  gravity_world: np.ndarray,
  rate: int = 1000,
) -> gtsam.PreintegratedCombinedMeasurements:
  return _pim_with_motion(dt_total, gravity_world, np.zeros(3), rate)


def test_visual_inertial_align_stationary_recovers_gravity_only() -> None:
  """Scale is unobservable at rest, but the velocity chain still yields gravity."""
  gravity = np.array([0.0, 0.0, -9.81])
  dt = 0.1
  n_kf = 6
  body_poses = [gtsam.Pose3() for _ in range(n_kf)]
  pims = [_stationary_pim(dt, gravity) for _ in range(n_kf - 1)]

  align = visual_inertial_align(body_poses, pims)
  assert isinstance(align, Alignment)
  assert np.allclose(align.gravity_world, gravity, atol=0.05)
  assert all(np.linalg.norm(v) < 0.05 for v in align.velocities)


def test_visual_inertial_align_recovers_scale_under_time_varying_acceleration() -> None:
  """Varying acceleration lets alignment recover scale, gravity and final velocity."""
  gravity = np.array([0.0, 0.0, -9.81])
  # Three acceleration phases to give the solver rank.
  accel_phases = [
    np.array([0.4, 0.3, 0.2]),
    np.array([-0.2, -0.1, 0.0]),
    np.array([0.1, -0.3, 0.2]),
    np.array([0.3, 0.1, -0.2]),
    np.array([-0.3, 0.4, 0.1]),
  ]
  dt = 0.1
  true_scale = 3.0
  pims = [_pim_with_motion(dt, gravity, a) for a in accel_phases]
  dts = [float(pim.deltaTij()) for pim in pims]

  # Integrate the accel phases forward from rest with constant-acceleration steps.
  metric_positions = [np.zeros(3)]
  metric_velocities = [np.zeros(3)]
  for i, a in enumerate(accel_phases):
    dt_i = dts[i]
    p_new = metric_positions[-1] + metric_velocities[-1] * dt_i + 0.5 * a * dt_i * dt_i
    v_new = metric_velocities[-1] + a * dt_i
    metric_positions.append(p_new)
    metric_velocities.append(v_new)
  body_poses = [
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(*(p / true_scale))) for p in metric_positions
  ]

  # The default Tikhonov weight would dominate the small synthetic velocities.
  align = visual_inertial_align(body_poses, pims, velocity_regularization=0.0)
  assert abs(align.scale - true_scale) < 0.05
  assert np.allclose(align.gravity_world, gravity, atol=0.1)
  assert np.allclose(align.velocities[-1], metric_velocities[-1], atol=0.05)


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


def test_triangulate_track_multiview_recovers_short_baseline() -> None:
  """A short-baseline, ten-view landmark at 1.2 m triangulates to within 1 mm."""
  body_T_lens = gtsam.Pose3()
  xyz_truth = np.array([0.05, 0.02, 1.2])
  # 10 poses with small translations (bbox ~10 cm).
  ts = np.linspace(-0.05, 0.05, 10)
  poses = [
    gtsam.Pose3(gtsam.Rot3.Rodrigues(0.0, 0.02 * t, 0.0), gtsam.Point3(t, 0.0, 0.0))
    for t in ts
  ]
  bearings = []
  for pose in poses:
    p_cam = np.asarray(pose.transformTo(gtsam.Point3(*xyz_truth)))
    bearings.append(p_cam / np.linalg.norm(p_cam))
  xyz_rec = triangulate_track_multiview(bearings, poses, body_T_lens)
  assert xyz_rec is not None
  assert np.allclose(xyz_rec, xyz_truth, atol=1e-3)


def test_triangulate_track_multiview_returns_none_for_single_view() -> None:
  body_T_lens = gtsam.Pose3()
  assert (
    triangulate_track_multiview(
      [np.array([0.0, 0.0, 1.0])],
      [gtsam.Pose3()],
      body_T_lens,
    )
    is None
  )


def test_triangulate_track_multiview_skips_bad_bearings_but_succeeds() -> None:
  """A bad-z bearing drops out; the three good views recover the point within 1 mm."""
  body_T_lens = gtsam.Pose3()
  xyz_truth = np.array([0.1, 0.0, 1.5])
  good_poses = [
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(t, 0.0, 0.0)) for t in (0.0, 0.1, 0.2)
  ]
  good_bearings = []
  for pose in good_poses:
    p_cam = np.asarray(pose.transformTo(gtsam.Point3(*xyz_truth)))
    good_bearings.append(p_cam / np.linalg.norm(p_cam))
  # Add one garbage bearing (z near 0).
  bearings = good_bearings + [np.array([1.0, 0.0, 0.001])]
  poses = good_poses + [gtsam.Pose3()]
  xyz_rec = triangulate_track_multiview(bearings, poses, body_T_lens)
  assert xyz_rec is not None
  assert np.allclose(xyz_rec, xyz_truth, atol=1e-3)


def test_triangulate_track_multiview_is_geometric_closest_point() -> None:
  """For skew rays the summed (I - ddᵀ)(X - C) gradient vanishes at the estimate."""
  body_T_lens = gtsam.Pose3()
  truth = np.array([0.1, -0.05, 1.5])
  poses = [
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(t, 0.0, 0.0)) for t in (-0.2, 0.0, 0.25)
  ]
  rng = np.random.default_rng(0)
  bearings = []
  for pose in poses:
    p_cam = np.asarray(pose.transformTo(gtsam.Point3(*truth)))
    b = p_cam / np.linalg.norm(p_cam) + 0.01 * rng.standard_normal(3)  # skew
    bearings.append(b / np.linalg.norm(b))
  xyz = triangulate_track_multiview(bearings, poses, body_T_lens)
  assert xyz is not None
  grad = np.zeros(3)
  for bearing, pose in zip(bearings, poses, strict=True):
    center = np.asarray(pose.translation())
    ray = pose.rotation().matrix() @ bearing
    ray = ray / np.linalg.norm(ray)
    grad += (np.eye(3) - np.outer(ray, ray)) @ (xyz - center)
  assert np.linalg.norm(grad) < 1e-9
