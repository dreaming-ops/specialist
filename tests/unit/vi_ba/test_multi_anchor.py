from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import pytest

from specialist.loader.mei_model import MeiParams, mei_project
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.compose import (
  CameraKeys,
  _bfs_world_alignments,
  _sighting_bearing,
  align_to_shared_world,
  anchored_keyframes,
)
from specialist.vi_ba.configs import (
  AnchorCfg,
  Args,
  SolverCfg,
  WindowCfg,
)
from specialist.vi_ba.features import mask_plane
from specialist.vi_ba.imu import make_preintegration_params
from specialist.vi_ba.reference_solve import (
  _secondary_anchor_seed,
  _solve_joint_graph,
)
from specialist.vi_ba.single_camera import (
  PreparedCamera,
  _build_non_anchor_tag_masks,
  _choose_prep_anchor,
  _print_anchor_geometry,
)
from specialist.vi_ba.tag_detection import (
  TagDetection,
  tag_center_bearing,
  tag_corners_in_frame,
)
from tests.mei_fixtures import narrow_mei_params


def _pose(rx: float, ry: float, rz: float, x: float, y: float, z: float) -> gtsam.Pose3:
  return gtsam.Pose3(gtsam.Rot3.Rodrigues(rx, ry, rz), gtsam.Point3(x, y, z))


def _tag(tag_id: int, cx: float, cy: float) -> TagDetection:
  half = 5.0
  corners = np.array(
    [
      [cx - half, cy - half],
      [cx + half, cy - half],
      [cx + half, cy + half],
      [cx - half, cy + half],
    ],
    dtype=np.float64,
  )
  return TagDetection(
    tag_id=tag_id, center=np.array([cx, cy], dtype=np.float64), corners=corners
  )


@pytest.mark.parametrize(
  ("anchor_ids", "expected"),
  [({1, 0}, (255, 255, 0)), ({1}, (255, 0, 0)), (set(), (0, 0, 0))],
)
def test_masks_unmask_exactly_the_configured_anchor_ids(
  anchor_ids: set[int], expected: tuple[int, int, int]
) -> None:
  """Configured anchor tags stay unmasked; every other tag region is masked."""
  frames = [np.zeros((60, 90), dtype=np.uint8)]
  tags = [[_tag(1, 15.0, 30.0), _tag(0, 45.0, 30.0), _tag(7, 75.0, 30.0)]]
  blocked = _build_non_anchor_tag_masks(frames[0].shape[:2], tags, anchor_ids, 2)
  plane = mask_plane(frames[0].shape[:2], blocked[0])
  for x, value in zip((15, 45, 75), expected, strict=True):
    assert plane[30, x] == value


def _two_anchor_scene() -> tuple[gtsam.Pose3, gtsam.Pose3, gtsam.Pose3]:
  """Both alignment tests' ``(world_T_anchor1, world_T_anchor0, shared_T_b_local)``."""
  return (
    _pose(0.1, 0.0, 0.0, 1.0, 0.0, 0.5),
    _pose(0.0, 0.2, 0.0, 0.0, 1.0, 0.3),
    _pose(0.0, 0.0, 0.3, 0.5, -0.2, 0.1),
  )


def test_bfs_alignment_joins_camera_through_secondary_anchor() -> None:
  """Camera "b", seeing only the secondary, aligns through it to its true transform."""
  anchor_1, anchor_0, shared_T_b_local = _two_anchor_scene()
  locals_by_camera = {
    "a": {1: anchor_1, 0: anchor_0},
    "b": {0: shared_T_b_local.inverse().compose(anchor_0)},
  }
  aligns, world_T_anchor = _bfs_world_alignments(locals_by_camera, (1, 0))
  assert np.allclose(aligns["a"].matrix(), np.eye(4), atol=1e-12)
  assert np.allclose(aligns["b"].matrix(), shared_T_b_local.matrix(), atol=1e-9)
  assert np.allclose(world_T_anchor[1].matrix(), anchor_1.matrix(), atol=1e-12)
  assert np.allclose(world_T_anchor[0].matrix(), anchor_0.matrix(), atol=1e-12)


def test_bfs_alignment_disconnected_raises() -> None:
  locals_by_camera: dict[str, dict[int, gtsam.Pose3]] = {
    "a": {1: gtsam.Pose3()},
    "b": {},
  }
  with pytest.raises(RuntimeError, match="'b'"):
    _bfs_world_alignments(locals_by_camera, (1, 0))


def test_bfs_alignment_without_reference_sighting_raises() -> None:
  with pytest.raises(RuntimeError, match="reference"):
    _bfs_world_alignments({"a": {0: gtsam.Pose3()}}, (1, 0))


def _dummy_camera(
  name: str,
  body_poses: list[gtsam.Pose3],
  extra_anchor_obs: dict[int, dict[int, dict[str, gtsam.Pose3]]],
  front_ext: gtsam.Pose3,
  rear_ext: gtsam.Pose3,
  prep_anchor_tag_id: int = 1,
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]] | None = None,
  anchor_local: gtsam.Pose3 | None = None,
  extra_anchor_locals: dict[int, gtsam.Pose3] | None = None,
  biases: list[gtsam.imuBias.ConstantBias] | None = None,
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements] | None = None,
  front_tags_per_kf: list[list[TagDetection]] | None = None,
  velocities: list[np.ndarray] | None = None,
) -> PreparedCamera:
  n = len(body_poses)
  return PreparedCamera(
    name=name,
    body_poses=body_poses,
    velocities=(
      velocities if velocities is not None else [np.zeros(3) for _ in range(n)]
    ),
    biases=biases or [],
    preintegrated=preintegrated or [],
    front_lens=LensSetup(body_T_lens=front_ext, params=narrow_mei_params()),
    rear_lens=LensSetup(body_T_lens=rear_ext, params=narrow_mei_params()),
    front_tracks_tri={},
    rear_tracks_tri={},
    front_points={},
    rear_points={},
    decoded_width=2880,
    anchor_obs_by_tag=(
      {prep_anchor_tag_id: anchor_obs_per_kf} if anchor_obs_per_kf else {}
    )
    | extra_anchor_obs,
    anchor_locals_by_tag=(
      {prep_anchor_tag_id: anchor_local} if anchor_local is not None else {}
    )
    | (extra_anchor_locals or {}),
    body_T_base_tag=None,
    keyframe_times=np.zeros(n),
    front_keypoints=[],
    rear_keypoints=[],
    front_tags_per_kf=(
      front_tags_per_kf if front_tags_per_kf is not None else [[] for _ in range(n)]
    ),
    rear_tags_per_kf=[[] for _ in range(n)],
    front_rejected={},
    rear_rejected={},
    rear_static_mask_rects=[],
    finger_reproj={},
    left_finger_tag_id=0,
    right_finger_tag_id=0,
  )


def _lens_T_tag(
  body: gtsam.Pose3, body_T_lens: gtsam.Pose3, world_T_tag: gtsam.Pose3
) -> gtsam.Pose3:
  return body.compose(body_T_lens).inverse().compose(world_T_tag)


def _front_obs(
  bodies: list[gtsam.Pose3], ext: gtsam.Pose3, world_T_tag: gtsam.Pose3
) -> dict[int, dict[str, gtsam.Pose3]]:
  return {
    k: {"front": _lens_T_tag(body, ext, world_T_tag)} for k, body in enumerate(bodies)
  }


def test_secondary_anchor_seed_matches_world_tag_pose() -> None:
  """PnP sightings of a known ``world_T_tag`` seed back exactly that pose."""
  world_T_tag = _pose(0.2, -0.1, 0.3, 0.4, 0.6, 1.1)
  front_ext = _pose(0.05, 0.0, 0.0, 0.03, 0.0, 0.0)
  rear_ext = _pose(0.0, -0.04, 0.0, -0.02, 0.01, 0.0)
  bodies_a = [_pose(0.0, 0.02 * k, 0.0, 0.1 * k, 0.0, 0.4) for k in range(4)]
  bodies_b = [_pose(0.01 * k, 0.0, 0.0, 0.0, 0.1 * k, 0.6) for k in range(3)]
  obs_a = {
    k: {
      "front": _lens_T_tag(bodies_a[k], front_ext, world_T_tag),
      "rear": _lens_T_tag(bodies_a[k], rear_ext, world_T_tag),
    }
    for k in range(4)
  }
  obs_b = _front_obs(bodies_b, front_ext, world_T_tag)
  cameras = [
    _dummy_camera("a", bodies_a, {5: obs_a}, front_ext, rear_ext),
    _dummy_camera("b", bodies_b, {5: obs_b}, front_ext, rear_ext),
  ]
  seed = _secondary_anchor_seed(cameras, 5)
  assert seed is not None
  assert np.allclose(seed.matrix(), world_T_tag.matrix(), atol=1e-9)


def test_secondary_anchor_seed_median_ignores_outlier_sighting() -> None:
  """One flipped PnP sighting leaves the median seed on the true pose."""
  world_T_tag = _pose(0.0, 0.1, 0.0, 0.3, 0.2, 0.9)
  ext = gtsam.Pose3()
  bodies = [_pose(0.0, 0.01 * k, 0.0, 0.05 * k, 0.0, 0.5) for k in range(5)]
  obs = _front_obs(bodies, ext, world_T_tag)
  obs[2] = {"front": _lens_T_tag(bodies[2], ext, _pose(1.5, 0.0, 0.0, 5.0, -3.0, 2.0))}
  camera = _dummy_camera("a", bodies, {5: obs}, ext, ext)
  seed = _secondary_anchor_seed([camera], 5)
  assert seed is not None
  assert np.allclose(seed.matrix(), world_T_tag.matrix(), atol=1e-9)


def test_secondary_anchor_seed_none_when_unseen() -> None:
  camera = _dummy_camera("a", [gtsam.Pose3()], {}, gtsam.Pose3(), gtsam.Pose3())
  assert _secondary_anchor_seed([camera], 7) is None


def _obs_at_kfs(n_kfs: int) -> dict[int, dict[str, gtsam.Pose3]]:
  return {kf: {"front": gtsam.Pose3()} for kf in range(n_kfs)}


def test_prep_anchor_choice_picks_most_sighted_id() -> None:
  obs_by_tag = {1: _obs_at_kfs(59), 0: _obs_at_kfs(1225)}
  assert _choose_prep_anchor((1, 0), obs_by_tag) == 0


def test_prep_anchor_choice_single_id_and_tie_break() -> None:
  """Equal counts break toward the earlier id in the configured tuple."""
  assert _choose_prep_anchor((1,), {1: _obs_at_kfs(3)}) == 1
  assert _choose_prep_anchor((1,), {1: {}}) == 1
  assert _choose_prep_anchor((1, 0), {1: _obs_at_kfs(4), 0: _obs_at_kfs(4)}) == 1


def test_alignment_grounds_camera_via_its_prep_anchor() -> None:
  anchor_1, anchor_0, shared_T_b_local = _two_anchor_scene()
  a = _dummy_camera(
    "a",
    [gtsam.Pose3()],
    {},
    gtsam.Pose3(),
    gtsam.Pose3(),
    prep_anchor_tag_id=1,
    anchor_local=anchor_1,
    extra_anchor_locals={0: anchor_0},
  )
  b = _dummy_camera(
    "b",
    [gtsam.Pose3()],
    {},
    gtsam.Pose3(),
    gtsam.Pose3(),
    prep_anchor_tag_id=0,
    anchor_local=shared_T_b_local.inverse().compose(anchor_0),
  )
  ref = align_to_shared_world([a, b], (1, 0))
  assert np.allclose(ref.matrix(), anchor_1.matrix())
  assert np.allclose(a.body_poses[0].matrix(), np.eye(4), atol=1e-12)
  assert np.allclose(b.body_poses[0].matrix(), shared_T_b_local.matrix(), atol=1e-9)


def test_secondary_anchor_seed_includes_prep_reference_observations() -> None:
  """The seed consumes prep-reference sightings stored in ``anchor_obs_per_kf``."""
  world_T_tag = _pose(0.1, 0.0, -0.2, 0.4, 0.3, 1.0)
  ext = gtsam.Pose3()
  bodies = [_pose(0.0, 0.02 * k, 0.0, 0.06 * k, 0.0, 0.5) for k in range(4)]
  obs = _front_obs(bodies, ext, world_T_tag)
  camera = _dummy_camera(
    "b", bodies, {}, ext, ext, prep_anchor_tag_id=5, anchor_obs_per_kf=obs
  )
  seed = _secondary_anchor_seed([camera], 5)
  assert seed is not None
  assert np.allclose(seed.matrix(), world_T_tag.matrix(), atol=1e-9)


def test_secondary_anchor_seed_uses_pose_override() -> None:
  """The between-round re-estimate composes sightings against the given solved poses."""
  world_T_tag = _pose(0.0, 0.1, 0.0, 0.2, 0.4, 0.9)
  ext = gtsam.Pose3()
  true_bodies = [_pose(0.0, 0.01 * k, 0.0, 0.05 * k, 0.0, 0.4) for k in range(3)]
  obs = _front_obs(true_bodies, ext, world_T_tag)
  drift = _pose(0.0, 0.0, 0.1, 0.3, -0.2, 0.1)
  prep_bodies = [drift.compose(p) for p in true_bodies]
  camera = _dummy_camera("a", prep_bodies, {5: obs}, ext, ext)
  drifted = _secondary_anchor_seed([camera], 5)
  assert drifted is not None
  assert not np.allclose(drifted.matrix(), world_T_tag.matrix(), atol=1e-3)
  seed = _secondary_anchor_seed([camera], 5, body_poses_by_tracker={"a": true_bodies})
  assert seed is not None
  assert np.allclose(seed.matrix(), world_T_tag.matrix(), atol=1e-9)


def _project_tag(
  body: gtsam.Pose3,
  params: MeiParams,
  world_T_tag: gtsam.Pose3,
  tag_id: int,
  tag_size: float,
) -> TagDetection:
  corners_lens = np.array(
    [body.transformTo(c) for c in tag_corners_in_frame(world_T_tag, tag_size)]
  )
  center_lens = np.asarray(body.transformTo(world_T_tag.translation()))
  return TagDetection(
    tag_id=tag_id,
    center=mei_project(center_lens[None, :], params)[0],
    corners=mei_project(corners_lens, params),
  )


def _pim_chain(
  n: int, body_rot: gtsam.Rot3
) -> list[gtsam.PreintegratedCombinedMeasurements]:
  """An exact IMU chain for the fixture's constant-velocity, fixed-attitude motion."""
  params = make_preintegration_params(
    gyro_noise_sigma=0.01,
    accel_noise_sigma=0.2,
    gyro_bias_rw_sigma=1.0e-3,
    accel_bias_rw_sigma=1.0e-3,
    gravity_world=(0.0, 0.0, -9.81),
  )
  accel_body = body_rot.matrix().T @ np.array([0.0, 0.0, 9.81])
  pims = []
  for _ in range(n - 1):
    pim = gtsam.PreintegratedCombinedMeasurements(params, gtsam.imuBias.ConstantBias())
    for _ in range(10):
      pim.integrateMeasurement(accel_body, np.zeros(3), 0.01)
    pims.append(pim)
  return pims


def _pinned_solve_fixture(
  with_secondary: bool,
) -> tuple[list[PreparedCamera], CameraKeys, Args, gtsam.Pose3, gtsam.Pose3]:
  """A constant-velocity camera: exact anchor sightings, perturbed prep poses."""
  ext = gtsam.Pose3()
  world_T_ref = _pose(0.02, 0.01, 0.0, 0.3, 0.0, 1.2)
  world_T_sec = _pose(-0.01, 0.03, 0.02, -0.3, 0.1, 1.1)
  n = 4
  attitude = gtsam.Rot3.Rodrigues(0.02, -0.01, 0.03)
  true_poses = [
    gtsam.Pose3(attitude, gtsam.Point3(0.1 * k, 0.02 * k, 0.01 * k)) for k in range(n)
  ]
  velocity = np.array([1.0, 0.2, 0.1])
  wobble = _pose(0.02, -0.015, 0.01, 0.05, -0.04, 0.03)
  prep_poses = [true_poses[0]] + [true_poses[k].compose(wobble) for k in range(1, n)]
  tag_size = 0.1
  tags_per_kf = []
  for k in range(n):
    dets = [_project_tag(true_poses[k], narrow_mei_params(), world_T_ref, 1, tag_size)]
    if with_secondary and k == 1:
      dets.append(
        _project_tag(true_poses[k], narrow_mei_params(), world_T_sec, 5, tag_size)
      )
    tags_per_kf.append(dets)
  anchor_obs = _front_obs(true_poses, ext, world_T_ref)
  extra = (
    {5: {1: {"front": _lens_T_tag(true_poses[1], ext, world_T_sec)}}}
    if with_secondary
    else {}
  )
  camera = _dummy_camera(
    "a",
    prep_poses,
    extra,
    ext,
    ext,
    prep_anchor_tag_id=1,
    anchor_obs_per_kf=anchor_obs,
    front_tags_per_kf=tags_per_kf,
    velocities=[velocity.copy() for _ in range(n)],
    biases=[gtsam.imuBias.ConstantBias() for _ in range(n)],
    preintegrated=_pim_chain(n, attitude),
  )
  args = Args(
    input_insvs=(Path("a.insv"),),
    window=WindowCfg(duration=None),
    anchor=AnchorCfg(tag_ids=(1, 5) if with_secondary else (1,)),
    base_tag_ids=(2,),
    solver=SolverCfg(velocity_prior_sigma=0.01),
  )
  return [camera], CameraKeys(["a"]), args, world_T_ref, world_T_sec


def _assert_no_corner_point_variables(values: gtsam.Values) -> None:
  """Solved values hold no landmark points; freed corners have their own symbol char."""
  key_chars = {gtsam.Symbol(k).chr() for k in values.keys()}
  assert ord("A") not in key_chars


def test_pinned_solve_reestimates_secondary_pose(
  capsys: pytest.CaptureFixture[str],
) -> None:
  """From a seed off the true secondary pose, the post-solve re-estimate lands on it."""
  cameras, keys, args, world_T_ref, world_T_sec = _pinned_solve_fixture(True)
  seed = _secondary_anchor_seed(cameras, 5)
  assert seed is not None
  true_t = np.asarray(world_T_sec.translation(), dtype=np.float64)
  seed_err = float(np.linalg.norm(np.asarray(seed.translation()) - true_t))
  assert seed_err > 0.02
  solve = _solve_joint_graph(cameras, keys, args, [], world_T_ref)
  world = solve.secondary_world
  assert 5 in world
  assert solve.secondary_moves[5] == solve.worst_secondary_move
  final_err = float(np.linalg.norm(np.asarray(world[5].translation()) - true_t))
  assert final_err < 0.01
  assert final_err < seed_err / 3
  out = capsys.readouterr().out
  assert "secondary anchor 5: post-solve re-estimate moved" in out
  _assert_no_corner_point_variables(solve.result.values)


def test_single_anchor_takes_single_solve_branch(
  capsys: pytest.CaptureFixture[str],
) -> None:
  """Without a sighted secondary anchor, one cull-reoptimize pass is the whole solve."""
  cameras, keys, args, world_T_ref, _sec = _pinned_solve_fixture(False)
  solve = _solve_joint_graph(cameras, keys, args, [], world_T_ref)
  out = capsys.readouterr().out
  assert solve.secondary_world == {}
  assert all(np.isnan(v) for v in solve.worst_secondary_move)
  assert out.count("solve 1:") == 1
  _assert_no_corner_point_variables(solve.result.values)


def test_sighting_bearing_falls_back_to_detection_ray() -> None:
  """Without PnP, the detection-center ray keeps the behind-camera guard active."""
  lens = LensSetup(body_T_lens=gtsam.Pose3(), params=narrow_mei_params())
  lens_T_tag = _pose(0.0, 0.0, 0.0, 0.1, 0.0, 0.9)
  det = _project_tag(gtsam.Pose3(), narrow_mei_params(), lens_T_tag, 1, 0.1)
  with_obs = _sighting_bearing({0: {"front": lens_T_tag}}, 0, "front", det, lens)
  direction = np.asarray(lens_T_tag.translation())
  assert np.allclose(with_obs, direction / np.linalg.norm(direction))
  assert np.isclose(np.linalg.norm(with_obs), 1.0)
  fallback = _sighting_bearing({}, 0, "front", det, lens)
  assert np.allclose(fallback, tag_center_bearing(det, lens.params))


def test_anchored_keyframes_union_over_all_anchors() -> None:
  camera = _dummy_camera(
    "a",
    [gtsam.Pose3() for _ in range(4)],
    {5: {2: {"front": gtsam.Pose3()}}},
    gtsam.Pose3(),
    gtsam.Pose3(),
    anchor_obs_per_kf={0: {"front": gtsam.Pose3()}, 1: {"front": gtsam.Pose3()}},
  )
  assert anchored_keyframes(camera) == {0, 1, 2}


def test_anchor_geometry_print_is_observational(
  capsys: pytest.CaptureFixture[str],
) -> None:
  """The geometry line keeps inputs, returns nothing and skips unsighted anchors."""
  ext = gtsam.Pose3()
  world_T_tag = _pose(0.0, 0.0, 0.0, 0.1, 0.0, 0.9)
  det = _project_tag(ext, narrow_mei_params(), world_T_tag, 5, 0.1)
  lens = LensSetup(body_T_lens=ext, params=narrow_mei_params())
  lens_sides = (("front", lens, [[det]], [np.eye(3)]),)
  obs = {0: {"front": world_T_tag}}
  before = {kf: dict(sides) for kf, sides in obs.items()}
  assert _print_anchor_geometry("a", 5, lens_sides, obs) is None
  out = capsys.readouterr().out
  assert "anchor 5 geometry" in out
  assert obs == before
  _print_anchor_geometry("a", 7, lens_sides, {})
  assert "anchor 7" not in capsys.readouterr().out
