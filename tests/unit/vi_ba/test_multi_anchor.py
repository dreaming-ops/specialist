from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import pytest
from specialist_camera import MeiParams, mei_project
from specialist_dataset.fiducials import tag_corners_in_frame

from specialist.vi_ba.build import LensChannel, LensSetup
from specialist.vi_ba.compose import (
  CameraKeys,
  _bfs_world_alignments,
  align_to_shared_world,
  anchored_keyframes,
)
from specialist.vi_ba.configs import (
  AnchorCfg,
  Args,
  SolverCfg,
)
from specialist.vi_ba.features import mask_plane
from specialist.vi_ba.geometry import average_pose
from specialist.vi_ba.imu import ImuSamples, make_preintegration_params
from specialist.vi_ba.reference_solve import ReferenceSolve, _solve_joint_graph
from specialist.vi_ba.single_camera import (
  PreparedCamera,
  _choose_prep_anchor,
)
from specialist.vi_ba.symbols import LANDMARK_CHAR
from specialist.vi_ba.tag_detection import (
  TagDetection,
  non_anchor_tag_masks,
)
from specialist.vi_ba.tag_discovery import GripperTags
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
  blocked = non_anchor_tag_masks(frames[0].shape[:2], tags, anchor_ids, 2)
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
  """Camera "a" sights both anchors and camera "b" only the secondary."""
  anchor_1, anchor_0, shared_T_b_local = _two_anchor_scene()
  locals_by_camera = {
    "a": {1: anchor_1, 0: anchor_0},
    "b": {0: shared_T_b_local.inverse().compose(anchor_0)},
  }
  counts = {
    slot: dict.fromkeys(locals_, 1) for slot, locals_ in locals_by_camera.items()
  }
  aligns, world_T_anchor = _bfs_world_alignments(locals_by_camera, counts, (1, 0))
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
    _bfs_world_alignments(
      locals_by_camera,
      {slot: dict.fromkeys(locals_, 1) for slot, locals_ in locals_by_camera.items()},
      (1, 0),
    )


def test_bfs_alignment_without_reference_sighting_raises() -> None:
  with pytest.raises(RuntimeError, match="reference"):
    _bfs_world_alignments({"a": {0: gtsam.Pose3()}}, {"a": {0: 1}}, (1, 0))


def _dummy_camera(
  name: str,
  body_poses: list[gtsam.Pose3],
  anchor_obs: dict[int, dict[int, gtsam.Pose3]] | None = None,
  anchor_locals: dict[int, gtsam.Pose3] | None = None,
  biases: list[gtsam.imuBias.ConstantBias] | None = None,
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements] | None = None,
  front_tags_per_kf: list[list[TagDetection]] | None = None,
  velocities: list[np.ndarray] | None = None,
) -> PreparedCamera:
  n = len(body_poses)
  setup = LensSetup(body_T_lens=gtsam.Pose3(), params=narrow_mei_params())
  return PreparedCamera(
    name=name,
    # Every capture carries one; these tests never propagate it.
    imu=ImuSamples(
      t=np.zeros(1, dtype=np.float64),
      gyro=np.zeros((1, 3)),
      accel=np.zeros((1, 3)),
    ),
    keyframe_frames=np.arange(n, dtype=np.int64) * 3,
    body_poses=body_poses,
    velocities=(
      velocities if velocities is not None else [np.zeros(3) for _ in range(n)]
    ),
    biases=biases or [],
    preintegrated=preintegrated or [],
    front=LensChannel(
      lens="front", setup=setup, tags=front_tags_per_kf or [[] for _ in range(n)]
    ),
    rear=LensChannel(lens="rear", setup=setup, tags=[[] for _ in range(n)]),
    anchor_obs_by_tag=anchor_obs or {},
    anchor_locals_by_tag=anchor_locals or {},
    body_T_base_tag=gtsam.Pose3(),
    keyframe_times=np.zeros(n),
    finger_reproj={},
    tags=GripperTags(2, (None, None)),
  )


def _front_obs(
  bodies: list[gtsam.Pose3], world_T_tag: gtsam.Pose3
) -> dict[int, gtsam.Pose3]:
  return {k: body.between(world_T_tag) for k, body in enumerate(bodies)}


def _obs_at_kfs(n_kfs: int) -> dict[int, gtsam.Pose3]:
  return {kf: gtsam.Pose3() for kf in range(n_kfs)}


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
  a = _dummy_camera("a", [gtsam.Pose3()], anchor_locals={1: anchor_1, 0: anchor_0})
  b = _dummy_camera(
    "b",
    [gtsam.Pose3()],
    anchor_locals={0: shared_T_b_local.inverse().compose(anchor_0)},
  )
  world = align_to_shared_world([a, b], (1, 0))
  assert np.allclose(
    world[1].between(world[0]).matrix(), anchor_1.between(anchor_0).matrix()
  )
  assert np.allclose(
    a.body_poses[0].between(world[1]).matrix(), anchor_1.matrix(), atol=1e-12
  )
  assert np.allclose(
    b.body_poses[0].between(world[0]).matrix(),
    shared_T_b_local.between(anchor_0).matrix(),
    atol=1e-9,
  )


def test_alignment_carries_velocities_and_landmarks() -> None:
  """One rigid transform moves body poses, velocities and landmarks together."""
  # A reference tag off world x, so the canonical frame carries a real rotation.
  anchor_local = _pose(0.12, -0.08, 0.7, 0.3, 0.4, 1.0)
  velocity = np.array([0.3, -0.2, 0.1])
  landmark = np.array([0.4, 0.5, 0.6])
  a = _dummy_camera(
    "a", [gtsam.Pose3()], anchor_locals={1: anchor_local}, velocities=[velocity.copy()]
  )
  a.front.points = {7: landmark.copy()}
  before = a.body_poses[0]
  align_to_shared_world([a], (1,))
  align = a.body_poses[0].compose(before.inverse())
  align_R = align.rotation().matrix()
  assert not np.allclose(align_R, np.eye(3))
  assert np.allclose(a.velocities[0], align_R @ velocity)
  assert np.allclose(
    a.front.points[7], align_R @ landmark + np.asarray(align.translation())
  )


def test_worst_anchor_move_reports_the_largest_non_reference_move() -> None:
  """The reference moves freely on its horizontal rotations, so it never counts."""
  solve = ReferenceSolve(
    ref_tag_id=1,
    result=None,  # type: ignore[arg-type]
    landmark_keep=None,  # type: ignore[arg-type]
    world_T_anchor={},
    body_poses={},
    anchor_moves={1: (9.0, 9.0), 5: (0.02, 0.03), 6: (0.07, 0.01)},
  )
  assert solve.worst_anchor_move == (0.07, 0.03)


def test_alignment_places_the_reference_anchor_canonically() -> None:
  """The reference lands at the origin, x leveled onto world x, tilt untouched."""
  anchor = _pose(0.1, -0.2, 0.7, 1.0, 2.0, 0.5)
  a = _dummy_camera("a", [gtsam.Pose3()], anchor_locals={1: anchor})
  world = align_to_shared_world([a], (1,))
  placed = world[1]
  assert np.allclose(np.asarray(placed.translation()), 0.0, atol=1e-12)
  x_axis = placed.rotation().matrix()[:, 0]
  assert abs(x_axis[1]) < 1e-12 and x_axis[0] > 0.0
  assert np.allclose(placed.rotation().matrix()[2], anchor.rotation().matrix()[2])
  assert np.allclose(
    a.body_poses[0].between(placed).matrix(), anchor.matrix(), atol=1e-12
  )


def test_alignment_seeds_an_anchor_from_the_camera_that_sighted_it_most() -> None:
  """The reference camera's two wrong sightings lose to another camera's ten."""
  anchor_1, anchor_0, _ = _two_anchor_scene()
  off = _pose(0.2, 0.0, 0.0, 0.1, 0.0, 0.0).compose(anchor_0)
  a = _dummy_camera(
    "a",
    [gtsam.Pose3()],
    anchor_obs={0: _obs_at_kfs(2)},
    anchor_locals={1: anchor_1, 0: off},
  )
  b = _dummy_camera(
    "b",
    [gtsam.Pose3()],
    anchor_obs={0: _obs_at_kfs(10)},
    anchor_locals={1: anchor_1, 0: anchor_0},
  )
  world = align_to_shared_world([a, b], (1, 0))
  assert np.allclose(
    world[1].between(world[0]).matrix(), anchor_1.between(anchor_0).matrix(), atol=1e-12
  )


def _project_tag(
  body: gtsam.Pose3,
  params: MeiParams,
  world_T_tag: gtsam.Pose3,
  tag_id: int,
  tag_size: float,
) -> TagDetection:
  corners_lens = np.array(
    [body.transformTo(c) for c in tag_corners_in_frame(world_T_tag.matrix(), tag_size)]
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


def _pinned_solve_fixture() -> tuple[
  list[PreparedCamera], CameraKeys, Args, gtsam.Pose3, gtsam.Pose3
]:
  """A constant-velocity camera: exact anchor sightings, perturbed prep poses."""
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
  tags_per_kf = [
    [
      _project_tag(pose, narrow_mei_params(), world_T, tag_id, tag_size)
      for tag_id, world_T in ((1, world_T_ref), (5, world_T_sec))
    ]
    for pose in true_poses
  ]
  camera = _dummy_camera(
    "a",
    prep_poses,
    anchor_obs={
      1: _front_obs(true_poses, world_T_ref),
      5: _front_obs(true_poses, world_T_sec),
    },
    front_tags_per_kf=tags_per_kf,
    velocities=[velocity.copy() for _ in range(n)],
    biases=[gtsam.imuBias.ConstantBias() for _ in range(n)],
    preintegrated=_pim_chain(n, attitude),
  )
  args = Args(
    base=Path("session_base.rrd"),
    anchor=AnchorCfg(tag_ids=(1, 5)),
    solver=SolverCfg(velocity_prior_sigma=0.01, gpu="off"),
  )
  return [camera], CameraKeys(["a"]), args, world_T_ref, world_T_sec


def _seed_for(
  cameras: list[PreparedCamera], args: Args, ref: gtsam.Pose3
) -> dict[int, gtsam.Pose3]:
  """The reference at ``ref``; each other anchor at its median prep-pose sighting."""
  seed = {int(args.anchor.tag_ids[0]): ref}
  for tag_id in args.anchor.tag_ids[1:]:
    candidates = [
      camera.body_poses[kf].compose(camera.front.setup.body_T_lens).compose(lens_T_tag)
      for camera in cameras
      for kf, lens_T_tag in camera.anchor_obs_by_tag.get(tag_id, {}).items()
    ]
    if candidates:
      seed[tag_id] = average_pose(candidates)
  return seed


def test_solve_recovers_a_secondary_anchor_from_a_drifted_seed(
  capsys: pytest.CaptureFixture[str],
) -> None:
  """The bundle owns the anchor, so its solved pose lands on truth."""
  cameras, keys, args, world_T_ref, world_T_sec = _pinned_solve_fixture()
  seed = _seed_for(cameras, args, world_T_ref)
  true_t = np.asarray(world_T_sec.translation(), dtype=np.float64)
  seed_err = float(np.linalg.norm(np.asarray(seed[5].translation()) - true_t))
  assert seed_err > 0.02
  solve = _solve_joint_graph(cameras, keys, args, [], seed)
  world = solve.world_T_anchor
  assert 5 in world
  assert solve.anchor_moves[5] == solve.worst_anchor_move
  final_err = float(np.linalg.norm(np.asarray(world[5].translation()) - true_t))
  assert final_err < 0.01
  assert final_err < seed_err / 3
  assert "tag 5 moved" in capsys.readouterr().out
  for tag_id, pose in world.items():
    assert pose.equals(solve.result.values.atPose3(keys.anchor_key(tag_id)), 1e-12)
  key_chars = {gtsam.Symbol(k).chr() for k in solve.result.values.keys()}
  assert ord(LANDMARK_CHAR) not in key_chars


def test_anchored_keyframes_union_over_all_anchors() -> None:
  camera = _dummy_camera(
    "a",
    [gtsam.Pose3() for _ in range(4)],
    anchor_obs={1: {0: gtsam.Pose3(), 1: gtsam.Pose3()}, 5: {2: gtsam.Pose3()}},
  )
  assert anchored_keyframes(camera) == {0, 1, 2}
