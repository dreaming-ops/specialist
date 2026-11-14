"""Each camera's estimate and summary from one solved joint graph."""

from dataclasses import dataclass, replace

import gtsam
import numpy as np
import numpy.typing as npt
from specialist_camera import mei_project
from specialist_dataset.fiducials import tag_corners_in_frame

from specialist.vi_ba.build import LensChannel, LensSetup
from specialist.vi_ba.compose import (
  CameraKeys,
  CrossSighting,
  LandmarkTracks,
  anchor_sightings,
  anchored_keyframes,
  points_from_values,
  seat_evidence,
)
from specialist.vi_ba.configs import Args
from specialist.vi_ba.finger_kinematics import pin_angles_to_keyframes
from specialist.vi_ba.geometry import bounding_box_diagonal
from specialist.vi_ba.gpu_lm import OptimizeResult
from specialist.vi_ba.highrate import propagate_highrate
from specialist.vi_ba.landmark_cull import cull_and_reproject
from specialist.vi_ba.reference_solve import ReferenceSolve
from specialist.vi_ba.single_camera import PreparedCamera
from specialist.vi_ba.summary import CameraSummary
from specialist.vi_ba.tag_detection import TagDetection, estimate_lens_T_tags
from specialist.vi_ba.urdf_joints import FINGER_JOINTS
from specialist.vi_ba.viz import CameraEstimate, LensEstimate

Float64Array = npt.NDArray[np.float64]

__all__ = ["RunEstimate", "build_run_estimate"]


def _rays(
  poses: list[gtsam.Pose3], by_kf: dict[int, list[tuple[LensSetup, Float64Array]]]
) -> list[Float64Array]:
  """Per keyframe, ``(n, 2, 3)`` segments from each sighting lens to each corner it saw."""
  return [
    np.asarray(
      [
        (np.asarray(world_T_body.compose(lens.body_T_lens).translation()), corner)
        for lens, corners in by_kf.get(kf, ())
        for corner in corners
      ],
      dtype=np.float64,
    ).reshape(-1, 2, 3)
    for kf, world_T_body in enumerate(poses)
  ]


def _factor_observation_lines(
  observer: PreparedCamera,
  observer_poses: list[gtsam.Pose3],
  cross_obs_for_observer: list[CrossSighting],
  anchor_corners_by_tag: dict[int, Float64Array],
  base_tag_corners_world: dict[str, list[Float64Array]],
) -> tuple[list[Float64Array], list[Float64Array]]:
  """Per-keyframe ``(n, 2, 3)`` lens-to-corner lines to anchors and base tags."""
  anchor_by_kf: dict[int, list[tuple[LensSetup, Float64Array]]] = {}
  for tag_id, corners_world in anchor_corners_by_tag.items():
    for ch, kf, _det in anchor_sightings(observer, tag_id):
      anchor_by_kf.setdefault(kf, []).append((ch.setup, corners_world))
  cross_by_kf: dict[int, list[tuple[LensSetup, Float64Array]]] = {}
  for s in cross_obs_for_observer:
    owner_corners = base_tag_corners_world.get(s.owner, [])
    if s.finger is None and s.owner_keyframe < len(owner_corners):
      cross_by_kf.setdefault(s.keyframe, []).append(
        (s.setup, owner_corners[s.owner_keyframe])
      )
  poses = observer_poses[: observer.n_keyframes]
  return _rays(poses, anchor_by_kf), _rays(poses, cross_by_kf)


def _per_kf_rejected_keypoints(
  rejected_tracks: LandmarkTracks, n_keyframes: int
) -> list[Float64Array]:
  buckets: list[list[Float64Array]] = [[] for _ in range(n_keyframes)]
  for obs in rejected_tracks.values():
    for kf, pix in obs.items():
      if 0 <= kf < n_keyframes:
        buckets[kf].append(np.asarray(pix, dtype=np.float64))
  return [np.stack(b) if b else np.empty((0, 2)) for b in buckets]


def _rms(values: list[float]) -> float:
  """Root-mean-square over the finite entries; NaN when there are none."""
  finite = [v for v in values if np.isfinite(v)]
  return float(np.sqrt(np.mean(np.square(finite)))) if finite else float("nan")


def _project_tag_axes_per_kf(
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  tag_size_by_id: dict[int, float],
) -> list[Float64Array]:
  """Per keyframe, the pixel segments of each sized tag's frame axes."""
  sightings = [
    (kf, order, det, size)
    for kf, tags in enumerate(tags_per_kf)
    for order, det in enumerate(tags)
    if (size := tag_size_by_id.get(int(det.tag_id))) is not None
  ]
  posed: dict[tuple[int, int], tuple[gtsam.Pose3, float]] = {}
  for size in sorted({size for *_, size in sightings}):
    mine = [s for s in sightings if s[3] == size]
    solved = estimate_lens_T_tags(
      np.stack([det.corners for _, _, det, _ in mine]), size, lens.params
    )
    for (kf, order, _, _), pose in zip(mine, solved, strict=True):
      if pose is not None:
        posed[kf, order] = (pose, size)
  keys = sorted(posed)
  strips: dict[int, list[Float64Array]] = {}
  if keys:
    # The tag frame's origin-to-tip segments along x, y and z.
    lens_tips = np.stack(
      [
        np.stack([np.zeros((3, 3)), size * np.eye(3)], axis=1).reshape(6, 3)
        @ pose.rotation().matrix().T
        + pose.translation()
        for pose, size in (posed[key] for key in keys)
      ]
    )
    pix = mei_project(lens_tips.reshape(-1, 3), lens.params).reshape(-1, 6, 2)
    for (kf, _), xyz, uv in zip(keys, lens_tips, pix, strict=True):
      if np.any(xyz[:, 2] <= 1e-3) or not np.all(np.isfinite(uv)):
        continue
      strips.setdefault(kf, []).append(uv.reshape(3, 2, 2))
  return [
    np.concatenate(strips[kf]) if kf in strips else np.empty((0, 2, 2))
    for kf in range(len(tags_per_kf))
  ]


def _build_camera_estimate(
  camera: PreparedCamera,
  poses: list[gtsam.Pose3],
  result: OptimizeResult,
  keys: CameraKeys,
  lens_keep: LandmarkTracks,
  args: Args,
  tag_size_by_id: dict[int, float],
  cross_obs: list[CrossSighting],
  anchor_corners_by_tag: dict[int, Float64Array],
  base_tag_corners_world: dict[str, list[Float64Array]],
) -> tuple[CameraEstimate, CameraSummary]:
  """One camera's estimate and summary line from the solved values."""
  body_R_rear_lens = result.values.atRot3(keys.seat_key(camera.slot))
  factory_body_R_rear_lens = camera.rear.setup.body_T_lens.rotation()
  body_T_rear_lens = gtsam.Pose3(
    body_R_rear_lens, camera.rear.setup.body_T_lens.translation()
  )
  seat_rotation = gtsam.Rot3.Logmap(factory_body_R_rear_lens.between(body_R_rear_lens))
  print(
    f"  [{camera.slot}/viz] rear seat correction: "
    f"{np.degrees(np.linalg.norm(seat_rotation)):.2f} deg"
  )
  points, reprojections, rmse_per_kf = cull_and_reproject(
    points_from_values(result.values, keys, camera.slot, lens_keep),
    camera.front.tracks,
    poses,
    camera.front.setup,
  )

  def lens_tags(channel: LensChannel) -> LensEstimate:
    """A lens's tag sightings and the axes they pose."""
    return LensEstimate(
      tag_detections=channel.tags,
      tag_axes=_project_tag_axes_per_kf(channel.tags, channel.setup, tag_size_by_id),
    )

  lenses: dict[str, LensEstimate] = {
    "front": replace(
      lens_tags(camera.front),
      points=points,
      keypoints=camera.front.keypoints,
      rejected_keypoints=_per_kf_rejected_keypoints(
        camera.front.rejected, camera.n_keyframes
      ),
      landmark_reprojections=reprojections,
      rmse_per_kf=rmse_per_kf,
    ),
    "rear": lens_tags(camera.rear),
  }
  front_rmse = _rms(rmse_per_kf)
  bbox_diagonal = bounding_box_diagonal(poses)
  seat_anchor_obs, seat_cross_obs = seat_evidence(
    camera, anchor_corners_by_tag, cross_obs
  )
  summary = CameraSummary(
    slot=camera.slot,
    n_keyframes=camera.n_keyframes,
    anchor_obs=len(anchored_keyframes(camera)),
    front_rmse=front_rmse,
    base_tag_id=camera.tags.base,
    seat_rotation=(
      float(seat_rotation[0]),
      float(seat_rotation[1]),
      float(seat_rotation[2]),
    ),
    seat_anchor_obs=seat_anchor_obs,
    seat_cross_obs=seat_cross_obs,
  )
  print(
    f"  [{camera.slot}] kf={camera.n_keyframes} bbox={bbox_diagonal:.3f} m "
    f"front_rmse={front_rmse:.2f}px"
  )
  biases_final = [
    result.values.atConstantBias(keys.bias_key(camera.slot, i))
    for i in range(camera.n_keyframes)
  ]
  velocities_final = [
    np.asarray(result.values.atVector(keys.velocity_key(camera.slot, i)))
    for i in range(camera.n_keyframes)
  ]
  # The propagated series keeps the IMU sample times and never resamples.
  highrate_times, highrate_positions, highrate_quats, highrate_vels = (
    propagate_highrate(
      poses, camera.keyframe_times, velocities_final, biases_final, camera.imu
    )
  )
  joint_frame_angles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
  kf_t = np.asarray(camera.keyframe_times, dtype=np.float64)
  for joint, (f_times, f_angles) in camera.finger_frame_angles.items():
    solved = np.asarray(camera.joint_angles[joint], dtype=np.float64)
    n_j = min(len(kf_t), len(solved))
    pinned = pin_angles_to_keyframes(f_times, f_angles, kf_t[:n_j], solved[:n_j])
    joint_frame_angles[joint] = (f_times, pinned)
  anchor_lines, cross_lines = _factor_observation_lines(
    camera,
    poses,
    cross_obs,
    anchor_corners_by_tag,
    base_tag_corners_world,
  )
  estimate = CameraEstimate(
    slot=camera.slot,
    body_poses=poses,
    keyframe_times=np.asarray(camera.keyframe_times, dtype=np.float64),
    keyframe_frames=np.asarray(camera.keyframe_frames, dtype=np.int64),
    lenses=lenses,
    velocities=velocities_final,
    gyro_biases=[np.asarray(b.gyroscope(), dtype=np.float64) for b in biases_final],
    accel_biases=[
      np.asarray(b.accelerometer(), dtype=np.float64) for b in biases_final
    ],
    anchor_obs_lines=anchor_lines,
    cross_camera_lines=cross_lines,
    tcp_in_base_tag=args.gripper.tcp_in_base_tag,
    highrate_times=highrate_times,
    highrate_positions=highrate_positions,
    highrate_quats=highrate_quats,
    highrate_vels=highrate_vels,
    joint_frame_angles=joint_frame_angles,
    body_T_base_tag=camera.body_T_base_tag,
    base_tag_id=camera.tags.base,
    body_T_rear_lens=body_T_rear_lens,
    joint_tags={
      joint: tag_id
      for joint, tag_id in zip(FINGER_JOINTS, camera.tags.fingers, strict=True)
      if tag_id is not None
    },
  )
  return estimate, summary


def _tag_size_by_id(args: Args, cameras: list[PreparedCamera]) -> dict[int, float]:
  """Side length (m) per tag id, for the anchors and every camera's gripper tags."""
  gripper_ids = {
    tid
    for camera in cameras
    for tid in (camera.tags.base, *camera.tags.fingers)
    if tid is not None
  }
  return {
    **{tid: float(args.anchor.tag_size) for tid in args.anchor.tag_ids},
    **{tid: float(args.gripper.tag_size) for tid in gripper_ids},
  }


def _base_tag_corners_world(
  cameras: list[PreparedCamera],
  optimized_body_poses: dict[str, list[gtsam.Pose3]],
  tag_size: float,
) -> dict[str, list[Float64Array]]:
  """Each camera's base-tag corners in the world, per keyframe."""
  return {
    camera.slot: [
      tag_corners_in_frame(pose.compose(camera.body_T_base_tag).matrix(), tag_size)
      for pose in optimized_body_poses[camera.slot]
    ]
    for camera in cameras
  }


@dataclass(frozen=True)
class RunEstimate:
  """A run's camera estimates, summary lines and anchor poses that hold the gauge."""

  cameras: list[CameraEstimate]
  per_camera: list[CameraSummary]
  world_T_anchor: dict[int, gtsam.Pose3]
  # The anchor whose pose pins the world gauge, one of world_T_anchor.
  reference_tag_id: int


def build_run_estimate(
  cameras: list[PreparedCamera],
  solve: ReferenceSolve,
  keys: CameraKeys,
  args: Args,
  cross_obs: list[CrossSighting],
) -> RunEstimate:
  """Each camera's payload from one solved graph, on the base recording's clock."""
  world_T_anchor = solve.world_T_anchor
  anchor_corners_by_tag = {
    tag_id: tag_corners_in_frame(pose.matrix(), args.anchor.tag_size)
    for tag_id, pose in world_T_anchor.items()
  }
  base_tag_corners = _base_tag_corners_world(
    cameras, solve.body_poses, args.gripper.tag_size
  )
  tag_sizes = _tag_size_by_id(args, cameras)
  by_observer: dict[str, list[CrossSighting]] = {camera.slot: [] for camera in cameras}
  for sighting in cross_obs:
    by_observer[sighting.observer].append(sighting)

  built = [
    _build_camera_estimate(
      camera,
      poses=solve.body_poses[camera.slot],
      result=solve.result,
      keys=keys,
      lens_keep=solve.landmark_keep[camera.slot],
      args=args,
      tag_size_by_id=tag_sizes,
      cross_obs=by_observer[camera.slot],
      anchor_corners_by_tag=anchor_corners_by_tag,
      base_tag_corners_world=base_tag_corners,
    )
    for camera in cameras
  ]
  return RunEstimate(
    [e for e, _ in built],
    [s for _, s in built],
    world_T_anchor,
    solve.ref_tag_id,
  )
