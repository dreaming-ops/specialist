"""Turn one solved run into the payloads that leave the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt
from specialist_camera import mei_project
from specialist_dataset.fiducials import tag_corners_in_frame

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.compose import (
  CameraKeys,
  CrossSighting,
  LandmarkTracks,
  anchor_sightings,
  anchored_keyframes,
  points_from_values,
)
from specialist.vi_ba.configs import Args
from specialist.vi_ba.finger_kinematics import pin_angles_to_keyframes
from specialist.vi_ba.geometry import bounding_box_diagonal
from specialist.vi_ba.highrate import propagate_highrate
from specialist.vi_ba.landmark_cull import cull_and_reproject
from specialist.vi_ba.optimize import OptimizeResult
from specialist.vi_ba.reference_solve import ReferenceSolve
from specialist.vi_ba.single_camera import PreparedCamera
from specialist.vi_ba.summary import CameraSummary
from specialist.vi_ba.tag_detection import TagDetection, estimate_lens_T_tag
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
  max_half_angle: float,
) -> tuple[list[Float64Array], list[Float64Array]]:
  """Per-keyframe ``(n, 2, 3)`` lens-to-corner lines to anchors and base tags."""
  anchor_by_kf: dict[int, list[tuple[LensSetup, Float64Array]]] = {}
  for tag_id, corners_world in anchor_corners_by_tag.items():
    for ch, kf, _det in anchor_sightings(observer, tag_id, max_half_angle):
      anchor_by_kf.setdefault(kf, []).append((ch.setup, corners_world))
  cross_by_kf: dict[int, list[tuple[LensSetup, Float64Array]]] = {}
  for s in cross_obs_for_observer:
    owner_corners = base_tag_corners_world.get(s.owner, [])
    if s.finger is None and s.keyframe < len(owner_corners):
      cross_by_kf.setdefault(s.keyframe, []).append(
        (s.setup, owner_corners[s.keyframe])
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
  out: list[Float64Array] = []
  for tags in tags_per_kf:
    strips: list[Float64Array] = []
    for det in tags:
      size = tag_size_by_id.get(int(det.tag_id))
      if size is None:
        continue
      pose = estimate_lens_T_tag(det.corners, size, lens.params)
      if pose is None:
        continue
      # The tag frame's origin-to-tip segments along x, y and z.
      lens_tips = (
        np.stack([np.zeros((3, 3)), size * np.eye(3)], axis=1).reshape(6, 3)
        @ pose.rotation().matrix().T
        + pose.translation()
      )
      if np.any(lens_tips[:, 2] <= 1e-3):
        continue
      pix = mei_project(lens_tips, lens.params)
      if not np.all(np.isfinite(pix)):
        continue
      strips.append(pix.reshape(3, 2, 2))
    out.append(np.concatenate(strips) if strips else np.empty((0, 2, 2)))
  return out


def _build_camera_estimate(
  camera: PreparedCamera,
  poses: list[gtsam.Pose3],
  result: OptimizeResult,
  keys: CameraKeys,
  lens_keeps: tuple[LandmarkTracks, LandmarkTracks],
  args: Args,
  tag_size_by_id: dict[int, float],
  cross_obs: list[CrossSighting],
  anchor_corners_by_tag: dict[int, Float64Array],
  base_tag_corners_world: dict[str, list[Float64Array]],
) -> tuple[CameraEstimate, CameraSummary]:
  """One camera's estimate and summary line from the solved values."""
  lenses: dict[str, LensEstimate] = {}
  lens_rmse: dict[str, list[float]] = {}
  for channel, landmark_key, keep in (
    (camera.front, keys.front_landmark_key, lens_keeps[0]),
    (camera.rear, keys.rear_landmark_key, lens_keeps[1]),
  ):
    points, reprojections, rmse_per_kf = cull_and_reproject(
      points_from_values(result.values, landmark_key, camera.name, keep),
      channel.tracks,
      poses,
      channel.setup,
    )
    lens_rmse[channel.lens] = rmse_per_kf
    lenses[channel.lens] = LensEstimate(
      points=points,
      keypoints=channel.keypoints,
      rejected_keypoints=_per_kf_rejected_keypoints(
        channel.rejected, camera.n_keyframes
      ),
      landmark_reprojections=reprojections,
      tag_detections=channel.tags,
      tag_axes=_project_tag_axes_per_kf(channel.tags, channel.setup, tag_size_by_id),
      rmse_per_kf=rmse_per_kf,
      static_mask_rects=channel.static_mask_rects,
    )
  front_rmse = _rms(lens_rmse["front"])
  rear_rmse = _rms(lens_rmse["rear"])
  bbox_diagonal = bounding_box_diagonal(poses)
  summary = CameraSummary(
    slot=camera.name,
    n_keyframes=camera.n_keyframes,
    anchor_obs=len(anchored_keyframes(camera)),
    front_rmse=front_rmse,
    rear_rmse=rear_rmse,
    base_tag_id=camera.tags.base,
  )
  rear_note = f" rear_rmse={rear_rmse:.2f}px" if np.isfinite(rear_rmse) else ""
  print(
    f"  [{camera.name}] kf={camera.n_keyframes} bbox={bbox_diagonal:.3f} m "
    f"front_rmse={front_rmse:.2f}px{rear_note}"
  )
  biases_final = [
    result.values.atConstantBias(keys.bias_key(camera.name, i))
    for i in range(camera.n_keyframes)
  ]
  velocities_final = [
    np.asarray(result.values.atVector(keys.velocity_key(camera.name, i)))
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
    args.anchor.max_half_angle,
  )
  estimate = CameraEstimate(
    slot=camera.name,
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
    t.name: [
      tag_corners_in_frame(pose.compose(t.body_T_base_tag).matrix(), tag_size)
      for pose in optimized_body_poses[t.name]
    ]
    for t in cameras
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
  world_T_anchor = {
    solve.ref_tag_id: solve.world_T_ref_anchor,
    **solve.secondary_anchor_world,
  }
  anchor_corners_by_tag = {
    tag_id: tag_corners_in_frame(pose.matrix(), args.anchor.tag_size)
    for tag_id, pose in world_T_anchor.items()
  }
  base_tag_corners = _base_tag_corners_world(
    cameras, solve.body_poses, args.gripper.tag_size
  )
  tag_sizes = _tag_size_by_id(args, cameras)
  by_observer: dict[str, list[CrossSighting]] = {t.name: [] for t in cameras}
  for sighting in cross_obs:
    by_observer[sighting.observer].append(sighting)

  built = [
    _build_camera_estimate(
      t,
      poses=solve.body_poses[t.name],
      result=solve.result,
      keys=keys,
      lens_keeps=solve.landmark_keep[t.name],
      args=args,
      tag_size_by_id=tag_sizes,
      cross_obs=by_observer[t.name],
      anchor_corners_by_tag=anchor_corners_by_tag,
      base_tag_corners_world=base_tag_corners,
    )
    for t in cameras
  ]
  return RunEstimate(
    [e for e, _ in built],
    [s for _, s in built],
    world_T_anchor,
    solve.ref_tag_id,
  )
