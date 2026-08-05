"""Columnar Rerun logging of a solved run into the caller's active recording."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import gtsam
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb

from specialist.record import schema
from specialist.vi_ba.geometry import matrices_to_quats, quat_xyzw, quats_to_matrices

if TYPE_CHECKING:
  from specialist.vi_ba.tag_detection import TagDetection

Float64Array = npt.NDArray[np.float64]

_MEASURED = (255, 0, 255)
_REJECTED = (230, 60, 60)
_PREDICTED = (80, 200, 120)
_ANCHOR = (255, 210, 0)
_GRIPPER_TAG = (90, 220, 220)
# By camera rank; vermilion and blue lead: they stay apart under red-green blindness.
_SLOT_COLORS = (
  (236, 56, 12),
  (24, 132, 180),
  (208, 255, 196),
  (240, 152, 36),
  (132, 128, 60),
)
_TRAJECTORY_TRAIL_DURATION = 1.0
_BODY_AXIS_LENGTH = 0.05

__all__ = ["CameraVizData", "log_multi_ba_entities", "run_blueprint"]


def _vec3_column(values: Float64Array) -> pa.FixedSizeListArray:
  """An ``(n, 3)`` float array as the arrow column AnyValues can carry."""
  flat = pa.array(np.ascontiguousarray(values, dtype=np.float64).reshape(-1))
  return pa.FixedSizeListArray.from_arrays(flat, 3)


def _pose_transform3d(pose: gtsam.Pose3) -> rr.Transform3D:
  return rr.Transform3D(
    translation=np.asarray(pose.translation(), dtype=np.float64),
    quaternion=rr.datatypes.Quaternion(xyzw=quat_xyzw(pose.rotation())),
  )


def _trail_strips(
  traj: Float64Array, times: Float64Array, duration: float
) -> list[Float64Array]:
  """Per-keyframe position strips in ``[t - duration, t]``; ``times`` must ascend."""
  starts = np.searchsorted(times, times - duration, side="left")
  return [traj[start : i + 1] for i, start in enumerate(starts)]


def _log_static_points(
  entity_path: str, points: dict[int, Float64Array], color: tuple[int, int, int]
) -> None:
  if not points:
    return
  xyz = np.asarray(list(points.values()), dtype=np.float64)
  rr.log(entity_path, rr.Points3D(xyz, radii=0.002, colors=color), static=True)


def _solid_colors(color: tuple[int, int, int], n: int) -> np.ndarray:
  """An ``(n, 3)`` uint8 array of one repeated RGB color."""
  return np.tile(np.asarray(color, dtype=np.uint8), (n, 1))


@dataclass(frozen=True)
class _OverlayIndex:
  """The timeline index for per-keyframe overlay batches, or a sequence fallback."""

  times: Float64Array | None  # None = sequence fallback
  n: int

  def resolve(self) -> rr.TimeColumn:
    if self.times is None:
      return rr.TimeColumn(schema.TIMELINE, sequence=np.arange(self.n, dtype=np.int64))
    return rr.TimeColumn(schema.TIMELINE, duration=self.times)


def _send_lens_tag_columns(
  entity_prefix: str,
  tags_per_kf: list[list[TagDetection]],
  index: _OverlayIndex,
  anchor_tag_ids: set[int],
  mask_pad_px: int,
) -> None:
  """AprilTag quads and ids per keyframe; negative ``mask_pad_px`` draws no mask box."""
  tags = [det for dets in tags_per_kf for det in dets]
  if not tags:
    return
  partition = [len(dets) for dets in tags_per_kf]
  ids = np.asarray([int(det.tag_id) for det in tags], dtype=np.int32)
  is_anchor = np.isin(ids, list(anchor_tag_ids))
  colors = np.where(is_anchor[:, None], _ANCHOR, _GRIPPER_TAG).astype(np.uint8)
  strips = [np.vstack([d.corners, d.corners[:1]]).astype(np.float32) for d in tags]
  time_col = index.resolve()
  rr.send_columns(
    f"{entity_prefix}/tags",
    indexes=[time_col],
    columns=rr.LineStrips2D.columns(
      strips=strips,
      colors=colors,
      radii=np.full(len(tags), 5.0, dtype=np.float32),
    ).partition(partition),
  )
  rr.send_columns(
    f"{entity_prefix}/tags/labels",
    indexes=[time_col],
    columns=rr.Points2D.columns(
      positions=np.asarray([d.center for d in tags], dtype=np.float32),
      labels=[str(i) for i in ids],
      show_labels=[True] * len(tags),
      colors=colors,
      radii=np.full(len(tags), 10.0, dtype=np.float32),
    ).partition(partition),
  )
  keep = ~is_anchor & (mask_pad_px >= 0)
  masked = [d.corners for d, kept in zip(tags, keep, strict=True) if kept]
  if not masked:
    return
  lo = np.stack([c.min(axis=0) for c in masked]) - float(mask_pad_px)
  hi = np.stack([c.max(axis=0) for c in masked]) + float(mask_pad_px)
  mask_col = index.resolve()
  rr.send_columns(
    f"{entity_prefix}/tags/masked",
    indexes=[mask_col],
    columns=rr.Boxes2D.columns(
      centers=((lo + hi) * 0.5).astype(np.float32),
      half_sizes=((hi - lo) * 0.5).astype(np.float32),
      colors=_solid_colors(_REJECTED, len(masked)),
      labels=["MASKED"] * len(masked),
      show_labels=[True] * len(masked),
    ).partition([int(k.sum()) for k in np.split(keep, np.cumsum(partition)[:-1])]),
  )


def _log_static_mask_rects(
  entity_prefix: str, rects: list[tuple[int, int, int, int]]
) -> None:
  if not rects:
    return
  centers: list[np.ndarray] = []
  half_sizes: list[np.ndarray] = []
  for x1, y1, x2, y2 in rects:
    centers.append(np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32))
    half_sizes.append(np.array([(x2 - x1) * 0.5, (y2 - y1) * 0.5], dtype=np.float32))
  rr.log(
    f"{entity_prefix}/static_mask",
    rr.Boxes2D(
      centers=np.stack(centers),
      half_sizes=np.stack(half_sizes),
      colors=_solid_colors(_REJECTED, len(rects)),
      labels=["STATIC"] * len(rects),
      show_labels=True,
    ),
    static=True,
  )


def _send_point_columns(
  entity_path: str,
  points_per_kf: list[Float64Array],
  index: _OverlayIndex,
  radius: float,
  color: tuple[int, int, int],
) -> None:
  partition = [int(kp.shape[0]) for kp in points_per_kf]
  total = sum(partition)
  if total == 0:
    return
  positions = np.concatenate([kp for kp in points_per_kf if kp.shape[0] > 0])
  time_col = index.resolve()
  rr.send_columns(
    entity_path,
    indexes=[time_col],
    columns=rr.Points2D.columns(
      positions=positions,
      radii=np.full(total, radius, dtype=np.float32),
      colors=_solid_colors(color, total),
    ).partition(partition),
  )


def _send_tag_axes_columns(
  entity_prefix: str,
  axes_per_kf: list[Float64Array],
  index: _OverlayIndex,
) -> None:
  partition = [len(a) for a in axes_per_kf]
  total = sum(partition)
  if total == 0:
    return
  time_col = index.resolve()
  rgb = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255)], dtype=np.uint8)
  rr.send_columns(
    f"{entity_prefix}/tags/axes",
    indexes=[time_col],
    columns=rr.LineStrips2D.columns(
      strips=np.concatenate(axes_per_kf).astype(np.float32),
      colors=rgb[np.arange(total) % 3],
      radii=np.full(total, 4.0, dtype=np.float32),
    ).partition(partition),
  )


def _send_factor_line_columns(
  entity_path: str,
  lines_per_kf: list[Float64Array],
  index: _OverlayIndex,
  color: tuple[int, int, int],
) -> None:
  """Per-keyframe ``(n_lines, 2, 3)`` endpoint arrays as LineStrips3D rows."""
  partition = [int(arr.shape[0]) for arr in lines_per_kf]
  total = sum(partition)
  if total == 0:
    return
  strips = [line.astype(np.float32) for arr in lines_per_kf for line in arr]
  time_col = index.resolve()
  rr.send_columns(
    entity_path,
    indexes=[time_col],
    columns=rr.LineStrips3D.columns(
      strips=strips,
      colors=_solid_colors(color, total),
      radii=np.full(total, 0.0008, dtype=np.float32),
    ).partition(partition),
  )


def _send_lens_columns(
  entity_path: str,
  metrics_path: str,
  keypoints: list[Float64Array] | None,
  index: _OverlayIndex,
  times: rr.TimeColumn,
  color: tuple[int, int, int],
) -> None:
  if not keypoints:
    return
  _send_point_columns(f"{entity_path}/keypoints", keypoints, index, 1.5, color)
  counts = np.asarray([kp.shape[0] for kp in keypoints], dtype=np.float64)
  rr.send_columns(
    metrics_path,
    indexes=[times],
    columns=rr.Scalars.columns(scalars=counts),
  )


@dataclass
class CameraVizData:
  """One tracker's Rerun payload from ``build_run_outputs`` for ``log_multi_ba_run``."""

  name: str
  body_poses: list[gtsam.Pose3] = field(default_factory=list)
  keyframe_times: Float64Array | None = None
  front_points: dict[int, Float64Array] = field(default_factory=dict)
  rear_points: dict[int, Float64Array] = field(default_factory=dict)
  front_keypoints: list[Float64Array] | None = None
  rear_keypoints: list[Float64Array] | None = None
  velocities: list[Float64Array] | None = None
  gyro_biases: list[Float64Array] | None = None
  accel_biases: list[Float64Array] | None = None
  front_rmse_per_kf: list[float] | None = None
  rear_rmse_per_kf: list[float] | None = None
  front_tag_detections: list[list[TagDetection]] | None = None
  rear_tag_detections: list[list[TagDetection]] | None = None
  front_rejected_keypoints: list[Float64Array] | None = None
  rear_rejected_keypoints: list[Float64Array] | None = None
  rear_static_mask_rects: list[tuple[int, int, int, int]] | None = None
  front_landmark_reprojections: list[Float64Array] | None = None
  rear_landmark_reprojections: list[Float64Array] | None = None
  front_tag_axes: list[Float64Array] | None = None
  rear_tag_axes: list[Float64Array] | None = None
  urdf_root_poses: list[gtsam.Pose3] = field(default_factory=list)
  # URDF joint name → per-keyframe angle, NaN where it could not be recovered.
  joint_angles: dict[str, list[float]] | None = None
  tcp_in_base_tag: tuple[float, float, float] | None = None
  front_lens_extrinsic: gtsam.Pose3 | None = None
  rear_lens_extrinsic: gtsam.Pose3 | None = None
  # Per-keyframe ``(n, 2, 3)`` lens-to-corner rays, one per firing reprojection factor.
  anchor_obs_lines: list[Float64Array] | None = None
  cross_tracker_lines: list[Float64Array] | None = None
  # IMU-propagated body series at the IMU sample times; times on the shared clock.
  highrate_times: Float64Array | None = None
  highrate_positions: Float64Array | None = None
  highrate_quats: Float64Array | None = None
  highrate_vels: Float64Array | None = None
  # Per-joint pinned finger angles at the frame rate, as (times, angles).
  joint_frame_angles: dict[str, tuple[Float64Array, Float64Array]] | None = None
  # The solved body -> base-tag transform (a BA variable), logged statically.
  body_T_base_tag: gtsam.Pose3 | None = None
  # MEI intrinsics at decode resolution per lens, logged as lens statics.
  front_lens_mei: dict[str, float] | None = None
  rear_lens_mei: dict[str, float] | None = None


def _gripper_points(data: CameraVizData) -> Float64Array:
  """Per-keyframe gripper keypoints given a keypoint offset, else raw body positions."""
  if data.tcp_in_base_tag is not None:
    offset = np.asarray(data.tcp_in_base_tag, dtype=np.float64)
    return np.stack(
      [
        np.asarray(p.translation(), dtype=np.float64)
        + np.asarray(p.rotation().matrix(), dtype=np.float64) @ offset
        for p in data.urdf_root_poses
      ]
    )
  return np.asarray(
    [np.asarray(p.translation(), dtype=np.float64) for p in data.body_poses]
  )


def _log_anchor(entity: str, world_T_anchor: gtsam.Pose3, tag_size: float) -> None:
  half = tag_size / 2.0
  square = np.array(
    [
      [+half, -half, 0.0],
      [+half, +half, 0.0],
      [-half, +half, 0.0],
      [-half, -half, 0.0],
      [+half, -half, 0.0],
    ],
    dtype=np.float64,
  )
  rr.log(
    f"world/{entity}",
    _pose_transform3d(world_T_anchor),
    rr.LineStrips3D([square], colors=_ANCHOR, radii=0.003),
    static=True,
  )
  rr.log(
    f"world/{entity}/frame",
    rr.TransformAxes3D(axis_length=half),
    static=True,
  )


def _log_tracker(
  data: CameraVizData,
  spatial_root: str,
  image_root: str,
  metrics_suffix: str,
  frame_prefix: str | None,
  urdf_path: Path | None,
  anchor_tag_ids: set[int],
  mask_pad_px: int,
  color: tuple[int, int, int],
) -> None:
  """Log every per-tracker entity; the caller logs the shared statics."""
  pose_translations = np.asarray(
    [np.asarray(p.translation(), dtype=np.float64) for p in data.body_poses]
  )
  traj = _gripper_points(data)

  _log_static_points(f"{spatial_root}/landmarks/front", data.front_points, color)
  _log_static_points(f"{spatial_root}/landmarks/rear", data.rear_points, color)
  rr.log(
    f"{spatial_root}/body",
    rr.TransformAxes3D(axis_length=_BODY_AXIS_LENGTH),
    static=True,
  )
  rr.log(
    f"{spatial_root}/body/marker",
    rr.Points3D(np.zeros((1, 3)), radii=0.002, colors=color),
    static=True,
  )
  for lens_name, lens_pose, lens_mei in (
    ("front", data.front_lens_extrinsic, data.front_lens_mei),
    ("rear", data.rear_lens_extrinsic, data.rear_lens_mei),
  ):
    if lens_pose is None:
      continue
    rr.log(
      f"{spatial_root}/body/{lens_name}_lens",
      _pose_transform3d(lens_pose),
      static=True,
    )
    rr.log(
      f"{spatial_root}/body/{lens_name}_lens",
      rr.TransformAxes3D(axis_length=_BODY_AXIS_LENGTH),
      static=True,
    )
    if lens_mei is not None:
      rr.log(
        f"{spatial_root}/body/{lens_name}_lens",
        rr.AnyValues(**cast("dict[str, Any]", lens_mei)),
        static=True,
      )
  if data.body_T_base_tag is not None:
    rr.log(
      schema.base_tag_entity(data.name),
      _pose_transform3d(data.body_T_base_tag),
      rr.TransformAxes3D(axis_length=0.02),
      static=True,
    )

  urdf_root_frame: str | None = None
  tree: rr.urdf.UrdfTree | None = None
  if urdf_path is not None:
    try:
      tree = rr.urdf.UrdfTree.from_file_path(
        str(urdf_path), entity_path_prefix=spatial_root, frame_prefix=frame_prefix
      )
      tree.log_urdf_to_recording()
      for link in [tree.root_link(), *map(tree.get_joint_child, tree.joints())]:
        for p in tree.get_collision_geometry_paths(link):
          rr.log(p, rr.Clear(recursive=True), static=True)
      urdf_root_frame = (
        f"{frame_prefix}{tree.root_link().name}"
        if frame_prefix
        else tree.root_link().name
      )
      rr.log(
        f"{spatial_root}/gripper",
        rr.Transform3D(parent_frame="world", child_frame=urdf_root_frame),
        static=True,
      )
      rr.log(
        f"{spatial_root}/gripper",
        rr.TransformAxes3D(axis_length=_BODY_AXIS_LENGTH),
        static=True,
      )
    except Exception as exc:  # noqa: BLE001 - the URDF model is optional; keep the rest
      traceback.print_exc()
      print(f"  [viz] URDF model skipped ({exc}); logging the trajectory without it")
      tree = None
      urdf_root_frame = None

  n = len(data.body_poses)
  if data.keyframe_times is not None:
    trail_times = np.asarray(data.keyframe_times, dtype=np.float64)
    times = rr.TimeColumn(schema.TIMELINE, duration=trail_times)
    overlay_index = _OverlayIndex(times=trail_times, n=n)
  else:
    trail_times = np.arange(n, dtype=np.float64)
    times = rr.TimeColumn(schema.TIMELINE, sequence=np.arange(n, dtype=np.int64))
    overlay_index = _OverlayIndex(times=None, n=n)

  hr: tuple[Float64Array, Float64Array, Float64Array] | None = None
  if (
    data.highrate_times is not None
    and data.highrate_positions is not None
    and data.highrate_quats is not None
  ):
    hr = (data.highrate_times, data.highrate_positions, data.highrate_quats)
  if hr is not None:
    hr_times, hr_positions, hr_quats = hr
    rr.send_columns(
      f"{spatial_root}/body",
      indexes=[rr.TimeColumn(schema.TIMELINE, duration=hr_times)],
      columns=rr.Transform3D.columns(translation=hr_positions, quaternion=hr_quats),
    )
  else:
    quats = np.stack([quat_xyzw(p.rotation()) for p in data.body_poses])
    rr.send_columns(
      f"{spatial_root}/body",
      indexes=[times],
      columns=rr.Transform3D.columns(translation=pose_translations, quaternion=quats),
    )
  # The exact keyframe poses, which the scorers align and regression diffs compare.
  kf_quats = np.stack([quat_xyzw(p.rotation()) for p in data.body_poses])
  rr.send_columns(
    schema.body_keyframes_entity(data.name),
    indexes=[times],
    columns=rr.Transform3D.columns(translation=pose_translations, quaternion=kf_quats),
  )
  if hr is not None and data.highrate_vels is not None:
    rr.send_columns(
      f"{spatial_root}/body",
      indexes=[rr.TimeColumn(schema.TIMELINE, duration=hr[0])],
      columns=rr.AnyValues.columns(velocity=_vec3_column(data.highrate_vels)),
    )
  elif data.velocities is not None:
    rr.send_columns(
      f"{spatial_root}/body",
      indexes=[times],
      columns=rr.AnyValues.columns(velocity=_vec3_column(np.stack(data.velocities))),
    )
  world_T_base_tag: tuple[Float64Array, Float64Array] | None = None
  if urdf_root_frame is not None:
    poses = data.urdf_root_poses
    if hr is not None and data.body_poses and poses:
      hr_times, hr_positions, hr_quats = hr
      body_T_base_tag = data.body_poses[0].inverse().compose(poses[0])
      body_R_base_tag = np.asarray(body_T_base_tag.rotation().matrix())
      body_t_base_tag = np.asarray(body_T_base_tag.translation())
      world_R_body = quats_to_matrices(hr_quats)
      world_t_base_tag = hr_positions + world_R_body @ body_t_base_tag
      world_R_base_tag = world_R_body @ body_R_base_tag
      world_T_base_tag = (world_t_base_tag, world_R_base_tag)
      count = len(world_t_base_tag)
      rr.send_columns(
        f"{spatial_root}/gripper",
        indexes=[rr.TimeColumn(schema.TIMELINE, duration=hr_times)],
        columns=rr.Transform3D.columns(
          translation=world_t_base_tag,
          quaternion=matrices_to_quats(world_R_base_tag),
          parent_frame=["world"] * count,
          child_frame=[urdf_root_frame] * count,
        ),
      )
    else:
      urdf_traj = np.asarray(
        [np.asarray(p.translation(), dtype=np.float64) for p in poses]
      )
      urdf_quats = np.stack([quat_xyzw(p.rotation()) for p in poses])
      rr.send_columns(
        f"{spatial_root}/gripper",
        indexes=[times],
        columns=rr.Transform3D.columns(
          translation=urdf_traj,
          quaternion=urdf_quats,
          parent_frame=["world"] * len(poses),
          child_frame=[urdf_root_frame] * len(poses),
        ),
      )
    if data.joint_angles is not None:
      assert tree is not None  # reached only on the path that set urdf_root_frame
      for joint_name, raw_values in data.joint_angles.items():
        joint = tree.get_joint_by_name(joint_name)
        if joint is None:
          continue
        frame_series = (data.joint_frame_angles or {}).get(joint_name)
        if frame_series is not None:
          f_times, f_angles = frame_series
          rr.send_columns(
            f"{spatial_root}/{joint_name}",
            indexes=[rr.TimeColumn(schema.TIMELINE, duration=f_times)],
            columns=joint.compute_transform_columns([float(a) for a in f_angles]),
          )
          rr.send_columns(
            f"{spatial_root}/{joint_name}",
            indexes=[rr.TimeColumn(schema.TIMELINE, duration=f_times)],
            columns=rr.AnyValues.columns(angle=np.asarray(f_angles, dtype=np.float64)),
          )
          continue
        first_finite = next((float(v) for v in raw_values if np.isfinite(v)), None)
        if first_finite is None:
          continue
        # Hold each finite angle through dropouts, seeded from the first finite value.
        clean: list[float] = []
        last = first_finite
        for v in raw_values:
          if np.isfinite(v):
            last = float(v)
          clean.append(last)
        rr.send_columns(
          f"{spatial_root}/{joint_name}",
          indexes=[times],
          columns=joint.compute_transform_columns(clean),
        )
        rr.send_columns(
          f"{spatial_root}/{joint_name}",
          indexes=[times],
          columns=rr.AnyValues.columns(angle=np.asarray(raw_values, dtype=np.float64)),
        )
  # Trajectory: the propagated series, or the keyframe polyline without an IMU.
  if hr is not None:
    hr_times, hr_positions, _ = hr
    if world_T_base_tag is not None and data.tcp_in_base_tag is not None:
      offset = np.asarray(data.tcp_in_base_tag, dtype=np.float64)
      world_t_base_tag, world_R_base_tag = world_T_base_tag
      traj_points = world_t_base_tag + world_R_base_tag @ offset
    else:
      traj_points = hr_positions
    traj_times = rr.TimeColumn(schema.TIMELINE, duration=hr_times)
    traj_trail_times = hr_times
  else:
    traj_points = traj
    traj_times = times
    traj_trail_times = trail_times
  rr.send_columns(
    f"{spatial_root}/trajectory",
    indexes=[traj_times],
    columns=rr.LineStrips3D.columns(
      strips=_trail_strips(traj_points, traj_trail_times, _TRAJECTORY_TRAIL_DURATION),
      radii=np.full(len(traj_points), 0.002, dtype=np.float32),
      colors=_solid_colors(color, len(traj_points)),
    ),
  )
  rr.send_columns(
    f"metrics/distance_from_start{metrics_suffix}",
    indexes=[traj_times],
    columns=rr.Scalars.columns(
      scalars=np.linalg.norm(traj_points - traj_points[0:1], axis=1)
    ),
  )

  if data.anchor_obs_lines is not None:
    _send_factor_line_columns(
      f"{spatial_root}/factor_lines/anchor",
      data.anchor_obs_lines,
      overlay_index,
      _ANCHOR,
    )
  if data.cross_tracker_lines is not None:
    _send_factor_line_columns(
      f"{spatial_root}/factor_lines/cross_tracker",
      data.cross_tracker_lines,
      overlay_index,
      _GRIPPER_TAG,
    )

  for lens_name, kps in (
    ("front", data.front_keypoints),
    ("rear", data.rear_keypoints),
  ):
    _send_lens_columns(
      f"{image_root}/{lens_name}",
      f"metrics/n_keypoints/{lens_name}{metrics_suffix}",
      kps,
      overlay_index,
      times,
      _MEASURED,
    )
  for lens_name, tags in (
    ("front", data.front_tag_detections),
    ("rear", data.rear_tag_detections),
  ):
    if tags is None:
      continue
    _send_lens_tag_columns(
      f"{image_root}/{lens_name}", tags, overlay_index, anchor_tag_ids, mask_pad_px
    )
  for lens_name, rejected in (
    ("front", data.front_rejected_keypoints),
    ("rear", data.rear_rejected_keypoints),
  ):
    if rejected is None:
      continue
    _send_point_columns(
      f"{image_root}/{lens_name}/keypoints_rejected",
      rejected,
      overlay_index,
      1.5,
      _REJECTED,
    )
  for lens_name, reproj in (
    ("front", data.front_landmark_reprojections),
    ("rear", data.rear_landmark_reprojections),
  ):
    if reproj is None:
      continue
    _send_point_columns(
      f"{image_root}/{lens_name}/landmark_reprojection",
      reproj,
      overlay_index,
      3.0,
      _PREDICTED,
    )
  for lens_name, axes in (
    ("front", data.front_tag_axes),
    ("rear", data.rear_tag_axes),
  ):
    if axes is None:
      continue
    _send_tag_axes_columns(f"{image_root}/{lens_name}", axes, overlay_index)
  if data.rear_static_mask_rects:
    _log_static_mask_rects(f"{image_root}/rear", data.rear_static_mask_rects)
  velocity: tuple[rr.TimeColumn, Float64Array] | None = None
  if hr is not None and data.highrate_vels is not None:
    velocity = (rr.TimeColumn(schema.TIMELINE, duration=hr[0]), data.highrate_vels)
  elif data.velocities is not None:
    velocity = (times, np.stack(data.velocities))
  if velocity is not None:
    velocity_times, velocity_values = velocity
    rr.send_columns(
      f"metrics/speed/body{metrics_suffix}",
      indexes=[velocity_times],
      columns=rr.Scalars.columns(scalars=np.linalg.norm(velocity_values, axis=1)),
    )
  for metric_name, biases in (
    ("gyro_bias", data.gyro_biases),
    ("accel_bias", data.accel_biases),
  ):
    if biases is None:
      continue
    for axis, axis_name in enumerate("xyz"):
      rr.send_columns(
        f"metrics/{metric_name}/{axis_name}{metrics_suffix}",
        indexes=[times],
        columns=rr.Scalars.columns(
          scalars=np.asarray([float(b[axis]) for b in biases], dtype=np.float64)
        ),
      )
  for lens_name, rmse in (
    ("front", data.front_rmse_per_kf),
    ("rear", data.rear_rmse_per_kf),
  ):
    if rmse is None:
      continue
    rr.send_columns(
      f"metrics/rmse/{lens_name}{metrics_suffix}",
      indexes=[times],
      columns=rr.Scalars.columns(scalars=np.asarray(rmse, dtype=np.float64)),
    )


def blueprint_for_names(
  slots: list[str], floor_z: float | None = None
) -> rrb.Blueprint:
  """Multi-tracker blueprint: a 3D view beside one image-pair column per tracker."""
  line_grid = (
    rrb.LineGrid3D(
      plane=rr.datatypes.Plane3D(normal=(0.0, 0.0, 1.0), distance=float(floor_z))
    )
    if floor_z is not None
    else None
  )
  image_columns = [
    rrb.Vertical(
      rrb.Spatial2DView(name=f"{n}/front", origin=f"/image/{n}/front"),
      rrb.Spatial2DView(name=f"{n}/rear", origin=f"/image/{n}/rear"),
      name=n,
    )
    for n in slots
  ]
  return rrb.Blueprint(
    rrb.Horizontal(
      rrb.Spatial3DView(
        name="3D",
        origin="/",
        contents=["+ /**", "- /image/**"],
        eye_controls=rrb.archetypes.EyeControls3D(
          kind=rrb.components.Eye3DKind.Orbital,
          tracking_entity=f"/{slots[0]}/body/marker",
          position=(0.8, 0.8, 0.8),
          look_target=(0.0, 0.0, 0.0),
        ),
        line_grid=line_grid,
      ),
      rrb.Horizontal(*image_columns) if len(image_columns) > 1 else image_columns[0],
      column_shares=[2, max(1, len(slots))],
    ),
    rrb.BlueprintPanel(state=rrb.PanelState.Collapsed),
    rrb.SelectionPanel(state=rrb.PanelState.Collapsed),
    rrb.TimePanel(state=rrb.PanelState.Collapsed),
  )


def run_blueprint(cameras: list[CameraVizData]) -> rrb.Blueprint:
  """A solved run's dashboard, its floor grid at the lowest trajectory point."""
  gripper_points = [_gripper_points(t) for t in cameras]
  floor_zs = [float(pts[:, 2].min()) for pts in gripper_points if len(pts)]
  floor_z = min(floor_zs) if floor_zs else None
  return blueprint_for_names([t.name for t in cameras], floor_z)


def log_multi_ba_entities(
  cameras: list[CameraVizData],
  world_T_anchor: dict[int, gtsam.Pose3],
  reference_tag_id: int,
  anchor_tag_size: float,
  mask_pad_px: int = 0,
  urdf_path: Path | None = None,
) -> None:
  """Log every entity of a solved run into the caller's active recording."""
  rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
  rr.log("world", rr.Transform3D(child_frame="world"), static=True)
  for tag_id, pose in world_T_anchor.items():
    entity = "anchor_tag" if tag_id == reference_tag_id else f"anchor_tag_{tag_id}"
    _log_anchor(entity, pose, anchor_tag_size)
  for i, t in enumerate(cameras):
    _log_tracker(
      t,
      spatial_root=t.name,
      image_root=schema.image_root(t.name),
      metrics_suffix=f"/{t.name}",
      frame_prefix=f"{t.name}_",
      urdf_path=urdf_path,
      anchor_tag_ids=set(world_T_anchor),
      mask_pad_px=mask_pad_px,
      color=_SLOT_COLORS[i % len(_SLOT_COLORS)],
    )
