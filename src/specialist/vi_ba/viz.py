"""Rerun chunks of a solved run, each handed to the sink of the layer it belongs in."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gtsam
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from rerun.experimental import Chunk

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

__all__ = [
  "LensVizData",
  "CameraVizData",
  "blueprint_for_names",
  "solution_chunks",
  "run_blueprint",
]


Sink = Callable[[Chunk], None]


def _send(emit: Sink, entity_path: str, indexes: Any, columns: Any) -> None:
  """One temporal chunk, shaped exactly as the columnar log call was."""
  emit(Chunk.from_columns(entity_path, indexes=indexes, columns=columns))


def _static(emit: Sink, entity_path: str, *columns: Any) -> None:
  """One static chunk carrying every archetype logged at that entity."""
  emit(
    Chunk.from_columns(
      entity_path, indexes=[], columns=[c for group in columns for c in group]
    )
  )


def _pose_columns(pose: gtsam.Pose3) -> Any:
  """A single-row Transform3D column list for one pose."""
  return rr.Transform3D.columns(
    translation=np.asarray(pose.translation(), dtype=np.float64)[None, :],
    quaternion=quat_xyzw(pose.rotation())[None, :],
  )


def _frame_columns(
  translations: Float64Array, quats: Float64Array, child_frame: str
) -> Any:
  """A world-parented Transform3D column list, one row per sample."""
  return rr.Transform3D.columns(
    translation=translations,
    quaternion=quats,
    parent_frame=["world"] * len(translations),
    child_frame=[child_frame] * len(translations),
  )


def _vec3_column(values: Float64Array) -> pa.FixedSizeListArray:
  """An ``(n, 3)`` float array as the arrow column AnyValues can carry."""
  flat = pa.array(np.ascontiguousarray(values, dtype=np.float64).reshape(-1))
  return pa.FixedSizeListArray.from_arrays(flat, 3)


def _trail_segments(traj: Float64Array, times: Float64Array, duration: float) -> Any:
  """Segments from each sample's predecessor, zero-length past a ``duration`` gap."""
  previous = np.concatenate([traj[:1], traj[:-1]])
  outran = np.concatenate(([0.0], np.diff(times))) > duration
  previous[outran] = traj[outran]
  return np.stack([previous, traj], axis=1)


def _send_static_points(
  emit: Sink,
  entity_path: str,
  points: dict[int, Float64Array],
  color: tuple[int, int, int],
) -> None:
  if not points:
    return
  xyz = np.asarray(list(points.values()), dtype=np.float64)
  _static(
    emit,
    entity_path,
    rr.Points3D.columns(positions=xyz).partition([len(xyz)]),
    rr.Points3D.columns(radii=[0.002], colors=[color]),
  )


def _solid_colors(color: tuple[int, int, int], n: int) -> np.ndarray:
  """An ``(n, 3)`` uint8 array of one repeated RGB color."""
  return np.tile(np.asarray(color, dtype=np.uint8), (n, 1))


# Overlay data-row and clearing-row indexes; clearing rows never carry ``keyframe``.
_OverlayIndex = tuple[list[rr.TimeColumn], list[rr.TimeColumn] | None]


def _overlay_index(
  trail_times: Float64Array | None,
  frame_period: float | None,
  fallback: list[rr.TimeColumn],
  keyframe_column: rr.TimeColumn | None,
) -> _OverlayIndex:
  """Indexes bounding each overlay to its video frame; ``fallback`` without a period."""
  if trail_times is None or frame_period is None:
    return fallback, None
  data = [rr.TimeColumn(schema.TIMELINE, duration=trail_times)]
  if keyframe_column is not None:
    data.append(keyframe_column)
  return data, [rr.TimeColumn(schema.TIMELINE, duration=trail_times + frame_period)]


def _cleared(archetype: Any, columns: dict[str, Any], n: int) -> Any:
  """The overlay's archetype with every column emptied, one blank row per keyframe."""
  return archetype.columns(**dict.fromkeys(columns, [])).partition([0] * n)


def _send_overlay(
  emit: Sink,
  entity_path: str,
  index: _OverlayIndex,
  archetype: Any,
  partition: list[int],
  **columns: Any,
) -> None:
  """An image overlay's keyframe rows, cleared by empty rows one frame period later."""
  data_indexes, clear_indexes = index
  rows = archetype.columns(**columns).partition(partition)
  _send(emit, entity_path, data_indexes, rows)
  if clear_indexes is not None:
    n = len(partition)
    _send(emit, entity_path, clear_indexes, _cleared(archetype, columns, n))


def _send_lens_tag_columns(
  emit: Sink,
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
  data_indexes, clear_indexes = index
  n = len(partition)
  tags_path = f"{entity_prefix}/tags"
  labels_path = f"{tags_path}/labels"
  tag_columns: dict[str, Any] = {
    "strips": strips,
    "colors": colors,
    "radii": np.full(len(tags), 5.0, dtype=np.float32),
  }
  label_columns: dict[str, Any] = {
    "positions": np.asarray([d.center for d in tags], dtype=np.float32),
    "labels": [str(i) for i in ids],
    "show_labels": [True] * len(tags),
    "colors": colors,
    "radii": np.full(len(tags), 10.0, dtype=np.float32),
  }
  tag_rows = rr.LineStrips2D.columns(**tag_columns).partition(partition)
  label_rows = rr.Points2D.columns(**label_columns).partition(partition)
  _send(emit, tags_path, data_indexes, tag_rows)
  _send(emit, labels_path, data_indexes, label_rows)
  if clear_indexes is not None:
    _send(emit, tags_path, clear_indexes, _cleared(rr.LineStrips2D, tag_columns, n))
    _send(emit, labels_path, clear_indexes, _cleared(rr.Points2D, label_columns, n))
  keep = ~is_anchor & (mask_pad_px >= 0)
  masked = [d.corners for d, kept in zip(tags, keep, strict=True) if kept]
  if not masked:
    return
  lo = np.stack([c.min(axis=0) for c in masked]) - float(mask_pad_px)
  hi = np.stack([c.max(axis=0) for c in masked]) + float(mask_pad_px)
  _send_overlay(
    emit,
    f"{tags_path}/masked",
    index,
    rr.Boxes2D,
    [int(k.sum()) for k in np.split(keep, np.cumsum(partition)[:-1])],
    centers=((lo + hi) * 0.5).astype(np.float32),
    half_sizes=((hi - lo) * 0.5).astype(np.float32),
    colors=_solid_colors(_REJECTED, len(masked)),
    labels=["MASKED"] * len(masked),
    show_labels=[True] * len(masked),
  )


def _log_static_mask_rects(
  emit: Sink, entity_prefix: str, rects: list[tuple[int, int, int, int]]
) -> None:
  centers: list[np.ndarray] = []
  half_sizes: list[np.ndarray] = []
  for x1, y1, x2, y2 in rects:
    centers.append(np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32))
    half_sizes.append(np.array([(x2 - x1) * 0.5, (y2 - y1) * 0.5], dtype=np.float32))
  _static(
    emit,
    f"{entity_prefix}/static_mask",
    rr.Boxes2D.columns(
      centers=np.stack(centers),
      half_sizes=np.stack(half_sizes),
      colors=_solid_colors(_REJECTED, len(rects)),
      labels=["STATIC"] * len(rects),
    ).partition([len(rects)]),
    rr.Boxes2D.columns(show_labels=[True]),
  )


def _send_pixel_columns(
  emit: Sink,
  entity_path: str,
  pixels_per_kf: list[Float64Array],
  index: _OverlayIndex,
  radius: float,
  color: tuple[int, int, int],
) -> None:
  partition = [int(kp.shape[0]) for kp in pixels_per_kf]
  total = sum(partition)
  if total == 0:
    return
  _send_overlay(
    emit,
    entity_path,
    index,
    rr.Points2D,
    partition,
    positions=np.concatenate([kp for kp in pixels_per_kf if kp.shape[0] > 0]),
    radii=np.full(total, radius, dtype=np.float32),
    colors=_solid_colors(color, total),
  )


def _send_tag_axes_columns(
  emit: Sink,
  entity_prefix: str,
  axes_per_kf: list[Float64Array],
  index: _OverlayIndex,
) -> None:
  partition = [len(a) for a in axes_per_kf]
  total = sum(partition)
  if total == 0:
    return
  rgb = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255)], dtype=np.uint8)
  _send_overlay(
    emit,
    f"{entity_prefix}/tags/axes",
    index,
    rr.LineStrips2D,
    partition,
    strips=np.concatenate(axes_per_kf).astype(np.float32),
    colors=rgb[np.arange(total) % 3],
    radii=np.full(total, 4.0, dtype=np.float32),
  )


def _send_factor_line_columns(
  emit: Sink,
  entity_path: str,
  lines_per_kf: list[Float64Array],
  indexes: list[rr.TimeColumn],
  color: tuple[int, int, int],
) -> None:
  """Per-keyframe ``(n_lines, 2, 3)`` endpoint arrays as LineStrips3D rows."""
  partition = [int(arr.shape[0]) for arr in lines_per_kf]
  total = sum(partition)
  if total == 0:
    return
  strips = [line.astype(np.float32) for arr in lines_per_kf for line in arr]
  _send(
    emit,
    entity_path,
    indexes,
    rr.LineStrips3D.columns(
      strips=strips,
      colors=_solid_colors(color, total),
      radii=np.full(total, 0.0008, dtype=np.float32),
    ).partition(partition),
  )


def _send_lens_columns(
  emit: Sink,
  entity_path: str,
  keypoints: list[Float64Array] | None,
  index: _OverlayIndex,
  color: tuple[int, int, int],
) -> None:
  if not keypoints:
    return
  _send_pixel_columns(emit, f"{entity_path}/keypoints", keypoints, index, 1.5, color)


@dataclass
class LensVizData:
  """One lens's share of a tracker's payload: map, overlays and calibration."""

  points: dict[int, Float64Array] = field(default_factory=dict)
  keypoints: list[Float64Array] | None = None
  rejected_keypoints: list[Float64Array] | None = None
  landmark_reprojections: list[Float64Array] | None = None
  tag_detections: list[list[TagDetection]] | None = None
  tag_axes: list[Float64Array] | None = None
  rmse_per_kf: list[float] | None = None
  body_T_lens: gtsam.Pose3 | None = None
  # MEI intrinsics at decode resolution, logged as a lens static.
  mei: dict[str, float] | None = None
  static_mask_rects: list[tuple[int, int, int, int]] | None = None


@dataclass
class CameraVizData:
  """One tracker's Rerun payload from ``build_run_outputs`` for ``solution_chunks``."""

  name: str
  # Keyframe instants on the session clock.
  keyframe_times: Float64Array
  body_poses: list[gtsam.Pose3] = field(default_factory=list)
  keyframe_frames: npt.NDArray[np.int64] | None = None
  # Per-lens payloads, keyed by the lens name their entity paths carry.
  lenses: dict[str, LensVizData] = field(default_factory=dict)
  velocities: list[Float64Array] | None = None
  gyro_biases: list[Float64Array] | None = None
  accel_biases: list[Float64Array] | None = None
  urdf_root_poses: list[gtsam.Pose3] = field(default_factory=list)
  # URDF joint name → per-keyframe angle, NaN where it could not be recovered.
  joint_angles: dict[str, list[float]] | None = None
  tcp_in_base_tag: tuple[float, float, float] | None = None
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


def _gripper_points(data: CameraVizData) -> Float64Array:
  """Keyframe gripper keypoints if offset and root poses exist, else body positions."""
  if data.tcp_in_base_tag is not None and data.urdf_root_poses:
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


def _held(values: list[float]) -> list[float] | None:
  """Each finite angle held through dropouts; ``None`` when no value is finite."""
  last = next((float(v) for v in values if np.isfinite(v)), None)
  if last is None:
    return None
  held: list[float] = []
  for v in values:
    if np.isfinite(v):
      last = float(v)
    held.append(last)
  return held


def _send_joint_transform(
  emit: Sink,
  entity_path: str,
  joint: Any,
  times: Float64Array,
  angles: Any,
  timeline_start: float | None,
) -> None:
  """One joint's transform stream, padded back to the merged timeline's start."""
  if timeline_start is not None and len(times) and timeline_start < times[0]:
    times = np.concatenate(([timeline_start], times))
    angles = np.concatenate((angles[:1], angles))
  _send(
    emit,
    entity_path,
    [rr.TimeColumn(schema.TIMELINE, duration=times)],
    joint.compute_transform_columns([float(a) for a in angles]),
  )


def _send_anchor(
  emit: Sink, entity: str, world_T_anchor: gtsam.Pose3, tag_size: float
) -> None:
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
  _static(
    emit,
    f"world/{entity}",
    _pose_columns(world_T_anchor),
    rr.LineStrips3D.columns(strips=[square], colors=[_ANCHOR], radii=[0.003]),
  )
  _static(
    emit,
    f"world/{entity}/frame",
    rr.TransformAxes3D.columns(axis_length=[half]),
  )


@dataclass(frozen=True)
class _Indexes:
  """The timelines one tracker's streams are indexed on."""

  trail_times: Float64Array
  times: rr.TimeColumn
  keyframe_column: rr.TimeColumn | None
  keyframes: list[rr.TimeColumn]
  overlay: _OverlayIndex
  highrate: tuple[Float64Array, Float64Array, Float64Array] | None


def _tracker_indexes(data: CameraVizData, frame_period: float | None) -> _Indexes:
  """Every index a tracker's streams share, derived once from its keyframes."""
  trail_times = np.asarray(data.keyframe_times, dtype=np.float64)
  times = rr.TimeColumn(schema.TIMELINE, duration=trail_times)
  keyframe_column: rr.TimeColumn | None = None
  if data.keyframe_frames is not None and frame_period is not None:
    stride = schema.keyframe_stride(1.0 / frame_period)
    keyframe_column = rr.TimeColumn(
      schema.KEYFRAME_TIMELINE,
      sequence=schema.keyframe_index(data.keyframe_frames, stride),
    )
  keyframes = [times] if keyframe_column is None else [times, keyframe_column]
  highrate = None
  if (
    data.highrate_times is not None
    and data.highrate_positions is not None
    and data.highrate_quats is not None
  ):
    highrate = (data.highrate_times, data.highrate_positions, data.highrate_quats)
  return _Indexes(
    trail_times=trail_times,
    times=times,
    keyframe_column=keyframe_column,
    keyframes=keyframes,
    overlay=_overlay_index(trail_times, frame_period, keyframes, keyframe_column),
    highrate=highrate,
  )


def _tracker_statics(
  track: Sink, viz: Sink, data: CameraVizData, color: tuple[int, int, int]
) -> None:
  """The tracker's time-independent geometry: maps, body, lenses and base tag."""
  spatial_root = data.name
  body_path = f"{spatial_root}/body"
  for lens, lens_estimate in data.lenses.items():
    _send_static_points(
      viz, f"{spatial_root}/landmarks/{lens}", lens_estimate.points, color
    )
  _static(track, body_path, rr.TransformAxes3D.columns(axis_length=[_BODY_AXIS_LENGTH]))
  _static(
    viz,
    f"{body_path}/marker",
    rr.Points3D.columns(positions=np.zeros((1, 3)), radii=[0.002], colors=[color]),
  )
  for lens, lens_estimate in data.lenses.items():
    if lens_estimate.body_T_lens is None:
      continue
    lens_columns: list[Any] = [
      _pose_columns(lens_estimate.body_T_lens),
      rr.TransformAxes3D.columns(axis_length=[_BODY_AXIS_LENGTH]),
    ]
    if lens_estimate.mei is not None:
      # The MEI field names are data, so the call is keyed dynamically.
      mei: dict[str, Any] = {k: [v] for k, v in lens_estimate.mei.items()}
      lens_columns.append(rr.AnyValues.columns(**mei))
    _static(track, f"{body_path}/{lens}_lens", *lens_columns)
  if data.body_T_base_tag is not None:
    _static(
      track,
      schema.base_tag_entity(data.name),
      _pose_columns(data.body_T_base_tag),
      rr.TransformAxes3D.columns(axis_length=[0.02]),
    )


def _urdf_model(
  track: Sink, viz: Sink, data: CameraVizData, urdf_path: Path | None
) -> tuple[rr.urdf.UrdfTree, str] | None:
  """Log the gripper model; return its tree and root frame, or ``None`` if unusable."""
  if urdf_path is None:
    return None
  spatial_root = data.name
  try:
    tree = rr.urdf.UrdfTree.from_file_path(
      str(urdf_path),
      entity_path_prefix=spatial_root,
      frame_prefix=f"{data.name}_",
      static_transform_entity_path=f"{spatial_root}/tf_static",
    )
    links = [tree.root_link(), *map(tree.get_joint_child, tree.joints())]
    collision = [p for link in links for p in tree.get_collision_geometry_paths(link)]
    drawn = {p for link in links for p in tree.get_visual_geometry_paths(link)}
    drawn.add(f"/{spatial_root.lstrip('/')}/tf_static")
    model_chunks = tree.stream().drop(content=collision).to_chunks()
    root_frame = f"{data.name}_{tree.root_link().name}"
  except Exception as exc:  # noqa: BLE001 - the URDF model is optional; keep the rest
    traceback.print_exc()
    print(
      f"  [warn] [viz] URDF model skipped ({exc}); logging the trajectory without it"
    )
    return None
  for chunk in model_chunks:
    # The tree names its own paths, so its meshes are recognized by entity path.
    (viz if chunk.entity_path in drawn else track)(chunk)
  _static(
    track,
    f"{spatial_root}/gripper",
    rr.Transform3D.columns(parent_frame=["world"], child_frame=[root_frame]),
    rr.TransformAxes3D.columns(axis_length=[_BODY_AXIS_LENGTH]),
  )
  return tree, root_frame


def _body_streams(
  track: Sink, viz: Sink, data: CameraVizData, indexes: _Indexes
) -> None:
  """The body pose stream, exact keyframe poses and the speed from velocity states."""
  body_path = f"{data.name}/body"
  kf_columns = rr.Transform3D.columns(
    translation=np.stack([p.translation() for p in data.body_poses]),
    quaternion=np.stack([quat_xyzw(p.rotation()) for p in data.body_poses]),
  )
  if indexes.highrate is not None:
    hr_times, hr_positions, hr_quats = indexes.highrate
    _send(
      track,
      body_path,
      [rr.TimeColumn(schema.TIMELINE, duration=hr_times)],
      rr.Transform3D.columns(translation=hr_positions, quaternion=hr_quats),
    )
    if indexes.keyframe_column is not None:
      # Only the exact keyframe states go on the review timeline.
      _send(track, body_path, [indexes.keyframe_column], kf_columns)
  else:
    _send(track, body_path, indexes.keyframes, kf_columns)
  # The exact BA keyframe poses, kept beside the propagated stream.
  _send(track, schema.body_keyframes_entity(data.name), indexes.keyframes, kf_columns)
  velocity: tuple[rr.TimeColumn, Float64Array] | None = None
  if indexes.highrate is not None and data.highrate_vels is not None:
    velocity = (
      rr.TimeColumn(schema.TIMELINE, duration=indexes.highrate[0]),
      data.highrate_vels,
    )
  elif data.velocities is not None:
    velocity = (indexes.times, np.stack(data.velocities))
  if velocity is None:
    return
  velocity_times, velocity_values = velocity
  _send(
    track,
    body_path,
    [velocity_times],
    rr.AnyValues.columns(velocity=_vec3_column(velocity_values)),
  )
  _send(
    viz,
    f"metrics/speed/{data.name}",
    [velocity_times],
    rr.Scalars.columns(scalars=np.linalg.norm(velocity_values, axis=1)),
  )


def _gripper_streams(
  track: Sink, data: CameraVizData, indexes: _Indexes, root_frame: str
) -> tuple[Float64Array, Float64Array] | None:
  """Stream the URDF root pose; return its IMU-propagated ``(t, R)``, else ``None``."""
  gripper_path = f"{data.name}/gripper"
  poses = data.urdf_root_poses
  if indexes.highrate is None or not data.body_poses or not poses:
    _send(
      track,
      gripper_path,
      indexes.keyframes,
      _frame_columns(
        np.stack([p.translation() for p in poses]),
        np.stack([quat_xyzw(p.rotation()) for p in poses]),
        root_frame,
      ),
    )
    return None
  hr_times, hr_positions, hr_quats = indexes.highrate
  body_T_base_tag = data.body_poses[0].inverse().compose(poses[0])
  world_R_body = quats_to_matrices(hr_quats)
  world_t_base_tag = hr_positions + world_R_body @ np.asarray(
    body_T_base_tag.translation()
  )
  world_R_base_tag = world_R_body @ np.asarray(body_T_base_tag.rotation().matrix())
  _send(
    track,
    gripper_path,
    [rr.TimeColumn(schema.TIMELINE, duration=hr_times)],
    _frame_columns(world_t_base_tag, matrices_to_quats(world_R_base_tag), root_frame),
  )
  if indexes.keyframe_column is not None:
    _send(
      track,
      gripper_path,
      [indexes.keyframe_column],
      _frame_columns(
        np.stack([p.translation() for p in poses]),
        np.stack([quat_xyzw(p.rotation()) for p in poses]),
        root_frame,
      ),
    )
  return world_t_base_tag, world_R_base_tag


def _joint_streams(
  track: Sink,
  data: CameraVizData,
  indexes: _Indexes,
  tree: rr.urdf.UrdfTree,
  timeline_start: float | None,
) -> None:
  """Each solved joint's tag and, where a frame detected it, its transform and angle."""
  if data.joint_angles is None:
    return
  for joint_name, raw_values in data.joint_angles.items():
    joint = tree.get_joint_by_name(joint_name)
    if joint is None:
      continue
    joint_path = f"{data.name}/{joint_name}"
    frame_series = (data.joint_frame_angles or {}).get(joint_name)
    if frame_series is None:
      held = _held(raw_values)
      if held is None:
        continue
      _send_joint_transform(
        track, joint_path, joint, indexes.trail_times, held, timeline_start
      )
      _send(
        track,
        joint_path,
        indexes.keyframes,
        rr.AnyValues.columns(angle=np.asarray(raw_values, dtype=np.float64)),
      )
      continue
    f_times, f_angles = frame_series
    _send_joint_transform(track, joint_path, joint, f_times, f_angles, timeline_start)
    if indexes.keyframe_column is not None:
      kf_angles = _held(raw_values) or [0.0] * len(raw_values)
      _send(
        track,
        joint_path,
        [indexes.keyframe_column],
        joint.compute_transform_columns(kf_angles),
      )
    _send(
      track,
      joint_path,
      [rr.TimeColumn(schema.TIMELINE, duration=f_times)],
      rr.AnyValues.columns(angle=np.asarray(f_angles, dtype=np.float64)),
    )


def _trajectory_streams(
  viz: Sink,
  data: CameraVizData,
  indexes: _Indexes,
  world_T_base_tag: tuple[Float64Array, Float64Array] | None,
  color: tuple[int, int, int],
) -> None:
  """Fading gripper trail and distance from start: propagated series, else keyframes."""
  if indexes.highrate is not None:
    hr_times, hr_positions, _ = indexes.highrate
    if world_T_base_tag is not None and data.tcp_in_base_tag is not None:
      offset = np.asarray(data.tcp_in_base_tag, dtype=np.float64)
      world_t_base_tag, world_R_base_tag = world_T_base_tag
      points = world_t_base_tag + world_R_base_tag @ offset
    else:
      points = hr_positions
    times = rr.TimeColumn(schema.TIMELINE, duration=hr_times)
    trail_times = hr_times
  else:
    points = _gripper_points(data)
    times = indexes.times
    trail_times = indexes.trail_times
  entity_path = f"{data.name}/trajectory"
  _send(
    viz,
    entity_path,
    [times],
    rr.LineStrips3D.columns(
      strips=_trail_segments(points, trail_times, _TRAJECTORY_TRAIL_DURATION)
    ),
  )
  _static(
    viz,
    entity_path,
    rr.LineStrips3D.columns(radii=[0.002], colors=[color]),
  )
  _send(
    viz,
    f"metrics/distance_from_start/{data.name}",
    [times],
    rr.Scalars.columns(scalars=np.linalg.norm(points - points[0:1], axis=1)),
  )


def _factor_line_streams(viz: Sink, data: CameraVizData, indexes: _Indexes) -> None:
  """The rays of the reprojection factors that fired at each keyframe."""
  for kind, lines, color in (
    ("anchor", data.anchor_obs_lines, _ANCHOR),
    ("cross_tracker", data.cross_tracker_lines, _GRIPPER_TAG),
  ):
    if lines is None:
      continue
    _send_factor_line_columns(
      viz, f"{data.name}/factor_lines/{kind}", lines, indexes.keyframes, color
    )


def _lens_streams(
  viz: Sink,
  data: CameraVizData,
  indexes: _Indexes,
  anchor_tag_ids: set[int],
  mask_pad_px: int,
) -> None:
  """Every image overlay, a kind at a time so each kind's chunks land together."""
  image_root = schema.image_root(data.name)
  for lens, lens_estimate in data.lenses.items():
    _send_lens_columns(
      viz, f"{image_root}/{lens}", lens_estimate.keypoints, indexes.overlay, _MEASURED
    )
  _send_lens_series(
    viz,
    f"metrics/n_keypoints/{data.name}",
    indexes.keyframes,
    data,
    lambda lens: [kp.shape[0] for kp in lens.keypoints] if lens.keypoints else None,
  )
  for lens, lens_estimate in data.lenses.items():
    if lens_estimate.tag_detections is not None:
      _send_lens_tag_columns(
        viz,
        f"{image_root}/{lens}",
        lens_estimate.tag_detections,
        indexes.overlay,
        anchor_tag_ids,
        mask_pad_px,
      )
  for lens, lens_estimate in data.lenses.items():
    if lens_estimate.rejected_keypoints is not None:
      _send_pixel_columns(
        viz,
        f"{image_root}/{lens}/keypoints_rejected",
        lens_estimate.rejected_keypoints,
        indexes.overlay,
        1.5,
        _REJECTED,
      )
  for lens, lens_estimate in data.lenses.items():
    if lens_estimate.landmark_reprojections is not None:
      _send_pixel_columns(
        viz,
        f"{image_root}/{lens}/landmark_reprojection",
        lens_estimate.landmark_reprojections,
        indexes.overlay,
        3.0,
        _PREDICTED,
      )
  for lens, lens_estimate in data.lenses.items():
    if lens_estimate.tag_axes is not None:
      _send_tag_axes_columns(
        viz, f"{image_root}/{lens}", lens_estimate.tag_axes, indexes.overlay
      )
  for lens, lens_estimate in data.lenses.items():
    if lens_estimate.static_mask_rects:
      _log_static_mask_rects(
        viz, f"{image_root}/{lens}", lens_estimate.static_mask_rects
      )


def _bias_streams(viz: Sink, data: CameraVizData, indexes: _Indexes) -> None:
  """The IMU bias states and the per-lens reprojection error, as scalar series."""
  for metric_name, biases in (
    ("gyro_bias", data.gyro_biases),
    ("accel_bias", data.accel_biases),
  ):
    if biases is None:
      continue
    _send(
      viz,
      f"metrics/{metric_name}/{data.name}",
      indexes.keyframes,
      rr.Scalars.columns(scalars=np.asarray(biases, dtype=np.float64)),
    )
  _send_lens_series(
    viz,
    f"metrics/rmse/{data.name}",
    indexes.keyframes,
    data,
    lambda lens: lens.rmse_per_kf,
  )


def _log_tracker(
  track: Sink,
  viz: Sink,
  data: CameraVizData,
  urdf_path: Path | None,
  anchor_tag_ids: set[int],
  mask_pad_px: int,
  color: tuple[int, int, int],
  timeline_start: float | None,
  frame_period: float | None,
) -> None:
  """Log every per-tracker entity; the caller logs the shared statics."""
  _tracker_statics(track, viz, data, color)
  model = _urdf_model(track, viz, data, urdf_path)
  indexes = _tracker_indexes(data, frame_period)
  _body_streams(track, viz, data, indexes)
  world_T_base_tag = None
  if model is not None:
    tree, root_frame = model
    world_T_base_tag = _gripper_streams(track, data, indexes, root_frame)
    _joint_streams(track, data, indexes, tree, timeline_start)
  _trajectory_streams(viz, data, indexes, world_T_base_tag, color)
  _factor_line_streams(viz, data, indexes)
  _lens_streams(viz, data, indexes, anchor_tag_ids, mask_pad_px)
  _bias_streams(viz, data, indexes)


def _send_lens_series(
  emit: Sink,
  entity_path: str,
  indexes: list[rr.TimeColumn],
  data: CameraVizData,
  values: Callable[[LensVizData], list[float] | list[int] | None],
) -> None:
  """One entity for a per-lens metric, a series per lens that has it."""
  per_lens = {
    lens: values(lens_estimate) for lens, lens_estimate in data.lenses.items()
  }
  present = {lens: v for lens, v in per_lens.items() if v}
  if not present:
    return
  _send(
    emit,
    entity_path,
    indexes,
    rr.Scalars.columns(
      scalars=np.stack(
        [np.asarray(v, dtype=np.float64) for v in present.values()], axis=1
      )
    ),
  )


def _panels() -> tuple[rrb.BlueprintPanel, rrb.SelectionPanel, rrb.TimePanel]:
  """The chrome both blueprints hide, so the two cannot drift apart."""
  return (
    rrb.BlueprintPanel(state=rrb.PanelState.Collapsed),
    rrb.SelectionPanel(state=rrb.PanelState.Collapsed),
    rrb.TimePanel(state=rrb.PanelState.Collapsed, fps=10),
  )


def _trail_overrides(
  slots: list[str],
) -> dict[rr.datatypes.EntityPath | str, rrb.VisibleTimeRanges]:
  """The window every view that draws a trajectory ribbon has to set."""
  return {f"/{n}/trajectory": _trail_window() for n in slots}


def _hidden_overlay_overrides(
  tracker: str, lens: str
) -> dict[rr.datatypes.EntityPath | str, rrb.EntityBehavior]:
  """Visibility overrides hiding image overlays; a key covers its whole subtree."""
  roots = (
    "keypoints",
    "keypoints_rejected",
    "landmark_reprojection",
    "tags",
    "static_mask",
  )
  return {
    f"/image/{tracker}/{lens}/{root}": rrb.EntityBehavior(visible=False)
    for root in roots
  }


def blueprint_for_names(
  slots: list[str], floor_z: float | None = None
) -> rrb.Blueprint:
  """Multi-tracker blueprint from names alone: one image-pair column per tracker."""
  line_grid = (
    rrb.LineGrid3D(
      plane=rr.datatypes.Plane3D(normal=(0.0, 0.0, 1.0), distance=float(floor_z))
    )
    if floor_z is not None
    else None
  )
  hidden = rrb.EntityBehavior(visible=False)
  overrides: dict[
    rr.datatypes.EntityPath | str, rrb.EntityBehavior | rrb.VisibleTimeRanges
  ] = {**_trail_overrides(slots)}
  overrides.update({f"/{n}/landmarks": hidden for n in slots})
  image_columns = [
    rrb.Vertical(
      rrb.Spatial2DView(
        name=f"{n}/front",
        origin=f"/image/{n}/front",
        overrides=_hidden_overlay_overrides(n, "front"),
      ),
      rrb.Spatial2DView(
        name=f"{n}/rear",
        origin=f"/image/{n}/rear",
        overrides=_hidden_overlay_overrides(n, "rear"),
      ),
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
        overrides=overrides,
        line_grid=line_grid,
      ),
      rrb.Horizontal(*image_columns) if len(image_columns) > 1 else image_columns[0],
      column_shares=[2, max(1, len(slots))],
    ),
    *_panels(),
  )


def _trail_window() -> rrb.VisibleTimeRanges:
  """The time span a trajectory ribbon draws over; without it a view draws one step."""
  return rrb.VisibleTimeRanges(
    [
      rrb.VisibleTimeRange(
        schema.TIMELINE,
        start=rrb.TimeRangeBoundary.cursor_relative(
          seconds=-_TRAJECTORY_TRAIL_DURATION
        ),
        end=rrb.TimeRangeBoundary.cursor_relative(),
      )
    ]
  )


def segment_card_blueprint(slots: list[str]) -> rrb.Blueprint:
  """Segment-table card whose 3D view drops images, landmark clouds and factor lines."""
  contents = ["+ /**", "- /image/**"]
  for name in slots:
    contents.append(f"- /{name}/landmarks/**")
    contents.append(f"- /{name}/factor_lines/**")
  return rrb.Blueprint(
    rrb.Spatial3DView(
      name="3D",
      origin="/",
      contents=contents,
      overrides=_trail_overrides(slots),
    ),
    *_panels(),
  )


def run_blueprint(cameras: list[CameraVizData]) -> rrb.Blueprint:
  """A solved run's dashboard, its floor grid at the lowest trajectory point."""
  gripper_points = [_gripper_points(t) for t in cameras]
  floor_zs = [float(pts[:, 2].min()) for pts in gripper_points if len(pts)]
  floor_z = min(floor_zs) if floor_zs else None
  return blueprint_for_names([t.name for t in cameras], floor_z)


def solution_chunks(
  track: Sink,
  viz: Sink,
  cameras: list[CameraVizData],
  world_T_anchor: dict[int, gtsam.Pose3],
  reference_tag_id: int,
  anchor_tag_size: float,
  mask_pad_px: int,
  urdf_path: Path | None,
  timeline_start: float | None = None,
  frame_period: float | None = None,
) -> None:
  """Every entity of a solved run as chunks, sent to the ``track`` or ``viz`` sink."""
  _static(
    track,
    "world",
    rr.ViewCoordinates.columns(xyz=[rr.components.ViewCoordinates.RIGHT_HAND_Z_UP]),
    rr.Transform3D.columns(child_frame=["world"]),
  )
  for tag_id, pose in world_T_anchor.items():
    entity = "anchor_tag" if tag_id == reference_tag_id else f"anchor_tag_{tag_id}"
    _send_anchor(track, entity, pose, anchor_tag_size)
  for i, t in enumerate(cameras):
    _log_tracker(
      track,
      viz,
      t,
      urdf_path=urdf_path,
      anchor_tag_ids=set(world_T_anchor),
      mask_pad_px=mask_pad_px,
      color=_SLOT_COLORS[i % len(_SLOT_COLORS)],
      timeline_start=timeline_start,
      frame_period=frame_period,
    )
