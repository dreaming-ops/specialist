"""Rerun chunks of a solved run, sorted into the estimate layer and the viz layer."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gtsam
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
from rerun.experimental import Chunk
from specialist_dataset import schema
from specialist_dataset.poses import quats_to_matrices

from specialist.vi_ba.blueprint import BODY_AXIS, TAG_AXIS, TRAJECTORY_TRAIL_DURATION
from specialist.vi_ba.geometry import matrices_to_quats, quat_xyzw

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

__all__ = [
  "LensEstimate",
  "CameraEstimate",
  "Layers",
  "estimate_chunks",
]


Sink = Callable[[Chunk], None]


@dataclass(frozen=True)
class Layers:
  """Chunks by layer: values to compute with in ``estimate``, drawings in ``viz``."""

  estimate: Sink
  viz: Sink


def _send(emit: Sink, entity_path: str, indexes: Any, columns: Any) -> None:
  """One temporal chunk, shaped exactly as the columnar log call was."""
  emit(Chunk.from_columns(entity_path, indexes=indexes, columns=columns))


def _in_world(emit: Sink, entity_path: str) -> None:
  """Declare the entity's data in the world frame; never call it on a parent."""
  _static(emit, entity_path, rr.CoordinateFrame.columns(frame=[schema.WORLD_FRAME]))


def _static(emit: Sink, entity_path: str, *columns: Any) -> None:
  """One static chunk carrying every archetype logged at that entity."""
  emit(
    Chunk.from_columns(
      entity_path, indexes=[], columns=[c for group in columns for c in group]
    )
  )


def _send_series(
  layers: Layers,
  entity_path: str,
  indexes: list[rr.TimeColumn],
  values: Any,
  names: list[str],
) -> None:
  """A metric's series, its names as one partitioned static row."""
  _send(layers.estimate, entity_path, indexes, rr.Scalars.columns(scalars=values))
  _static(
    layers.estimate,
    entity_path,
    rr.AnyValues.columns(series=names).partition([len(names)]),
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
  """World-parented Transform3D columns, one row per sample, each naming its edge."""
  return rr.Transform3D.columns(
    translation=translations,
    quaternion=quats,
    parent_frame=[schema.WORLD_FRAME] * len(translations),
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
  layers: Layers,
  entity_path: str,
  points: dict[int, Float64Array],
  color: tuple[int, int, int],
) -> None:
  if not points:
    return
  _in_world(layers.estimate, entity_path)
  xyz = np.asarray(list(points.values()), dtype=np.float64)
  _static(
    layers.estimate,
    entity_path,
    rr.Points3D.columns(positions=xyz).partition([len(xyz)]),
  )
  _static(layers.viz, entity_path, rr.Points3D.columns(radii=[0.002], colors=[color]))


def _solid_colors(color: tuple[int, int, int], n: int) -> np.ndarray:
  """An ``(n, 3)`` uint8 array of one repeated RGB color."""
  return np.tile(np.asarray(color, dtype=np.uint8), (n, 1))


def _send_overlay(
  layers: Layers,
  entity_path: str,
  index: list[rr.TimeColumn],
  archetype: Any,
  partition: list[int],
  measured: dict[str, Any],
  drawn: dict[str, Any],
) -> None:
  """Keyframe overlay: ``measured`` columns to estimate, ``drawn`` columns to viz."""
  if measured:
    rows = archetype.columns(**measured).partition(partition)
    _send(layers.estimate, entity_path, index, rows)
  if drawn:
    _send(
      layers.viz, entity_path, index, archetype.columns(**drawn).partition(partition)
    )


def _send_lens_tag_columns(
  layers: Layers,
  tags_path: str,
  tags_per_kf: list[list[TagDetection]],
  index: list[rr.TimeColumn],
  anchor_tag_ids: set[int],
) -> None:
  """Per-keyframe AprilTag quads, centers and ids; anchors gold, other tags cyan."""
  tags = [det for dets in tags_per_kf for det in dets]
  if not tags:
    return
  partition = [len(dets) for dets in tags_per_kf]
  ids = np.asarray([int(det.tag_id) for det in tags], dtype=np.int32)
  is_anchor = np.isin(ids, list(anchor_tag_ids))[:, None]
  colors = np.where(is_anchor, _ANCHOR, _GRIPPER_TAG).astype(np.uint8)
  strips = [np.vstack([d.corners, d.corners[:1]]).astype(np.float32) for d in tags]
  _send_overlay(
    layers,
    tags_path,
    index,
    rr.LineStrips2D,
    partition,
    measured={"strips": strips},
    drawn={"colors": colors, "radii": np.full(len(tags), 5.0, dtype=np.float32)},
  )
  _send_overlay(
    layers,
    tags_path,
    index,
    rr.AnyValues,
    partition,
    measured={"tag_id": ids},
    drawn={},
  )
  _send_overlay(
    layers,
    f"{tags_path}/center",
    index,
    rr.Points2D,
    partition,
    measured={"positions": np.asarray([d.center for d in tags], dtype=np.float32)},
    drawn={
      "labels": [str(i) for i in ids],
      "show_labels": [True] * len(tags),
      "colors": colors,
      "radii": np.full(len(tags), 10.0, dtype=np.float32),
    },
  )


def _send_pixel_columns(
  layers: Layers,
  entity_path: str,
  pixels_per_kf: list[Float64Array],
  index: list[rr.TimeColumn],
  radius: float,
  color: tuple[int, int, int],
  measured: bool,
) -> None:
  """Per-keyframe image points, ``measured`` by the lens or predicted by the solve."""
  partition = [int(kp.shape[0]) for kp in pixels_per_kf]
  total = sum(partition)
  if total == 0:
    return
  positions = {
    "positions": np.concatenate([kp for kp in pixels_per_kf if kp.shape[0] > 0])
  }
  drawn = {
    "radii": np.full(total, radius, dtype=np.float32),
    "colors": _solid_colors(color, total),
  }
  _send_overlay(
    layers,
    entity_path,
    index,
    rr.Points2D,
    partition,
    measured=positions if measured else {},
    drawn=drawn if measured else {**positions, **drawn},
  )


def _send_tag_axes_columns(
  layers: Layers,
  axes_path: str,
  axes_per_kf: list[Float64Array],
  index: list[rr.TimeColumn],
) -> None:
  """The tag's pose drawn as three colored strokes, viz only."""
  partition = [len(a) for a in axes_per_kf]
  total = sum(partition)
  if total == 0:
    return
  rgb = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255)], dtype=np.uint8)
  _send_overlay(
    layers,
    axes_path,
    index,
    rr.LineStrips2D,
    partition,
    measured={},
    drawn={
      "strips": np.concatenate(axes_per_kf).astype(np.float32),
      "colors": rgb[np.arange(total) % 3],
      "radii": np.full(total, 4.0, dtype=np.float32),
    },
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
  _in_world(emit, entity_path)
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


@dataclass
class LensEstimate:
  """One lens's map and its per-keyframe image measurements."""

  points: dict[int, Float64Array] = field(default_factory=dict)
  keypoints: list[Float64Array] = field(default_factory=list)
  rejected_keypoints: list[Float64Array] = field(default_factory=list)
  landmark_reprojections: list[Float64Array] = field(default_factory=list)
  tag_detections: list[list[TagDetection]] = field(default_factory=list)
  tag_axes: list[Float64Array] = field(default_factory=list)
  rmse_per_kf: list[float] = field(default_factory=list)


@dataclass
class CameraEstimate:
  """One camera's solved state over a run and what each lens saw."""

  slot: str
  # Keyframe instants on the session clock.
  keyframe_times: Float64Array
  keyframe_frames: npt.NDArray[np.int64]
  # The base tag's pose in the body frame, a bundle-adjustment variable.
  body_T_base_tag: gtsam.Pose3
  base_tag_id: int
  body_T_rear_lens: gtsam.Pose3
  # The series propagated between the keyframes, at the IMU's own sample times.
  highrate_times: Float64Array
  highrate_positions: Float64Array
  highrate_quats: Float64Array
  highrate_vels: Float64Array
  # The solved body pose at each keyframe; the camera entity carries the propagated one.
  body_poses: list[gtsam.Pose3]
  # Per-lens payloads, keyed by the lens name their entity paths carry.
  lenses: dict[str, LensEstimate] = field(default_factory=dict)
  velocities: list[Float64Array] = field(default_factory=list)
  gyro_biases: list[Float64Array] = field(default_factory=list)
  accel_biases: list[Float64Array] = field(default_factory=list)
  tcp_in_base_tag: tuple[float, float, float] = (0.0, 0.0, 0.0)
  # Per-keyframe ``(n, 2, 3)`` lens-to-corner rays, one per firing reprojection factor.
  anchor_obs_lines: list[Float64Array] = field(default_factory=list)
  cross_camera_lines: list[Float64Array] = field(default_factory=list)
  # Per-joint pinned finger angles at the frame rate, as (times, angles).
  joint_frame_angles: dict[str, tuple[Float64Array, Float64Array]] = field(
    default_factory=dict
  )
  # Per revolute joint name, the id of the tag riding it.
  joint_tags: dict[str, int] = field(default_factory=dict)


def _send_joint_transform(
  emit: Sink,
  entity_path: str,
  joint: Any,
  times: Float64Array,
  angles: Any,
  indexes: _Indexes,
) -> None:
  """Stream one joint's transforms, held back to ``timeline_start`` for the viewer."""
  if len(times) and indexes.timeline_start < times[0]:
    times = np.concatenate(([indexes.timeline_start], times))
    angles = np.concatenate((angles[:1], angles))
  _send(
    emit,
    entity_path,
    indexes.dense(times),
    joint.compute_transform_columns([float(a) for a in angles]),
  )


def _send_tag(
  layers: Layers,
  entity_path: str,
  placement: Any,
  axis_length: float = TAG_AXIS,
  **values: Any,
) -> None:
  """One tag's placement, values and triad; ``placement`` is a pose or a frame name."""
  cells: dict[str, Any] = {name: [v] for name, v in values.items()}
  _static(layers.estimate, entity_path, placement, rr.AnyValues.columns(**cells))
  _static(
    layers.viz, entity_path, rr.TransformAxes3D.columns(axis_length=[axis_length])
  )


@dataclass(frozen=True)
class _Indexes:
  """The timelines one camera's streams are indexed on."""

  keyframes: list[rr.TimeColumn]
  overlay: list[rr.TimeColumn]
  frame_period: float
  timeline_start: float

  def dense(self, times: Float64Array) -> list[rr.TimeColumn]:
    """Session time and each sample's keyframe-timeline bucket for keyframe joins."""
    stride = schema.keyframe_stride(1.0 / self.frame_period)
    rows = np.round(times / self.frame_period).astype(np.int64)
    return [
      rr.TimeColumn(schema.TIMELINE, duration=times),
      rr.TimeColumn(
        schema.KEYFRAME_TIMELINE, sequence=schema.keyframe_index(rows, stride)
      ),
    ]


def _camera_indexes(
  data: CameraEstimate, frame_period: float, timeline_start: float
) -> _Indexes:
  """Every index a camera's streams share, derived once from its keyframes."""
  stride = schema.keyframe_stride(1.0 / frame_period)
  keyframe_column = rr.TimeColumn(
    schema.KEYFRAME_TIMELINE,
    sequence=schema.keyframe_index(data.keyframe_frames, stride),
  )
  return _Indexes(
    keyframes=[
      rr.TimeColumn(schema.TIMELINE, duration=data.keyframe_times),
      keyframe_column,
    ],
    overlay=[keyframe_column],
    frame_period=frame_period,
    timeline_start=timeline_start,
  )


def _camera_statics(
  layers: Layers, data: CameraEstimate, color: tuple[int, int, int]
) -> None:
  """Static geometry: lens maps, body triad and marker, base tag and rear lens seat."""
  for lens, lens_estimate in data.lenses.items():
    _send_static_points(
      layers,
      schema.landmark(data.slot, lens),
      lens_estimate.points,
      color,
    )
  body = schema.camera(data.slot)
  _static(layers.viz, body, rr.TransformAxes3D.columns(axis_length=[BODY_AXIS]))
  _static(
    layers.viz,
    f"{body}/marker",
    rr.Points3D.columns(positions=np.zeros((1, 3)), radii=[0.002], colors=[color]),
  )
  # A child of the body: entity nesting composes the body -> tag transform.
  _send_tag(
    layers,
    schema.base_tag(data.slot),
    _pose_columns(data.body_T_base_tag),
    tag_id=data.base_tag_id,
  )
  _static(
    layers.estimate,
    schema.lens_seat(data.slot, "rear"),
    _pose_columns(data.body_T_rear_lens),
  )


def _urdf_model(
  layers: Layers, data: CameraEstimate, urdf_path: Path
) -> tuple[rr.urdf.UrdfTree, str]:
  """The gripper model and its root frame; an unreadable mesh costs only pictures."""
  gripper_root = "/" + schema.gripper(data.slot)
  # The loader escapes prefix slashes, so chunks move under the gripper after loading.
  tree = rr.urdf.UrdfTree.from_file_path(
    str(urdf_path),
    entity_path_prefix=data.slot,
    frame_prefix=f"{data.slot}_",
    static_transform_entity_path=schema.gripper_part(data.slot, "tf_static"),
  )
  root_frame = f"{data.slot}_{tree.root_link().name}"
  model_root = f"/{data.slot}/{tree.name}"
  try:
    links = [tree.root_link(), *map(tree.get_joint_child, tree.joints())]
    collision = [p for link in links for p in tree.get_collision_geometry_paths(link)]
    drawn = {p for link in links for p in tree.get_visual_geometry_paths(link)}
    model_chunks = tree.stream().drop(content=collision).to_chunks()
  except Exception as exc:  # noqa: BLE001 - a mesh failure costs pictures alone
    print(f"  [warn] [model] geometry skipped ({exc})")
    drawn = set()
    model_chunks = [
      chunk
      for chunk in tree.stream().to_chunks()
      if "geometries" not in chunk.entity_path
    ]
  # Schema paths carry no leading slash; a chunk's entity path does.
  static_path = f"/{schema.gripper_part(data.slot, 'tf_static')}"
  for chunk in model_chunks:
    # The static transforms are already at their final path.
    tail = chunk.entity_path.removeprefix(model_root)
    sink = layers.viz if chunk.entity_path in drawn else layers.estimate
    if chunk.entity_path == static_path:
      for fixed in _fixed_joint_transforms(chunk, tree, data.slot):
        sink(fixed)
      continue
    sink(
      chunk
      if tail == chunk.entity_path
      else chunk.with_entity_path(gripper_root + tail)
    )
  # CoordinateFrame places the entity; a two-frame transform only defines an edge.
  _static(
    layers.estimate,
    gripper_root,
    rr.CoordinateFrame.columns(frame=[root_frame]),
    rr.AnyValues.columns(
      robot=[tree.name],
      urdf_sha=[hashlib.sha256(urdf_path.read_bytes()).hexdigest()],
    ),
  )
  return tree, root_frame


def _fixed_joint_transforms(chunk: Any, tree: Any, slot: str) -> list[Any]:
  """The model's static transforms, less the joints the solve writes per keyframe."""
  fixed = {
    f"{slot}_{tree.get_joint_child(joint).name}"
    for joint in tree.joints()
    if joint.joint_type == "fixed"
  }
  batch = chunk.to_record_batch()
  child = batch.column(batch.schema.get_field_index("Transform3D:child_frame"))
  keep = pc.is_in(pc.list_flatten(child), value_set=pa.array(sorted(fixed)))
  out = Chunk.from_record_batch(batch.filter(keep))
  return out if isinstance(out, list) else [out]


def _pose_streams(layers: Layers, data: CameraEstimate, indexes: _Indexes) -> None:
  """Body pose and velocity at the keyframes and, propagated, on the camera entity."""
  # Naming both frames keeps path composition from stacking this on the camera pose.
  _send(
    layers.estimate,
    schema.state(data.slot, "pose"),
    indexes.keyframes,
    _frame_columns(
      np.stack([p.translation() for p in data.body_poses]),
      np.stack([quat_xyzw(p.rotation()) for p in data.body_poses]),
      schema.state_frame(data.slot),
    ),
  )
  _send(
    layers.estimate,
    schema.camera(data.slot),
    indexes.dense(data.highrate_times),
    rr.Transform3D.columns(
      translation=data.highrate_positions, quaternion=data.highrate_quats
    ),
  )
  if data.velocities:
    _send(
      layers.estimate,
      schema.state(data.slot, "lin_vel"),
      indexes.keyframes,
      rr.AnyValues.columns(lin_vel=_vec3_column(np.stack(data.velocities))),
    )
  _send(
    layers.estimate,
    schema.lin_vel(data.slot),
    indexes.dense(data.highrate_times),
    rr.AnyValues.columns(lin_vel=_vec3_column(data.highrate_vels)),
  )


def _gripper_streams(
  layers: Layers, data: CameraEstimate, indexes: _Indexes, root_frame: str
) -> tuple[Float64Array, Float64Array]:
  """Stream the URDF root pose; return its propagated translations and rotations."""
  body_T_base_tag = data.body_T_base_tag
  world_R_body = quats_to_matrices(data.highrate_quats)
  world_t_base_tag = data.highrate_positions + world_R_body @ np.asarray(
    body_T_base_tag.translation()
  )
  world_R_base_tag = world_R_body @ np.asarray(body_T_base_tag.rotation().matrix())
  _send(
    layers.estimate,
    schema.gripper_part(data.slot, "transforms"),
    indexes.dense(data.highrate_times),
    _frame_columns(world_t_base_tag, matrices_to_quats(world_R_base_tag), root_frame),
  )
  return world_t_base_tag, world_R_base_tag


def _joint_streams(
  layers: Layers,
  data: CameraEstimate,
  indexes: _Indexes,
  tree: rr.urdf.UrdfTree,
) -> None:
  """Each tagged joint's tag and, where a frame detected it, its transform and angle."""
  for joint_name, tag_id in data.joint_tags.items():
    joint = tree.get_joint_by_name(joint_name)
    if joint is None:
      continue
    joint_path = schema.gripper_part(data.slot, joint_name)
    child = tree.get_joint_child(joint).name
    _static(
      layers.estimate,
      schema.joint_model(data.slot, joint_name),
      rr.AnyValues.columns(
        joint_type=[joint.joint_type],
        axis=[np.asarray(joint.axis, dtype=np.float64)],
        limit_lower=[joint.limit_lower],
        limit_upper=[joint.limit_upper],
      ),
    )
    _send_tag(
      layers,
      schema.joint_tag(data.slot, joint_name),
      rr.CoordinateFrame.columns(frame=[f"{data.slot}_{child}_tag"]),
      tag_id=tag_id,
    )
    frame_series = data.joint_frame_angles.get(joint_name)
    if frame_series is None:
      continue
    _send_joint_transform(
      layers.estimate,
      schema.gripper_part(data.slot, "transforms"),
      joint,
      *frame_series,
      indexes,
    )
    _static(
      layers.estimate,
      joint_path,
      rr.CoordinateFrame.columns(frame=[f"{data.slot}_{child}"]),
    )
    # The angle keeps the measured frame times; only the transform is held back.
    times, angles = frame_series
    _send(
      layers.estimate,
      joint_path,
      indexes.dense(times),
      rr.AnyValues.columns(angle=np.asarray(angles, dtype=np.float64)),
    )


def _trajectory_streams(
  layers: Layers,
  data: CameraEstimate,
  world_T_base_tag: tuple[Float64Array, Float64Array],
  color: tuple[int, int, int],
) -> None:
  """The fading trail of the gripper TCP along the propagated series."""
  world_t_base_tag, world_R_base_tag = world_T_base_tag
  world_t_tcp = world_t_base_tag + world_R_base_tag @ np.asarray(
    data.tcp_in_base_tag, dtype=np.float64
  )
  entity_path = schema.trajectory(data.slot)
  _in_world(layers.viz, entity_path)
  _send(
    layers.viz,
    entity_path,
    [rr.TimeColumn(schema.TIMELINE, duration=data.highrate_times)],
    rr.LineStrips3D.columns(
      strips=_trail_segments(
        world_t_tcp, data.highrate_times, TRAJECTORY_TRAIL_DURATION
      )
    ),
  )
  _static(
    layers.viz,
    entity_path,
    rr.LineStrips3D.columns(radii=[0.002], colors=[color]),
  )


def _factor_line_streams(
  layers: Layers, data: CameraEstimate, indexes: _Indexes
) -> None:
  """The rays of the reprojection factors that fired at each keyframe."""
  for kind, lines, color in (
    ("anchor", data.anchor_obs_lines, _ANCHOR),
    ("cross_camera", data.cross_camera_lines, _GRIPPER_TAG),
  ):
    _send_factor_line_columns(
      layers.viz, schema.factor(data.slot, kind), lines, indexes.keyframes, color
    )


def _lens_streams(
  layers: Layers,
  data: CameraEstimate,
  indexes: _Indexes,
  anchor_tag_ids: set[int],
) -> None:
  """Every image overlay, a kind at a time so each kind's chunks land together."""
  for lens, lens_estimate in data.lenses.items():
    _send_pixel_columns(
      layers,
      schema.detection(data.slot, lens, schema.KEYPOINTS_DETECTION),
      lens_estimate.keypoints,
      indexes.overlay,
      1.5,
      _MEASURED,
      measured=True,
    )
  for lens, lens_estimate in data.lenses.items():
    _send_lens_tag_columns(
      layers,
      schema.detection(data.slot, lens, schema.TAGS_DETECTION),
      lens_estimate.tag_detections,
      indexes.overlay,
      anchor_tag_ids,
    )
  for lens, lens_estimate in data.lenses.items():
    _send_pixel_columns(
      layers,
      schema.detection(data.slot, lens, schema.REJECTED_KEYPOINTS_DETECTION),
      lens_estimate.rejected_keypoints,
      indexes.overlay,
      1.5,
      _REJECTED,
      measured=True,
    )
  for lens, lens_estimate in data.lenses.items():
    _send_pixel_columns(
      layers,
      schema.overlay(data.slot, lens, schema.LANDMARK_REPROJECTION_OVERLAY),
      lens_estimate.landmark_reprojections,
      indexes.overlay,
      3.0,
      _PREDICTED,
      measured=False,
    )
  for lens, lens_estimate in data.lenses.items():
    _send_tag_axes_columns(
      layers,
      schema.overlay(data.slot, lens, schema.TAG_AXES_OVERLAY),
      lens_estimate.tag_axes,
      indexes.overlay,
    )


def _bias_streams(layers: Layers, data: CameraEstimate, indexes: _Indexes) -> None:
  """The solved IMU biases, beside the raw streams they correct."""
  for kind, biases in (("gyro", data.gyro_biases), ("accel", data.accel_biases)):
    if not biases:
      continue
    _send_series(
      layers,
      schema.imu_bias(data.slot, kind),
      indexes.keyframes,
      np.asarray(biases, dtype=np.float64),
      [f"{data.slot}/{kind}_bias/{axis}" for axis in "xyz"],
    )


def _fit_streams(layers: Layers, data: CameraEstimate, indexes: _Indexes) -> None:
  """Per-keyframe keypoint count and reprojection RMSE, a series per lens that has it."""
  for metric, per_lens in (
    (
      "n_keypoints",
      {lens: [len(kp) for kp in e.keypoints] for lens, e in data.lenses.items()},
    ),
    ("rmse", {lens: e.rmse_per_kf for lens, e in data.lenses.items()}),
  ):
    present = {lens: v for lens, v in per_lens.items() if v}
    if present:
      _send_series(
        layers,
        schema.fit(data.slot, metric),
        indexes.keyframes,
        np.stack([np.asarray(v, dtype=np.float64) for v in present.values()], axis=1),
        [f"{data.slot}/{lens}" for lens in present],
      )


def _send_camera(
  layers: Layers,
  data: CameraEstimate,
  urdf_path: Path,
  anchor_tag_ids: set[int],
  color: tuple[int, int, int],
  timeline_start: float,
  frame_period: float,
) -> None:
  """Log every entity under one camera; the caller logs the shared world statics."""
  _camera_statics(layers, data, color)
  tree, root_frame = _urdf_model(layers, data, urdf_path)
  indexes = _camera_indexes(data, frame_period, timeline_start)
  _pose_streams(layers, data, indexes)
  world_T_base_tag = _gripper_streams(layers, data, indexes, root_frame)
  _joint_streams(layers, data, indexes, tree)
  _trajectory_streams(layers, data, world_T_base_tag, color)
  _factor_line_streams(layers, data, indexes)
  _lens_streams(layers, data, indexes, anchor_tag_ids)
  _bias_streams(layers, data, indexes)
  _fit_streams(layers, data, indexes)


def estimate_chunks(
  layers: Layers,
  cameras: list[CameraEstimate],
  world_T_anchor: dict[int, gtsam.Pose3],
  reference_tag_id: int,
  anchor_tag_size: float,
  urdf_path: Path,
  timeline_start: float,
  frame_period: float,
) -> None:
  """Every entity of a solved run as chunks, sorted into the estimate and viz layers."""
  _static(
    layers.estimate,
    schema.WORLD,
    rr.ViewCoordinates.columns(xyz=[rr.components.ViewCoordinates.RIGHT_HAND_Z_UP]),
    rr.Transform3D.columns(child_frame=[schema.WORLD_FRAME]),
  )
  for tag_id, pose in world_T_anchor.items():
    _send_tag(
      layers,
      schema.anchor(tag_id),
      _pose_columns(pose),
      axis_length=anchor_tag_size / 2.0,
      tag_id=tag_id,
      is_reference=tag_id == reference_tag_id,
      tag_size=anchor_tag_size,
    )
  for i, camera in enumerate(cameras):
    _send_camera(
      layers,
      camera,
      urdf_path=urdf_path,
      anchor_tag_ids=set(world_T_anchor),
      color=_SLOT_COLORS[i % len(_SLOT_COLORS)],
      timeline_start=timeline_start,
      frame_period=frame_period,
    )
