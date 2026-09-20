"""The single source of dataset entity paths, timeline names and timeline decoding."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from specialist_camera import LENSES as LENSES
from specialist_camera import LensName as LensName

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

APPLICATION_ID = "specialist"
# One property group per layer: a name has exactly one writer after a merge.
VIZ_PROPERTIES = "viz"
TIMELINE = "video_time"
KEYFRAME_TIMELINE = "keyframe"
ROW_TIMELINE = "row"
# An image's own capture instant on the row lattice, a value beside its blob.
CAPTURE_ROW = "capture_row"
NS_PER_SECOND = 1_000_000_000

CAMERA_ROOT = "camera"
WORLD = "world"
# A world quantity under a posed parent must name this frame, or it rides that pose.
WORLD_FRAME = "world"
BASE = "base"
ESTIMATE = "estimate"
VIZ = "viz"
MOCAP = "mocap"
STATE = "state"


def state_layer(scale: int, f_row: float) -> str:
  """A state configuration's directory: ``state_s8_r100`` is scale 8 at 100 rows/s."""
  return f"{STATE}_s{scale}_r{round(f_row)}"


def layer_dir(dataset: Path, layer: str) -> Path:
  """Where one layer's recordings sit in a layer-first dataset."""
  return dataset / layer


def layer_rrd(dataset: Path, layer: str, session: str) -> Path:
  """Where one session's layer sits in a layer-first dataset."""
  return layer_dir(dataset, layer) / f"{session}.rrd"


def camera_slots(serials: Sequence[str]) -> list[str]:
  """Each capture's slot, its serial's rank, which names no camera across sessions."""
  if not all(serials) or len(set(serials)) != len(serials):
    raise ValueError(f"cameras need one distinct serial each, got {list(serials)}")
  rank = {serial: i for i, serial in enumerate(sorted(serials))}
  return [f"cam{rank[serial]:02d}" for serial in serials]


def keyframe_stride(fps: float) -> int:
  """Frames per keyframe, about 0.1 s."""
  return max(1, round(fps / 10.0))


def keyframe_index(session_frames: Int64Array, stride: int) -> Int64Array:
  """Each session-clock frame's ``keyframe``: the solver's ``kf`` at phase 0."""
  return (session_frames + stride - 1) // stride


def times_ns(column: pa.Array | pa.ChunkedArray) -> Int64Array:
  """A duration column as integer nanoseconds; raises when the read would copy."""
  values = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
  out: Int64Array = values.to_numpy(zero_copy_only=True).astype("int64")
  return out


def seconds(column: pa.Array | pa.ChunkedArray) -> Float64Array:
  out: Float64Array = times_ns(column) / NS_PER_SECOND
  return out


def camera(slot: str) -> str:
  """The camera body, whose solved pose defines the body frame every child rides."""
  return f"{CAMERA_ROOT}/{slot}"


def lens(slot: str, name: str) -> str:
  """One lens: its pose in the camera body and its projection model."""
  return f"{camera(slot)}/{name}"


def lens_seat(slot: str, name: str) -> str:
  """The body_T_lens pose the bundle solved."""
  return f"{camera(slot)}/{name}_seat"


def video(slot: str, name: str) -> str:
  """A lens's encoded frames, a leaf with no temporal children."""
  return f"{lens(slot, name)}/video"


HANDS = ("left", "right")


def state_hand(hand: str) -> str:
  """What the policy sees of one hand, with the camera slot as a value."""
  return f"{STATE}/{hand}"


def state_image(hand: str) -> str:
  """Rear-lens images at the policy's resolution, blanked outside the fisheye disc."""
  return f"{state_hand(hand)}/image"


def state_pose(hand: str) -> str:
  """The gripper's base-tag frame in the world."""
  return f"{state_hand(hand)}/pose"


def state_lin_vel(hand: str) -> str:
  """The base-tag origin's velocity in the world frame, m/s."""
  return f"{state_hand(hand)}/lin_vel"


def state_ang_vel(hand: str) -> str:
  """The gripper's angular velocity in the world frame, rad/s."""
  return f"{state_hand(hand)}/ang_vel"


def state_opening(hand: str) -> str:
  """The coupled finger joint angle, rad, in the left finger's sign."""
  return f"{state_hand(hand)}/opening"


TAGS_DETECTION = "tags"
KEYPOINTS_DETECTION = "keypoints"
REJECTED_KEYPOINTS_DETECTION = "keypoints_rejected"
DETECTION_KINDS = (TAGS_DETECTION, KEYPOINTS_DETECTION, REJECTED_KEYPOINTS_DETECTION)

TAG_AXES_OVERLAY = "tag_axes"
LANDMARK_REPROJECTION_OVERLAY = "landmark_reprojection"
OVERLAY_KINDS = (TAG_AXES_OVERLAY, LANDMARK_REPROJECTION_OVERLAY)


def detection(slot: str, name: str, kind: str) -> str:
  """What the lens measured, in its own pixels."""
  return f"{lens(slot, name)}/detection/{kind}"


def tag_center(slot: str, lens: str) -> str:
  """A detected tag's center pixel, the cheirality ray's origin."""
  return f"{detection(slot, lens, TAGS_DETECTION)}/center"


def overlay(slot: str, name: str, kind: str) -> str:
  """A picture the solve draws in a lens's pixels: nothing reads it back."""
  return f"{lens(slot, name)}/overlay/{kind}"


def imu(slot: str) -> str:
  """The IMU: its raw-axes-to-body rotation and its factory error terms."""
  return f"{camera(slot)}/imu"


def imu_signal(slot: str, kind: str) -> str:
  """A raw inertial stream, ``gyro`` or ``accel``, on the IMU's own axes."""
  return f"{imu(slot)}/{kind}"


def imu_bias(slot: str, kind: str) -> str:
  """One solved bias stream, ``gyro`` or ``accel``, on the body axes."""
  return f"{imu(slot)}/bias/{kind}"


def exposure(slot: str) -> str:
  return f"{camera(slot)}/exposure"


def autoexposure(slot: str) -> str:
  return f"{camera(slot)}/autoexposure"


def lin_vel(slot: str) -> str:
  """Linear velocity in the world frame, at IMU rate."""
  return f"{camera(slot)}/lin_vel"


def base_tag(slot: str) -> str:
  """The gripper tag bolted to the camera body, its tag id stored as a value."""
  return f"{camera(slot)}/base_tag"


def joint_model(slot: str, joint: str) -> str:
  """What the model says about one revolute joint."""
  return f"{gripper_part(slot, joint)}/model"


def joint_tag(slot: str, joint: str) -> str:
  """The tag riding one revolute joint, on the link it turns with."""
  return f"{gripper_part(slot, joint)}/tag"


def state_frame(slot: str) -> str:
  """The keyframe pose's frame, apart from the camera's: Rerun defines a frame once."""
  return f"{slot}_state_body"


def state(slot: str, quantity: str) -> str:
  """One quantity of the solved state at keyframes: ``pose`` or ``lin_vel``."""
  return f"{camera(slot)}/state/{quantity}"


def gripper(slot: str) -> str:
  """The gripper this camera rides, in the world frame."""
  return f"{camera(slot)}/gripper"


def gripper_part(slot: str, part: str) -> str:
  """A named part under the gripper entity, such as a finger joint or ``tf_static``."""
  return f"{gripper(slot)}/{part}"


def fit(slot: str, metric: str | None = None) -> str:
  """Whole-session fit numbers, or with ``metric`` that metric's per-keyframe series."""
  return f"{camera(slot)}/fit/{metric}" if metric else f"{camera(slot)}/fit"


def landmark(slot: str, name: str) -> str:
  """The map one lens contributes, in the world frame it names for itself."""
  return f"{camera(slot)}/landmark/{name}"


def factor(slot: str, kind: str) -> str:
  """The rays of a factor that fired, in the world frame it names for itself."""
  return f"{camera(slot)}/factor/{kind}"


def trajectory(slot: str) -> str:
  return f"{camera(slot)}/trajectory"


def anchor(tag_id: int) -> str:
  """A world-static tag, addressed by the tag it is."""
  return f"{WORLD}/anchor/{tag_id}"


def slot_of(path: str) -> str | None:
  """The camera slot an entity path belongs to, or ``None``."""
  parts = path.strip("/").split("/")
  return parts[1] if len(parts) > 1 and parts[0] == CAMERA_ROOT else None


def slots_in(paths: Iterable[str]) -> list[str]:
  """The camera slots a recording holds, read off its entity paths."""
  return sorted({slot for slot in map(slot_of, paths) if slot})


# Stored component spellings, addressed as "/entity/path:Component:field".
SCALARS = "Scalars:scalars"
POSE_TRANSLATION = "Transform3D:translation"
POSE_QUATERNION = "Transform3D:quaternion"


def selector(entity: str, component: str) -> str:
  """One field of one entity, fully qualified."""
  return f"/{entity}:{component}"
