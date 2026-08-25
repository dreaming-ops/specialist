"""The single source of record entity paths, timeline names and timeline decoding."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pyarrow as pa

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

# The two sides of a stitched dual-fisheye capture.
LensName = Literal["front", "rear"]
LENSES: tuple[LensName, LensName] = ("front", "rear")

APPLICATION_ID = "specialist"
TIMELINE = "video_time"
# Sequence timeline: frame ``i`` carries keyframe index ``ceil(i / stride)``.
KEYFRAME_TIMELINE = "keyframe"
NS_PER_SECOND = 1_000_000_000

CAMERA_ROOT = "camera"
WORLD = "world"
# The frame a world-frame entity under a posed parent must declare.
WORLD_FRAME = "world"
ESTIMATE = "estimate"


def camera_slots(serials: Sequence[str]) -> list[str]:
  """Slots by serial rank in the take; refuses missing or shared serials."""
  if not all(serials) or len(set(serials)) != len(serials):
    raise ValueError(f"cameras need one distinct serial each, got {list(serials)}")
  rank = {serial: i for i, serial in enumerate(sorted(serials))}
  return [f"cam{rank[serial]:02d}" for serial in serials]


def keyframe_stride(fps: float) -> int:
  """The auto keyframe interval in frames, ``max(1, round(fps / 10))`` (about 0.1 s)."""
  return max(1, round(fps / 10.0))


def keyframe_index(session_frames: Int64Array, stride: int) -> Int64Array:
  """The ``keyframe`` timeline value ``ceil(row / stride)`` per session frame row, equal
  to the solver's ``kf`` only at the auto stride and phase zero."""
  return (session_frames + stride - 1) // stride


def times_ns(column: pa.Array | pa.ChunkedArray) -> Int64Array:
  """A duration column as int64 nanoseconds; the numpy read raises if it would copy."""
  values = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
  out: Int64Array = values.to_numpy(zero_copy_only=True).astype("int64")
  return out


def seconds(column: pa.Array | pa.ChunkedArray) -> Float64Array:
  out: Float64Array = times_ns(column) / NS_PER_SECOND
  return out


def camera(slot: str) -> str:
  """The camera body frame; world-frame entities under it must name ``WORLD_FRAME``."""
  return f"{CAMERA_ROOT}/{slot}"


def lens(slot: str, name: str) -> str:
  """One lens: its pose in the camera body and its projection model."""
  return f"{camera(slot)}/{name}"


def video(slot: str, name: str) -> str:
  """A lens's encoded frames; a leaf that must hold no temporal children."""
  return f"{lens(slot, name)}/video"


TAGS_DETECTION = "tags"
KEYPOINTS_DETECTION = "keypoints"
REJECTED_KEYPOINTS_DETECTION = "keypoints_rejected"
DETECTION_KINDS = (TAGS_DETECTION, KEYPOINTS_DETECTION, REJECTED_KEYPOINTS_DETECTION)

TAG_AXES_OVERLAY = "tag_axes"
LANDMARK_REPROJECTION_OVERLAY = "landmark_reprojection"
STATIC_MASK_OVERLAY = "static_mask"
OVERLAY_KINDS = (TAG_AXES_OVERLAY, LANDMARK_REPROJECTION_OVERLAY, STATIC_MASK_OVERLAY)


def detection(slot: str, name: str, kind: str) -> str:
  """What the lens measured in its own pixels: the pixels a factor reads."""
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
  """The gripper tag bolted to the camera, a child of the body it is measured from."""
  return f"{camera(slot)}/base_tag"


def joint_model(slot: str, joint: str) -> str:
  """What the model says about one revolute joint, on a child of the link."""
  return f"{gripper_part(slot, joint)}/model"


def joint_tag(slot: str, joint: str) -> str:
  """The tag riding one revolute joint, on the link it turns with."""
  return f"{gripper_part(slot, joint)}/tag"


def state_frame(slot: str) -> str:
  """The body frame the solved keyframe pose defines, apart from the propagated one."""
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
  """The camera's whole-take solve quality, and one per-keyframe series per child."""
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
  """A world-static tag by id; a component on it says whether it pins the gauge."""
  return f"{WORLD}/anchor/{tag_id}"


def anchor_id_of(path: str) -> int | None:
  """The tag an anchor path addresses, or ``None`` when the path is not one."""
  parts = path.strip("/").split("/")
  if len(parts) != 3 or parts[:2] != [WORLD, "anchor"]:
    return None
  tag = parts[2]
  return int(tag) if tag.isascii() and tag.isdigit() else None


def anchor_ids_in(paths: Iterable[str]) -> list[int]:
  """The world-static tags a recording holds, read off its entity paths."""
  return sorted({tag for tag in map(anchor_id_of, paths) if tag is not None})


def slot_of(path: str) -> str | None:
  """The camera slot an entity path belongs to, or ``None``."""
  parts = path.strip("/").split("/")
  return parts[1] if len(parts) > 1 and parts[0] == CAMERA_ROOT else None


def slots_in(paths: Iterable[str]) -> list[str]:
  """The camera slots a recording holds, read off its entity paths."""
  return sorted({slot for slot in map(slot_of, paths) if slot})
