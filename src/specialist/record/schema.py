"""The single source of record entity paths, timeline names and timeline decoding."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pyarrow as pa

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

APPLICATION_ID = "specialist"
TIMELINE = "video_time"
# Sequence timeline: frame ``i`` carries keyframe index ``ceil(i / stride)``.
KEYFRAME_TIMELINE = "keyframe"
NS_PER_SECOND = 1_000_000_000

# Entity names under their owning tracker, and the roots shared entities hang off.
IMAGE_ROOT = "image"
BODY = "body"
BODY_KEYFRAMES = "body_keyframes"
BASE_TAG = f"{BODY}/base_tag"
BODY_MARKER = f"{BODY}/marker"
LANDMARKS = "landmarks"
FACTOR_LINES = "factor_lines"
TRAJECTORY = "trajectory"
TF_STATIC = "tf_static"
METRICS_ROOT = "metrics"


def keyframe_stride(fps: float) -> int:
  """The auto keyframe interval in frames, ``max(1, round(fps / 10))`` (about 0.1 s)."""
  return max(1, round(fps / 10.0))


def keyframe_index(frame_rows: Int64Array, stride: int) -> Int64Array:
  """``ceil(row / stride)`` per frame row: the ``keyframe`` timeline value."""
  return (frame_rows + stride - 1) // stride


def times_ns(column: pa.Array | pa.ChunkedArray) -> Int64Array:
  """A duration column as int64 nanoseconds; the numpy read raises if it would copy."""
  values = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
  out: Int64Array = values.to_numpy(zero_copy_only=True).astype("int64")
  return out


def seconds(column: pa.Array | pa.ChunkedArray) -> Float64Array:
  out: Float64Array = times_ns(column) / NS_PER_SECOND
  return out


def image_root(tracker: str) -> str:
  return f"{IMAGE_ROOT}/{tracker}"


def image_entity(tracker: str, lens: str) -> str:
  return f"{image_root(tracker)}/{lens}"


def imu_entity(tracker: str, kind: str) -> str:
  return f"imu/{tracker}/{kind}"


def exposure_entity(tracker: str) -> str:
  return f"{image_root(tracker)}/exposure"


def ae_entity(tracker: str) -> str:
  return f"{image_root(tracker)}/ae"


def calibration_entity(tracker: str, part: str) -> str:
  return f"calibration/{tracker}/{part}"


def body_keyframes_entity(tracker: str) -> str:
  return f"{tracker}/{BODY_KEYFRAMES}"


def base_tag_entity(tracker: str) -> str:
  return f"{tracker}/{BASE_TAG}"
