"""The state-layer writer that specialist and specialist_sim share."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import rerun as rr
from rerun.experimental import Chunk, LazyChunkStream, OptimizationProfile, send_chunks

from specialist_dataset import schema

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]
# (times, position, xyzw quaternion, lin_vel, ang_vel), in the world frame.
State = tuple[Float64Array, Float64Array, Float64Array, Float64Array, Float64Array]


def encode(image: npt.NDArray[np.uint8]) -> bytes:
  """One RGB frame as the JPEG rerun compresses it to, at quality 95."""
  jpeg = rr.Image(image).compress(jpeg_quality=95)
  if not isinstance(jpeg, rr.EncodedImage) or jpeg.blob is None:
    raise ValueError(f"rerun cannot compress a {image.shape} {image.dtype} frame")
  return bytes(jpeg.blob.as_arrow_array()[0].values.to_numpy())


def clock(times: Float64Array, rows: Int64Array) -> list[rr.TimeColumn]:
  """The session-clock and ``row`` index columns of a series measured at ``times``."""
  ns = np.rint(times * schema.NS_PER_SECOND).astype("timedelta64[ns]")
  return [
    rr.TimeColumn(schema.TIMELINE, duration=ns),
    rr.TimeColumn(schema.ROW_TIMELINE, sequence=rows),
  ]


def hand_chunks(
  hand: str,
  slot: str,
  frames: tuple[Float64Array, Int64Array, Int64Array],
  blobs: list[bytes],
  rows: Int64Array,
  state: State,
  opening_rows: Float64Array,
  width: int,
) -> list[Chunk]:
  """Every chunk the layer holds for one hand, each series at its measured times."""
  t, position, quaternion, lin_vel, ang_vel = state
  frame_times, label, capture = frames
  # One image plane holds both hands, left then right.
  column = float(schema.HANDS.index(hand) * width)
  index = clock(t, rows)
  return [
    Chunk.from_columns(
      schema.state_hand(hand), indexes=[], columns=rr.AnyValues.columns(slot=[slot])
    ),
    Chunk.from_columns(
      schema.state_image(hand),
      indexes=[],
      columns=rr.Transform3D.columns(translation=[[column, 0.0, 0.0]]),
    ),
    *[
      Chunk.from_columns(
        schema.state_image(hand),
        indexes=clock(frame_times[i : i + 1], label[i : i + 1]),
        columns=[
          *rr.EncodedImage.columns(blob=[blob], media_type=["image/jpeg"]),
          *cast("Any", rr.AnyValues).columns(
            **{schema.CAPTURE_ROW: capture[i : i + 1]}
          ),
        ],
      )
      for i, blob in enumerate(blobs)
    ],
    Chunk.from_columns(
      schema.state_pose(hand),
      indexes=index,
      columns=rr.Transform3D.columns(translation=position, quaternion=quaternion),
    ),
    *[
      Chunk.from_columns(
        path(hand), indexes=index, columns=rr.Scalars.columns(scalars=values)
      )
      for path, values in (
        (schema.state_lin_vel, lin_vel),
        (schema.state_ang_vel, ang_vel),
        (schema.state_opening, opening_rows[:, None]),
      )
    ],
  ]


def write_layer(
  path: Path,
  session_id: str,
  account: dict[str, Any],
  chunks: list[Chunk],
) -> None:
  """Write the layer under the session's id, renaming a partial file once complete."""
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.parent / (path.name + ".partial")
  # One chunk per frame, so a random read loads one frame; only the rest is compacted.
  images = {schema.state_image(hand) for hand in schema.HANDS}
  frames = [c for c in chunks if c.entity_path.strip("/") in images and not c.is_static]
  rest = [c for c in chunks if c.entity_path.strip("/") not in images or c.is_static]
  store = LazyChunkStream.from_iter(rest).collect(
    optimize=OptimizationProfile.OBJECT_STORE
  )
  try:
    with rr.RecordingStream(
      schema.APPLICATION_ID, recording_id=session_id, send_properties=False
    ) as stream:
      stream.set_sinks(rr.FileSink(partial))
      stream.send_property(schema.STATE, rr.AnyValues(**account))
      send_chunks(store, recording=stream)
      send_chunks(frames, recording=stream)
    partial.replace(path)
  finally:
    partial.unlink(missing_ok=True)
