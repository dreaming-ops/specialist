"""Base-recording assembly: video, IMU and calibration in one rrd per session."""

import dataclasses
from pathlib import Path
from typing import Any, cast

import gtsam
import numpy as np
import numpy.typing as npt
import rerun as rr
from rerun.chunk import Chunk, LazyChunkStream, OptimizationProfile
from specialist_camera import MeiParams
from specialist_dataset import schema

from specialist.loader.parse_insv import InsvData
from specialist.record.offset_v3 import LensOffset, parse_offset_v3
from specialist.vi_ba.geometry import quat_xyzw

Float64Array = npt.NDArray[np.float64]


def imu_chunks(
  slot: str,
  insv: InsvData,
  time_shift: float,
  window: tuple[float, float],
) -> list[Chunk]:
  """Gyro (rad/s) and accel (m/s^2) at native times, trimmed row-exact to ``window``."""
  shifted = insv.imu_timestamps + time_shift
  keep = (shifted >= window[0]) & (shifted <= window[1])
  times = rr.TimeColumn(schema.TIMELINE, duration=shifted[keep])
  gyro = insv.gyro[keep]
  return [
    Chunk.from_columns(
      schema.imu_signal(slot, "gyro"),
      indexes=[times],
      columns=rr.Scalars.columns(scalars=gyro),
    ),
    Chunk.from_columns(
      schema.imu_signal(slot, "accel"),
      indexes=[times],
      columns=rr.Scalars.columns(scalars=insv.accel[keep]),
    ),
  ]


def _lens_calibration(entity: str, offset: LensOffset, pose: gtsam.Pose3) -> Chunk:
  return Chunk.from_columns(
    entity,
    indexes=[],
    columns=[
      *rr.Transform3D.columns(
        translation=np.asarray(pose.translation(), dtype=np.float64)[None, :],
        quaternion=quat_xyzw(pose.rotation())[None, :],
      ),
      *cast("Any", rr.AnyValues).columns(
        **{f.name: [getattr(offset, f.name)] for f in dataclasses.fields(MeiParams)},
        frame_width=[float(offset.frame_width)],
        frame_height=[float(offset.frame_height)],
      ),
    ],
  )


BODY_R_IMU: Float64Array = np.array(
  [(0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)], dtype=np.float64
)


def exposure_chunks(
  slot: str,
  insv: InsvData,
  time_shift: float,
  window: tuple[float, float],
) -> list[Chunk]:
  """Exposure durations and auto-exposure reports, trimmed row-exact to ``window``."""
  chunks: list[Chunk] = []
  if insv.exposure_timestamps.size:
    shifted = insv.exposure_timestamps + time_shift
    keep = (shifted >= window[0]) & (shifted <= window[1])
    chunks.append(
      Chunk.from_columns(
        schema.exposure(slot),
        indexes=[rr.TimeColumn(schema.TIMELINE, duration=shifted[keep])],
        columns=rr.Scalars.columns(scalars=insv.exposure_durations[keep]),
      )
    )
  if insv.ae_timestamps.size:
    shifted = insv.ae_timestamps + time_shift
    keep = (shifted >= window[0]) & (shifted <= window[1])
    chunks.append(
      Chunk.from_columns(
        schema.autoexposure(slot),
        indexes=[rr.TimeColumn(schema.TIMELINE, duration=shifted[keep])],
        columns=cast("Any", rr.AnyValues).columns(
          **{name: values[keep] for name, values in insv.ae_stats.items()}
        ),
      )
    )
  return chunks


def calibration_chunks(slot: str, insv: InsvData) -> list[Chunk]:
  """Each lens's raw-unit MEI intrinsics and body_T_lens, and the IMU mounting."""
  offset_v3 = parse_offset_v3(insv.offset_v3)
  rotation = gtsam.Rot3(BODY_R_IMU)
  return [
    _lens_calibration(
      schema.lens(slot, "front"),
      offset_v3.front,
      offset_v3.body_T_front_lens,
    ),
    _lens_calibration(
      schema.lens(slot, "rear"),
      offset_v3.rear,
      offset_v3.body_T_rear_lens,
    ),
    Chunk.from_columns(
      schema.imu(slot),
      indexes=[],
      columns=[
        *rr.Transform3D.columns(
          translation=np.zeros((1, 3)), quaternion=quat_xyzw(rotation)[None, :]
        ),
        *rr.AnyValues.columns(
          rolling_shutter_readout=[insv.rolling_shutter_readout],
          gyro_bias=[insv.gyro_calib_numbers[:3]],
          accel_bias=[insv.gyro_calib_numbers[3:]],
        ),
      ],
    ),
  ]


def capture_chunk(
  slot: str,
  insv: InsvData,
  source: str,
  clock_shift: float,
  clock_confidence: float,
) -> Chunk:
  """The camera's metadata and its shift ``t_session = t_capture + clock_shift``."""
  return Chunk.from_columns(
    schema.camera(slot),
    indexes=[],
    columns=cast("Any", rr.AnyValues).columns(
      source=[source],
      clock_shift=[clock_shift],
      clock_confidence=[clock_confidence],
      **{name: [value] for name, value in insv.capture.items()},
    ),
  )


def write_base_recording(
  out_path: Path,
  session_id: str,
  video_streams: list[LazyChunkStream],
  sidecar_chunks: list[Chunk],
) -> None:
  """Merge all streams, optimize for storage, and write one base rrd."""
  merged = LazyChunkStream.merge(
    *video_streams, LazyChunkStream.from_iter(sidecar_chunks)
  )
  # fix_keyframe re-derives the markers, as the GOP trim keeps the marker chunk whole.
  optimize = dataclasses.replace(OptimizationProfile.OBJECT_STORE, fix_keyframe=True)
  store = merged.collect(optimize=optimize)
  partial = out_path.parent / (out_path.name + ".partial")
  partial.unlink(missing_ok=True)
  try:
    with rr.RecordingStream(schema.APPLICATION_ID, recording_id=session_id) as stream:
      stream.save(str(partial))
      rr.send_chunks(store, recording=stream)
    partial.replace(out_path)
  finally:
    partial.unlink(missing_ok=True)
