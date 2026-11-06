"""The solve's view of a base recording: slots, video, IMU and calibration."""

import shutil
import tempfile
from pathlib import Path

import gtsam
import numpy as np
import numpy.typing as npt
from rerun.chunk import RrdReader
from specialist_camera import LensName
from specialist_dataset import frames, schema
from specialist_dataset.poses import quats_to_matrices
from specialist_dataset.read import Recording, lens_from_row
from specialist_dataset.state import nearest

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.imu import ImuSamples

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

# Radius, as a fraction of the decoded width, past which an anchor corner's noise grows.
_CALIBRATED_RADIUS_FRAC = 1000.0 / 2880.0

__all__ = [
  "SessionRecording",
  "covered_samples",
  "fps_from_times",
  "nearest_samples",
]


def fps_from_times(times: Float64Array) -> float:
  """Frame rate from per-frame times, since the telemetry rounds NTSC rates."""
  if len(times) < 2:
    raise ValueError(f"need at least two frame times, got {len(times)}")
  return (len(times) - 1) / float(times[-1] - times[0])


def nearest_samples(times: Float64Array, grid: Float64Array) -> Int64Array:
  """The sample of ``times`` nearest each ``grid`` time; raises beyond half a frame."""
  if len(times) < 2:
    raise ValueError(f"need at least two frame times, got {len(times)}")
  samples, deltas = nearest(times, grid)
  half_frame = 0.5 * float(np.median(np.diff(times)))
  if float(deltas.max()) > half_frame:
    raise ValueError(
      f"nearest-PTS match misses the grid by {float(deltas.max()):.4f} s, "
      f"more than half a frame ({half_frame:.4f} s)"
    )
  return samples


def covered_samples(
  front: Float64Array, rear: Float64Array, imu: Float64Array, fps: float
) -> Int64Array:
  """The front-lens samples inside the span front, rear and IMU all cover."""
  half_frame = 0.5 / fps
  lo = max(front[0], rear[0], imu[0]) - half_frame
  hi = min(front[-1], rear[-1], imu[-1]) + half_frame
  samples: Int64Array = np.flatnonzero((front >= lo) & (front <= hi)).astype(np.int64)
  return samples


class SessionRecording:
  """One session's base recording: the solve's only input."""

  def __init__(self, base_rrd: Path, stage: Path | None = None) -> None:
    self.path = base_rrd
    recordings = RrdReader(str(base_rrd)).recordings()
    if not recordings:
      raise ValueError(f"{base_rrd} holds no recording")
    self.session_id: str = recordings[0].recording_id
    self._recording = Recording(base_rrd)
    self._stage = stage
    self._owns_stage = stage is None

  @property
  def stage(self) -> Path:
    """The directory the lenses' encoded samples stage in."""
    if self._stage is None:
      self._stage = Path(tempfile.mkdtemp(prefix="specialist-lenses-"))
    return self._stage

  def close(self) -> None:
    """Release every loaded lens and its samples, and a stage this recording made."""
    frames.release_videos()
    if self._owns_stage and self._stage is not None:
      shutil.rmtree(self._stage, ignore_errors=True)
      self._stage = None

  @property
  def slots(self) -> list[str]:
    """The camera slots this recording holds video for."""
    paths = self._recording.entity_paths()
    slots = [
      slot
      for slot in schema.slots_in(paths)
      if any(schema.video(slot, lens) in paths for lens in schema.LENSES)
    ]
    if not slots:
      raise ValueError(f"{self.path} holds no /{schema.CAMERA_ROOT}/<slot> video")
    return slots

  def lens_times(self, slot: str, lens: str) -> Float64Array:
    """One lens's frame times (s), one per encoded sample."""
    return frames.lens_times(self._recording, schema.video(slot, lens))

  def lens_video(self, slot: str, lens: str) -> frames.LensVideo:
    """One lens's encoded samples, ready for banded decoding."""
    return frames.lens_video(self._recording, slot, lens, self.stage)

  def decoded_shape(self, slot: str, lens: LensName = "front") -> tuple[int, int]:
    """``(height, width)`` of one lens's decoded frames, from a one-frame decode."""
    frame = frames.LensFrames(
      self.lens_video(slot, lens), np.zeros(1, dtype=np.int64), band=1
    )[0]
    return int(frame.shape[0]), int(frame.shape[1])

  def lens_setups(self, slot: str, decoded_width: int) -> tuple[LensSetup, LensSetup]:
    """Both lenses at the decode resolution, as the solve holds them."""
    return (
      self._lens_setup(slot, "front", decoded_width),
      self._lens_setup(slot, "rear", decoded_width),
    )

  def _lens_setup(self, slot: str, lens: LensName, decoded_width: int) -> LensSetup:
    calibration = lens_from_row(
      self._recording.row(schema.lens(slot, lens)), decoded_width, lens
    )
    return LensSetup(
      body_T_lens=gtsam.Pose3(calibration.body_T_lens),
      params=calibration.params,
      calibrated_radius=_CALIBRATED_RADIUS_FRAC * decoded_width,
    )

  def imu(self, slot: str) -> ImuSamples:
    """Raw-axis IMU samples as stored: t (s), gyro (rad/s), accel (m/s^2)."""
    t_gyro, gyro = self._recording.scalars(schema.imu_signal(slot, "gyro"))
    t_accel, accel = self._recording.scalars(schema.imu_signal(slot, "accel"))
    if not np.array_equal(t_gyro, t_accel):
      raise ValueError(f"{self.path}: gyro and accel rows disagree for {slot}")
    return ImuSamples(t=t_gyro, gyro=gyro, accel=accel)

  @property
  def converted_at(self) -> int:
    """The base recording's write time, in nanoseconds since the epoch."""
    return int(np.asarray(self._recording.property("RecordingInfo")["start_time"])[0])

  def body_R_imu(self, slot: str) -> Float64Array:
    row = self._recording.row(schema.imu(slot))
    matrix: Float64Array = quats_to_matrices(row[schema.POSE_QUATERNION])[0]
    return matrix

  def factory_bias(self, slot: str) -> gtsam.imuBias.ConstantBias:
    """The factory IMU bias in raw IMU axes (accel m/s^2, gyro rad/s)."""
    row = self._recording.row(schema.imu(slot))
    accel = np.asarray(row["accel_bias"], dtype=np.float64).reshape(3)
    gyro = np.asarray(row["gyro_bias"], dtype=np.float64).reshape(3)
    return gtsam.imuBias.ConstantBias(accel, gyro)
