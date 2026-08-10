"""The solve's view of a base recording: cameras, video, IMU and calibration."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import gtsam
import numpy as np
import numpy.typing as npt
from rerun.experimental import Chunk, RrdReader

from specialist.loader.mei_model import MeiParams
from specialist.record import frames, schema
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.imu import ImuSamples

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

__all__ = [
  "SessionRecording",
  "covered_samples",
  "fps_from_times",
  "nearest_samples",
  "newest_values",
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
  pos = np.searchsorted(times, grid)
  lo = np.clip(pos - 1, 0, len(times) - 1)
  hi = np.clip(pos, 0, len(times) - 1)
  samples = np.where(np.abs(times[hi] - grid) < np.abs(times[lo] - grid), hi, lo)
  deltas = np.abs(times[samples] - grid)
  half_frame = 0.5 * float(np.median(np.diff(times)))
  if float(deltas.max()) > half_frame:
    raise ValueError(
      f"nearest-PTS match misses the grid by {float(deltas.max()):.4f} s, "
      f"more than half a frame ({half_frame:.4f} s)"
    )
  out: Int64Array = samples.astype(np.int64)
  return out


def covered_samples(
  front: Float64Array, rear: Float64Array, imu: Float64Array, fps: float
) -> Int64Array:
  """The front-lens samples inside the span front, rear and IMU all cover."""
  half_frame = 0.5 / fps
  lo = max(front[0], rear[0], imu[0]) - half_frame
  hi = min(front[-1], rear[-1], imu[-1]) + half_frame
  samples: Int64Array = np.flatnonzero((front >= lo) & (front <= hi)).astype(np.int64)
  return samples


def newest_values(chunks: Iterable[Chunk]) -> dict[str, Any]:
  """Component name -> the value with the highest row id over ``chunks``."""
  out: dict[str, Any] = {}
  newest: dict[str, bytes] = {}
  for chunk in chunks:
    batch = chunk.to_record_batch()
    names = batch.schema.names
    row_ids = batch.column(names.index("rerun.controls.RowId")).to_pylist()
    for name, column in zip(names, batch.columns, strict=True):
      if name.startswith("rerun.controls.") or name == schema.TIMELINE:
        continue
      for i, value in enumerate(column.to_pylist()):
        if value is not None and row_ids[i] > newest.get(name, b""):
          newest[name] = row_ids[i]
          out[name] = value
  return out


def _rot3(row: dict[str, Any]) -> gtsam.Rot3:
  x, y, z, w = np.asarray(row["Transform3D:quaternion"], dtype=np.float64).reshape(4)
  return gtsam.Rot3.Quaternion(w, x, y, z)


def _pose(row: dict[str, Any]) -> gtsam.Pose3:
  translation = np.asarray(row["Transform3D:translation"], dtype=np.float64).reshape(3)
  return gtsam.Pose3(_rot3(row), gtsam.Point3(*translation))


class SessionRecording:
  """One session's base recording: the solve's only input."""

  def __init__(self, base_rrd: Path) -> None:
    self.path = base_rrd
    recordings = RrdReader(str(base_rrd)).recordings()
    if not recordings:
      raise ValueError(f"{base_rrd} holds no recording")
    self.session_id: str = recordings[0].recording_id
    self._rows: dict[str, dict[str, Any]] = {}

  def _stream(self) -> Any:
    return RrdReader(str(self.path)).store().stream()

  @property
  def slots(self) -> list[str]:
    """Sorted tracker names carrying video under ``/image/``."""
    root = f"/{schema.IMAGE_ROOT}/"
    names = {
      path.split("/")[2]
      for path in RrdReader(str(self.path)).store().schema().entity_paths()
      if path.startswith(root)
    }
    if not names:
      raise ValueError(f"{self.path} holds no {root}<tracker> video")
    return sorted(names)

  def _entity_row(self, entity: str) -> dict[str, Any]:
    """Component name -> newest value across the entity's chunks."""
    cached = self._rows.get(entity)
    if cached is None:
      cached = newest_values(self._stream().filter(content=f"/{entity}").to_chunks())
      self._rows[entity] = cached
    return cached

  @property
  def sources(self) -> list[str]:
    """The raw capture filenames the base recording was converted from."""
    row = self._entity_row("__properties/session")
    return [str(source) for source in row.get("sources", [])]

  def lens_times(self, tracker: str, lens: str) -> Float64Array:
    """One lens's frame times (s), one per encoded sample."""
    return frames.lens_times(self.path, schema.image_entity(tracker, lens))

  @property
  def frame_times(self) -> dict[str, Float64Array]:
    """Per-tracker front-lens frame times (s), the grid keyframes must land on."""
    return {name: self.lens_times(name, "front") for name in self.slots}

  def lens_video(self, tracker: str, lens: str) -> frames.LensVideo:
    """One lens's encoded samples, ready for banded decoding."""
    return frames.LensVideo(self.path, tracker, lens)

  def decoded_shape(self, tracker: str) -> tuple[int, int]:
    """``(height, width)`` of decoded front-lens frames, from a one-frame decode."""
    frame = frames.LensFrames(
      self.lens_video(tracker, "front"), np.zeros(1, dtype=np.int64), band=1
    )[0]
    return int(frame.shape[0]), int(frame.shape[1])

  def lens_setups(
    self, tracker: str, decoded_width: int
  ) -> tuple[LensSetup, LensSetup]:
    """Both lenses at the decode resolution, as the solve holds them."""
    return (
      self._lens_setup(tracker, "front", decoded_width),
      self._lens_setup(tracker, "rear", decoded_width),
    )

  def _lens_setup(self, tracker: str, lens: str, decoded_width: int) -> LensSetup:
    row = self._entity_row(schema.calibration_entity(tracker, lens))

    def scalar(name: str) -> float:
      return float(np.asarray(row[name], dtype=np.float64).reshape(-1)[0])

    # Rear-lens raw ``cx`` is measured from the stitched-frame left edge.
    frame_width = scalar("frame_width")
    scale = 2 * decoded_width / frame_width
    left_edge = frame_width / 2.0 if lens == "rear" else 0.0
    params = MeiParams(
      fx=scalar("fx") * scale,
      fy=scalar("fy") * scale,
      cx=(scalar("cx") - left_edge) * scale,
      cy=scalar("cy") * scale,
      **{name: scalar(name) for name in ("k1", "k2", "k3", "p1", "p2", "xi")},
    )
    return LensSetup(body_T_lens=_pose(row), params=params)

  def _scalar_rows(self, entity: str) -> tuple[Float64Array, Float64Array]:
    """One entity's scalar series sorted by time; raises ValueError when it has none."""
    times: list[Float64Array] = []
    rows: list[Float64Array] = []
    for chunk in self._stream().filter(content=f"/{entity}").to_chunks():
      if chunk.is_static:
        continue
      batch = chunk.to_record_batch()
      names = batch.schema.names
      values = batch.column(names.index("Scalars:scalars"))
      if not len(values):
        continue
      times.append(schema.seconds(batch.column(names.index(schema.TIMELINE))))
      rows.append(
        values.flatten().to_numpy(zero_copy_only=False).reshape(len(values), -1)
      )
    if not rows:
      raise ValueError(f"{self.path} has no samples at /{entity}")
    t = np.concatenate(times)
    order = np.argsort(t, kind="stable")
    return t[order], np.concatenate(rows)[order]

  def imu(self, tracker: str) -> ImuSamples:
    """Raw-axis IMU samples as stored: t (s), gyro (rad/s), accel (m/s^2)."""
    t_gyro, gyro = self._scalar_rows(schema.imu_entity(tracker, "gyro"))
    t_accel, accel = self._scalar_rows(schema.imu_entity(tracker, "accel"))
    if not np.array_equal(t_gyro, t_accel):
      raise ValueError(f"{self.path}: gyro and accel rows disagree for {tracker}")
    return ImuSamples(t=t_gyro, gyro=gyro, accel=accel)

  def body_R_imu(self, tracker: str) -> Float64Array:
    row = self._entity_row(schema.calibration_entity(tracker, "imu"))
    matrix: Float64Array = _rot3(row).matrix()
    return matrix

  def factory_bias(self, tracker: str) -> gtsam.imuBias.ConstantBias:
    """The factory IMU bias in raw IMU axes (accel m/s^2, gyro rad/s)."""
    row = self._entity_row(schema.calibration_entity(tracker, "imu"))
    accel = np.asarray(row["accel_bias"], dtype=np.float64).reshape(3)
    gyro = np.asarray(row["gyro_bias"], dtype=np.float64).reshape(3)
    return gtsam.imuBias.ConstantBias(accel, gyro)

  def rolling_shutter_readout(self, tracker: str) -> float:
    """The rolling-shutter readout time (s) stored with the IMU calibration."""
    row = self._entity_row(schema.calibration_entity(tracker, "imu"))
    return float(np.asarray(row["rolling_shutter_readout"]).reshape(-1)[0])
