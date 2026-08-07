"""Video-IMU offset and rolling-shutter probe; it needs a capture with yaw motion."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr
import tyro

if __package__ in (None, ""):  # `python scripts/x.py` puts scripts/ itself first
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._probe import finalize_probe, verdict
from specialist.loader.audio_sync import MIN_PEAK_RATIO, correlation_offset
from specialist.loader.parse_insv import load_insv
from specialist.vi_ba.features import detect_corners
from specialist.vi_ba.pipeline import probe_capture

Float64Array = npt.NDArray[np.float64]
UInt8Array = npt.NDArray[np.uint8]


@dataclass
class TimingArgs:
  """Video-IMU timing check of one capture."""

  input_insv: Path
  output_dir: Path = Path("logs")
  lens: Literal["front", "rear"] = "front"  # _00_ file / _10_ sibling
  max_frames: int = 600
  flow_max_corners: int = 200
  flow_quality_level: float = 0.01
  flow_min_distance: float = 8.0
  shear_min_yaw_rate: float = math.radians(30.0)
  hough_threshold: int = 80


@dataclass
class ShearResult:
  measured_readout: float
  factory_readout: float
  n_frames_used: int
  median_yaw_rate: float


@dataclass
class TimingSummary:
  source: str
  n_video_frames_used: int
  fps: float
  imu_rate: float
  offset: float
  peak_ratio: float
  shear: ShearResult | None
  verdict: str = "unknown"


def read_lens_grays(
  video_path: Path, lens: Literal["front", "rear"], *, max_frames: int
) -> tuple[list[UInt8Array], Float64Array, float]:
  """Gray frames, their container times and the fps of ``video_path`` or, for ``rear``, its ``_10_`` sibling."""
  rear_path, fps, _ = probe_capture(video_path)
  path = video_path if lens == "front" else rear_path
  cap = cv2.VideoCapture(str(path))
  if not cap.isOpened():
    raise RuntimeError(f"OpenCV failed to open {path}")
  try:
    grays: list[UInt8Array] = []
    times: list[float] = []
    while len(grays) < max_frames:
      ok, frame = cap.read()
      if not ok:
        break
      grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8))
      times.append(cap.get(cv2.CAP_PROP_POS_MSEC) * 1e-3)
  finally:
    cap.release()
  return grays, np.asarray(times, dtype=np.float64), fps


def _flow_magnitudes(grays: list[UInt8Array], args: TimingArgs) -> Float64Array:
  """Median Lucas-Kanade flow magnitude per frame transition."""

  def corners(img: UInt8Array) -> npt.NDArray[np.float32]:
    points = detect_corners(
      img,
      None,
      max_corners=args.flow_max_corners,
      quality_level=args.flow_quality_level,
      min_distance=args.flow_min_distance,
    )
    return points.reshape(-1, 1, 2).astype(np.float32)

  mags: list[float] = []
  prev, prev_points = grays[0], corners(grays[0])
  for curr in grays[1:]:
    # A frame down to a handful of points measures noise, not motion.
    if len(prev_points) < 10:
      mags.append(0.0)
    else:
      next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev, curr, prev_points, np.empty_like(prev_points)
      )
      good = status.flatten() == 1
      delta = (next_points[good] - prev_points[good]).reshape(-1, 2)
      norms = np.linalg.norm(delta, axis=1)
      mags.append(float(np.median(norms)) if norms.size else 0.0)
    prev, prev_points = curr, corners(curr)
  return np.asarray([0.0, *mags], dtype=np.float64)


def _estimate_offset(
  video_t: Float64Array,
  flow_mag: Float64Array,
  imu_t: Float64Array,
  gyro_mag: Float64Array,
) -> tuple[float, float]:
  """Clock offset (s) to add to the IMU timestamps to land them on the video clock, and its peak ratio."""
  rate = round(1.0 / float(np.median(np.diff(imu_t))))
  grid = np.arange(max(video_t[0], imu_t[0]), min(video_t[-1], imu_t[-1]), 1.0 / rate)
  return correlation_offset(
    np.interp(grid, imu_t, gyro_mag), np.interp(grid, video_t, flow_mag), rate
  )


def _vertical_line_shear(gray: UInt8Array, threshold: int) -> float | None:
  """Median per-pixel horizontal shift along near-vertical Hough line segments."""
  edges = cv2.Canny(gray, 60, 180)
  segments = cv2.HoughLinesP(
    edges, 1, np.pi / 360.0, threshold, minLineLength=80, maxLineGap=10
  )
  if segments is None:
    return None
  shifts: list[float] = []
  for x1, y1, x2, y2 in segments.reshape(-1, 4).astype(np.float64):
    dy = y2 - y1
    if abs(dy) >= 60 and abs(x2 - x1) <= np.tan(np.radians(25.0)) * abs(dy):
      shifts.append((x2 - x1) / dy)
  if not shifts:
    return None
  return float(np.median(shifts))


def _estimate_rolling_shutter(
  grays: list[UInt8Array],
  video_t: Float64Array,
  imu_t: Float64Array,
  gyro: Float64Array,
  *,
  args: TimingArgs,
  applied_offset: float,
  factory_readout: float,
) -> ShearResult | None:
  """Per frame, the readout time (s) that turns the yaw rate into the measured line shear."""
  height, width = grays[0].shape[:2]
  # Per pixel, a ~190 deg fisheye spread over the full lens-frame width.
  pixel_angle = math.radians(190.0) / width

  yaw_imu = gyro[:, 1]  # yaw axis after telemetry-parser normalization
  imu_t_shifted = imu_t + applied_offset

  readouts: list[float] = []
  used_yaw_rates: list[float] = []
  for k, gray in enumerate(grays):
    yaw_rate = float(np.interp(video_t[k], imu_t_shifted, yaw_imu))
    if abs(yaw_rate) < args.shear_min_yaw_rate:
      continue
    shear = _vertical_line_shear(gray, args.hough_threshold)
    if shear is None:
      continue
    image_rate = yaw_rate / pixel_angle
    if abs(image_rate) < 1.0:
      continue
    # shear (dx/dy) = image_rate * readout / height.
    readout = shear * height / image_rate
    if 0.0 < readout < 0.05:
      readouts.append(readout)
      used_yaw_rates.append(yaw_rate)

  if not readouts:
    return None
  return ShearResult(
    measured_readout=float(np.median(readouts)),
    factory_readout=factory_readout,
    n_frames_used=len(readouts),
    median_yaw_rate=float(np.median(np.abs(used_yaw_rates))),
  )


def _verdict(summary: TimingSummary) -> str:
  parts: list[str] = []
  if summary.peak_ratio < MIN_PEAK_RATIO:
    parts.append(f"offset unreliable: peak {summary.peak_ratio:.2f}x its tallest rival")
  elif abs(summary.offset) > 0.05:
    parts.append(f"large video<->IMU offset {summary.offset * 1e3:.1f} ms")
  if summary.shear is None:
    parts.append("RS shear unmeasured (need yaw motion + vertical edges)")
  elif abs(summary.shear.measured_readout - summary.shear.factory_readout) > 5e-3:
    parts.append(
      f"RS readout mismatch: {summary.shear.measured_readout * 1e3:.2f} ms vs "
      f"{summary.shear.factory_readout * 1e3:.2f} ms factory"
    )
  return verdict(parts)


def _log_to_rerun(
  video_t: Float64Array,
  flow_mag: Float64Array,
  imu_t: Float64Array,
  gyro_mag: Float64Array,
  applied_offset: float,
) -> None:
  rr.send_columns(
    "flow/median_magnitude",
    indexes=[rr.TimeColumn("t", duration=video_t)],
    columns=rr.Scalars.columns(scalars=flow_mag),
  )
  rr.send_columns(
    "gyro/magnitude",
    indexes=[rr.TimeColumn("t", duration=imu_t + applied_offset)],
    columns=rr.Scalars.columns(scalars=gyro_mag),
  )


def main(args: TimingArgs) -> None:
  data = load_insv(args.input_insv)

  grays, video_t, fps = read_lens_grays(
    args.input_insv, args.lens, max_frames=args.max_frames
  )
  if len(grays) < 5:
    raise RuntimeError(f"only {len(grays)} frames decoded; the capture is too short")

  flow_mag = _flow_magnitudes(grays, args)
  gyro_mag = np.linalg.norm(data.gyro, axis=1)
  imu_rate = 1.0 / float(np.median(np.diff(data.imu_timestamps)))

  offset, peak_ratio = _estimate_offset(
    video_t, flow_mag, data.imu_timestamps, gyro_mag
  )
  shear = _estimate_rolling_shutter(
    grays,
    video_t,
    data.imu_timestamps,
    data.gyro,
    args=args,
    applied_offset=offset,
    factory_readout=data.rolling_shutter_readout,
  )

  summary = TimingSummary(
    source=str(args.input_insv),
    n_video_frames_used=len(grays),
    fps=fps,
    imu_rate=imu_rate,
    offset=offset,
    peak_ratio=peak_ratio,
    shear=shear,
  )
  summary.verdict = _verdict(summary)

  finalize_probe(
    "video_imu_timing",
    args.output_dir,
    summary,
    lambda: _log_to_rerun(video_t, flow_mag, data.imu_timestamps, gyro_mag, offset),
  )


if __name__ == "__main__":
  main(tyro.cli(TimingArgs))
