"""Probe a capture: a square ``_00_`` master ``.insv`` and its ``_10_`` rear sibling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

__all__ = ["CaptureInfo", "probe_capture"]


@dataclass(frozen=True)
class CaptureInfo:
  """A capture pair's probe; ``duration`` is the seconds both lens files cover."""

  rear_path: Path
  fps: float
  duration: float


def _video_props(path: Path, *props: int) -> list[float]:
  cap = cv2.VideoCapture(str(path))
  if not cap.isOpened():
    raise RuntimeError(f"OpenCV cannot open {path}")
  values = [float(cap.get(prop)) for prop in props]
  cap.release()
  return values


def probe_capture(video_path: Path) -> CaptureInfo:
  """Probe both lens files; raises RuntimeError on a non-square master or no sibling."""
  raw_w, raw_h, fps, front_frames = _video_props(
    video_path,
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS,
    cv2.CAP_PROP_FRAME_COUNT,
  )
  width, height = int(raw_w), int(raw_h)
  rear_path = video_path.with_name(video_path.name.replace("_00_", "_10_", 1))
  if width != height or rear_path == video_path or not rear_path.is_file():
    raise RuntimeError(
      f"{video_path.name} ({width}x{height}) is not a square per-lens master "
      f"with sibling {rear_path.name}"
    )
  rear_frames = _video_props(rear_path, cv2.CAP_PROP_FRAME_COUNT)[0]
  return CaptureInfo(
    rear_path=rear_path,
    fps=fps,
    duration=min(int(front_frames), int(rear_frames)) / fps,
  )
