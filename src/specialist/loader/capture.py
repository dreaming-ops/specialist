"""Probe a capture: a square ``_00_`` master ``.insv`` and its ``_10_`` rear sibling."""

from __future__ import annotations

from pathlib import Path

import cv2

__all__ = ["capture_duration", "probe_capture"]


def _video_props(path: Path, *props: int) -> list[float]:
  cap = cv2.VideoCapture(str(path))
  if not cap.isOpened():
    raise RuntimeError(f"OpenCV cannot open {path}")
  values = [float(cap.get(prop)) for prop in props]
  cap.release()
  return values


def probe_capture(video_path: Path) -> tuple[Path, float, int]:
  """Rear-lens path, container fps and width; RuntimeError unless square and paired."""
  raw_w, raw_h, fps = _video_props(
    video_path, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS
  )
  width, height = int(raw_w), int(raw_h)
  rear_path = video_path.with_name(video_path.name.replace("_00_", "_10_", 1))
  if width != height or rear_path == video_path or not rear_path.is_file():
    raise RuntimeError(
      f"{video_path.name} ({width}x{height}) is not a square per-lens master "
      f"with sibling {rear_path.name}"
    )
  return rear_path, fps, width


def _shared_frame_count(front_path: Path, rear_path: Path) -> int:
  """Frames present in both lens files; one file can end a frame early."""
  return min(
    int(_video_props(path, cv2.CAP_PROP_FRAME_COUNT)[0])
    for path in (front_path, rear_path)
  )


def capture_duration(video_path: Path) -> float:
  """Seconds decodable from a capture, bounded by the shorter lens file."""
  rear_path, fps, _ = probe_capture(video_path)
  return _shared_frame_count(video_path, rear_path) / fps
