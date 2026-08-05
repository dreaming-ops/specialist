"""The single source of record entity paths and timeline names."""

from __future__ import annotations

APPLICATION_ID = "specialist"
TIMELINE = "video_time"


def image_root(tracker: str) -> str:
  return f"image/{tracker}"


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


def body_entity(tracker: str) -> str:
  return f"{tracker}/body"


def body_keyframes_entity(tracker: str) -> str:
  return f"{tracker}/body_keyframes"


def base_tag_entity(tracker: str) -> str:
  return f"{body_entity(tracker)}/base_tag"
