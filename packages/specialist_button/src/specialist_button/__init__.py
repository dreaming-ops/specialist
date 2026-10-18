"""Acoustic marker frames that name a talking button and an event: generator and decoder."""

from specialist_button.decoder import DEFAULT_CONFIG, DecoderConfig, Detection, decode
from specialist_button.frame import (
  FRAME_DURATION,
  FS,
  N_EVENTS,
  N_SOURCES,
  generate_frame,
)

__all__ = [
  "DEFAULT_CONFIG",
  "FRAME_DURATION",
  "FS",
  "N_EVENTS",
  "N_SOURCES",
  "DecoderConfig",
  "Detection",
  "decode",
  "generate_frame",
]
