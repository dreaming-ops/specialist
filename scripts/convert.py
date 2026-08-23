"""Convert one session's ``.insv`` captures into a base recording.

Usage:
    uv run python scripts/convert.py --input-insvs camA_00.insv camB_00.insv \
        --session-id <session-id>
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from specialist.loader.audio_sync import sync_two_captures
from specialist.loader.capture import probe_capture
from specialist.loader.parse_insv import load_insv
from specialist.record import schema
from specialist.record.base import (
  BODY_R_IMU,
  calibration_chunks,
  exposure_chunks,
  imu_chunks,
  write_base_recording,
)
from specialist.record.video import normalize_h264, video_chunks


@dataclass
class Args:
  """Base-recording conversion of one session."""

  # One _00_ master .insv per capture; the first defines the session clock.
  input_insvs: tuple[Path, ...]
  # Recording id shared by this session's base recording and track layers.
  session_id: str
  # Dataset root; the base recording lands as ``base/<session-id>.rrd``.
  output_dir: Path = Path("logs")
  # Seconds to add to each capture's clock, for clips that share no audible event.
  audio_offsets: tuple[float, ...] = ()
  # Replace an existing base recording.
  overwrite: bool = False

  # Clamp the session to its first N seconds, for short test conversions.
  duration: float | None = None
  # The IMU mounting rotation, stored with each capture's calibration for the solve.
  body_R_imu: tuple[tuple[float, float, float], ...] = tuple(map(tuple, BODY_R_IMU))


def main(args: Args) -> None:
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  names = [f"tracker_{i}" for i in range(len(args.input_insvs))]
  out_path = args.output_dir / "base" / f"{args.session_id}.rrd"
  if out_path.exists() and not args.overwrite:
    raise SystemExit(
      f"{out_path} exists; pass --overwrite to replace it (a registered base "
      f"must then be re-registered with register.py --replace-base "
      f"{args.session_id}, or its byte-range index goes stale)"
    )
  captures = [probe_capture(master) for master in args.input_insvs]
  if args.audio_offsets:
    if len(args.audio_offsets) != len(args.input_insvs):
      raise SystemExit("--audio-offsets needs one value per --input-insvs")
    offsets = list(args.audio_offsets)
    print(f"audio offsets given: {offsets}")
  else:
    offsets = [0.0]
    for later in args.input_insvs[1:]:
      sync = sync_two_captures(args.input_insvs[0], later)
      offsets.append(sync.offset)
      print(
        f"audio sync: {later.name} offset={sync.offset:+.3f}s "
        f"peak_ratio={sync.peak_ratio:.2f}"
      )

  window = (
    max(-off for off in offsets),
    min(cap.duration - off for cap, off in zip(captures, offsets, strict=True)),
  )
  if args.duration is not None:
    window = (window[0], min(window[1], window[0] + args.duration))
  print(f"session window: [{window[0]:.3f}, {window[1]:.3f}] s")

  video_streams = []
  sidecars = []
  body_R_imu = np.asarray(args.body_R_imu, dtype=np.float64)
  with tempfile.TemporaryDirectory(prefix="remux-") as tmp:
    for master, capture, tracker, offset in zip(
      args.input_insvs, captures, names, offsets, strict=True
    ):
      insv = load_insv(master)
      for lens, src in (("front", master), ("rear", capture.rear_path)):
        mp4 = normalize_h264(src, Path(tmp) / f"{tracker}_{lens}.mp4")
        video_streams.append(
          video_chunks(
            mp4,
            schema.image_entity(tracker, lens),
            time_shift=-offset,
            window=window,
          )
        )
      sidecars += imu_chunks(tracker, insv, time_shift=-offset, window=window)
      sidecars += exposure_chunks(tracker, insv, time_shift=-offset, window=window)
      sidecars += calibration_chunks(tracker, insv, body_R_imu)

    properties: dict[str, dict[str, Any]] = {
      "session": {"sources": [p.name for p in args.input_insvs]},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_base_recording(out_path, args.session_id, video_streams, sidecars, properties)
  print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main(tyro.cli(Args))
