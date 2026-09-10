"""Convert one session's ``.insv`` captures into a base recording.

Usage:
    uv run python scripts/convert.py --input-insvs camA_00.insv camB_00.insv
"""

from __future__ import annotations

import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tyro
from specialist_dataset import schema

from specialist.loader.audio_sync import sync_two_captures
from specialist.loader.capture import probe_capture
from specialist.loader.parse_insv import load_insv
from specialist.record.base import (
  calibration_chunks,
  capture_chunk,
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
  # Dataset root; the base recording lands as ``base/<session-id>.rrd``.
  output_dir: Path = Path("logs")
  # Replace an existing base recording.
  overwrite: bool = False

  # Clamp the session to its first N seconds, for short test conversions.
  duration: float | None = None


_CAPTURE_STAMP = re.compile(r"_(\d{8}_\d{6})_")


def session_name(master: Path, serial: str) -> str:
  """The instant the clock-defining capture started, and the device that took it."""
  stamp = _CAPTURE_STAMP.search(master.name)
  if not stamp or not serial:
    raise SystemExit(
      f"{master.name} (serial {serial!r}) carries no capture instant or "
      "serial to name the session from"
    )
  return f"{stamp[1]}_{serial}"


def main(args: Args) -> None:
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  loaded = [load_insv(master) for master in args.input_insvs]
  serials = [str(insv.capture.get("serial_number", "")) for insv in loaded]
  slots = schema.camera_slots(serials)
  print(f"camera slots: {dict(zip(slots, args.input_insvs, strict=True))}")
  session_id = session_name(args.input_insvs[0], serials[0])
  out_path = schema.layer_rrd(args.output_dir, schema.BASE, session_id)
  if out_path.exists() and not args.overwrite:
    raise SystemExit(f"{out_path} exists; pass --overwrite to replace it")
  captures = [probe_capture(master) for master in args.input_insvs]
  # Capture clock minus session clock (s), per capture.
  offsets, confidences = [0.0], [float("inf")]
  for later in args.input_insvs[1:]:
    sync = sync_two_captures(args.input_insvs[0], later)
    offsets.append(sync.offset)
    confidences.append(sync.peak_ratio)
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
  with tempfile.TemporaryDirectory(prefix="remux-") as tmp:
    for master, capture, insv, slot, offset, confidence in zip(
      args.input_insvs, captures, loaded, slots, offsets, confidences, strict=True
    ):
      for lens, src in (("front", master), ("rear", capture.rear_path)):
        mp4 = normalize_h264(src, Path(tmp) / f"{slot}_{lens}.mp4")
        video_streams.append(
          video_chunks(mp4, schema.video(slot, lens), time_shift=-offset, window=window)
        )
      sidecars += imu_chunks(slot, insv, time_shift=-offset, window=window)
      sidecars += exposure_chunks(slot, insv, time_shift=-offset, window=window)
      sidecars += calibration_chunks(slot, insv)
      sidecars.append(capture_chunk(slot, insv, master.name, -offset, confidence))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_base_recording(out_path, session_id, video_streams, sidecars)
  print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main(tyro.cli(Args))
