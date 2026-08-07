"""Convert one session's ``.insv`` captures into a base recording.

Usage:
    uv run python scripts/convert.py --input-insvs camA_00.insv camB_00.insv \
        --session-id <session-id>
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from specialist.loader.audio_sync import sync_two_captures
from specialist.loader.parse_insv import load_insv
from specialist.record import schema
from specialist.record.base import (
  calibration_chunks,
  exposure_chunks,
  imu_chunks,
  write_base_recording,
)
from specialist.record.video import normalize_h264, video_chunks
from specialist.vi_ba.configs import ImuCfg
from specialist.vi_ba.pipeline import capture_duration


@dataclass
class Args:
  """Base-recording conversion of one session."""

  # One _00_ master .insv per capture; the first defines the session clock.
  input_insvs: tuple[Path, ...] = ()
  # Recording id shared by this session's base recording and track layers.
  session_id: str = ""
  # One name per capture; it must match the solve's --slots.
  slots: tuple[str, ...] = ()
  output_dir: Path = Path("logs")
  # Seconds to add to each capture's clock, for clips that share no audible event.
  audio_offsets: tuple[float, ...] = ()
  imu: ImuCfg = field(default_factory=ImuCfg)


def main(args: Args) -> None:
  if not args.input_insvs or not args.session_id:
    raise SystemExit("--input-insvs and --session-id are required")
  names = (
    list(args.slots)
    if args.slots
    else [f"tracker_{i}" for i in range(len(args.input_insvs))]
  )
  if len(names) != len(args.input_insvs):
    raise SystemExit("--slots must match --input-insvs in length")
  rears = [p.with_name(p.name.replace("_00_", "_10_", 1)) for p in args.input_insvs]
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

  # Session window: latest capture start to earliest end, both lenses included.
  durations = [
    min(capture_duration(master), capture_duration(rear))
    for master, rear in zip(args.input_insvs, rears, strict=True)
  ]
  window = (
    max(-off for off in offsets),
    min(dur - off for dur, off in zip(durations, offsets, strict=True)),
  )
  print(f"session window: [{window[0]:.3f}, {window[1]:.3f}] s")

  video_streams = []
  sidecars = []
  body_R_imu = np.asarray(args.imu.body_R_imu, dtype=np.float64)
  with tempfile.TemporaryDirectory(prefix="remux-") as tmp:
    for master, rear, tracker, offset in zip(
      args.input_insvs, rears, names, offsets, strict=True
    ):
      insv = load_insv(master)
      for lens, src in (("front", master), ("rear", rear)):
        mp4 = normalize_h264(src, Path(tmp) / f"{tracker}_{lens}.mp4")
        video_streams.append(
          video_chunks(
            mp4,
            schema.image_entity(tracker, lens),
            schema.TIMELINE,
            time_shift=-offset,
            window=window,
          )
        )
      sidecars += imu_chunks(tracker, insv, time_shift=-offset, window=window)
      sidecars += exposure_chunks(tracker, insv, time_shift=-offset, window=window)
      sidecars += calibration_chunks(tracker, insv, body_R_imu)

    properties: dict[str, dict[str, Any]] = {
      "session": {
        "session_id": args.session_id,
        "sources": [p.name for p in args.input_insvs],
      },
      "audio_offset": dict(zip(names, offsets, strict=True)),
    }
    out_path = args.output_dir / "base" / f"{args.session_id}.rrd"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_base_recording(out_path, args.session_id, video_streams, sidecars, properties)
  print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main(tyro.cli(Args))
