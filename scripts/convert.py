"""Convert one session's ``.insv`` captures into a base recording.

Usage:
    uv run python scripts/convert.py --input-insvs camA_00.insv camB_00.insv
"""

import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro
from specialist_dataset import schema
from specialist_dataset.episodes import (
  Episode,
  detect,
  manifest,
  propose,
  read_manifest,
)

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
from specialist.record.episodes import write_review
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

  # Write candidate boundaries and a green Rerun review track; still converts the full base.
  detect_episodes: bool = False

  # Export separate bases from a reviewed JSON manifest, preserving telemetry.
  episodes: Path | None = None


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
  if out_path.exists() and not args.overwrite and args.episodes is None:
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

  if args.detect_episodes and args.episodes:
    raise SystemExit("choose --detect-episodes or --episodes, not both")
  exports: list[tuple[str, tuple[float, float], Episode | None]] = [
    (session_id, window, None)
  ]
  if args.episodes:
    episodes = read_manifest(args.episodes, session_id, window)
    exports = [
      (f"{session_id}_{e.episode_id}", (e.capture_start, e.capture_end), e)
      for e in episodes
    ]
  # Preflight all output paths before writing any episode.
  for name, _, _ in exports:
    target = schema.layer_rrd(args.output_dir, schema.BASE, name)
    if target.exists() and not args.overwrite:
      raise SystemExit(f"{target} exists; pass --overwrite to replace it")

  if args.episodes:
    write_review(
      args.output_dir / "episodes" / f"{session_id}.reviewed.rrd",
      session_id,
      slots,
      episodes,
      [],
    )

  if args.detect_episodes:
    times = {
      s: insv.imu_timestamps - off
      for s, insv, off in zip(slots, loaded, offsets, strict=True)
    }
    accel = {s: insv.accel for s, insv in zip(slots, loaded, strict=True)}
    for slot in slots:
      keep = (times[slot] >= window[0]) & (times[slot] <= window[1])
      times[slot], accel[slot] = times[slot][keep], accel[slot][keep]
    markers, warnings = detect(times, accel)
    try:
      candidates = propose(markers, window)
    except ValueError as exc:
      candidates = []
      warnings.append(str(exc))
    review_path = args.output_dir / "episodes" / f"{session_id}.json"
    if review_path.exists() and not args.overwrite:
      raise SystemExit(f"{review_path} exists; refusing to replace reviewed boundaries")
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(
      json.dumps(manifest(session_id, markers, candidates, warnings), indent=2),
      encoding="utf-8",
    )
    write_review(
      review_path.with_suffix(".rrd"), session_id, slots, candidates, markers
    )
    print(f"Review {review_path}: {len(candidates)} candidate episodes; {warnings}")

  with tempfile.TemporaryDirectory(prefix="remux-") as tmp:
    videos = {}
    for master, capture, slot in zip(args.input_insvs, captures, slots, strict=True):
      for lens, src in (("front", master), ("rear", capture.rear_path)):
        videos[slot, lens] = normalize_h264(src, Path(tmp) / f"{slot}_{lens}.mp4")
    for name, selected, episode in exports:
      video_streams = []
      sidecars = []
      for master, insv, slot, offset, confidence in zip(
        args.input_insvs, loaded, slots, offsets, confidences, strict=True
      ):
        for lens in schema.LENSES:
          video_streams.append(
            video_chunks(
              videos[slot, lens],
              schema.video(slot, lens),
              time_shift=-offset,
              window=selected,
            )
          )
        sidecars += imu_chunks(slot, insv, time_shift=-offset, window=selected)
        sidecars += exposure_chunks(slot, insv, time_shift=-offset, window=selected)
        sidecars += calibration_chunks(slot, insv)
        sidecars.append(capture_chunk(slot, insv, master.name, -offset, confidence))
      out_path = schema.layer_rrd(args.output_dir, schema.BASE, name)
      out_path.parent.mkdir(parents=True, exist_ok=True)
      write_base_recording(out_path, name, video_streams, sidecars)
      if episode:
        out_path.with_suffix(".episode.json").write_text(
          json.dumps(
            dict(version=1, source_session=session_id, **asdict(episode)), indent=2
          ),
          encoding="utf-8",
        )
      print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main(tyro.cli(Args))
