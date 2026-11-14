"""A session names itself from the capture that defines its clock."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import convert
from scripts.convert import session_name


def test_session_name_pairs_the_start_instant_with_the_device() -> None:
  master = Path("VID_20260908_044638_00_001.insv")
  assert session_name(master, "IAQEB2601UVVAJ") == "20260908_044638_IAQEB2601UVVAJ"
  assert session_name(master, "IAQEB2601UVVAJ") < session_name(
    Path("VID_20260908_044840_00_002.insv"), "IAQEB2601UVVAJ"
  )


def test_session_name_refuses_a_capture_it_cannot_name() -> None:
  with pytest.raises(SystemExit, match="no capture instant or serial"):
    session_name(Path("clip.insv"), "IAQEB2601UVVAJ")
  with pytest.raises(SystemExit, match="no capture instant or serial"):
    session_name(Path("VID_20260908_044638_00_001.insv"), "")


@pytest.mark.parametrize("sync_method", ["single", "audio", "imu"])
def test_split_preserves_clock_and_telemetry_windows(
  tmp_path, monkeypatch, sync_method
) -> None:
  """Exercise conversion orchestration without camera IO or a native solve."""
  import json

  from specialist_dataset.episodes import Episode, manifest

  source = Path("VID_20260908_044638_00_001.insv")
  other = Path("VID_20260908_044640_00_002.insv")
  name = session_name(source, "serial")
  episode = Episode("ep0001", 3.0, 12.0, 13.0, "stack")
  data = manifest(name, [], [episode], [])
  data["reviewed"] = True
  review = tmp_path / "episodes.json"
  review.write_text(json.dumps(data))
  monkeypatch.setattr(
    convert.sys,
    "stdout",
    SimpleNamespace(
      reconfigure=lambda **_kw: None, write=lambda _x: None, flush=lambda: None
    ),
  )
  monkeypatch.setattr(
    convert,
    "load_insv",
    lambda p: SimpleNamespace(
      capture={"serial_number": "serial" if p == source else "serial2"},
      imu_timestamps=None,
      accel=None,
    ),
  )

  def audio_sync(*_args):
    if sync_method == "imu":
      raise RuntimeError("audio alignment is ambiguous")
    return SimpleNamespace(offset=-1.87, peak_ratio=3)

  monkeypatch.setattr(convert, "sync_two_captures", audio_sync)
  monkeypatch.setattr(
    convert,
    "sync_two_imus",
    lambda *_args: SimpleNamespace(
      offset=-1.87, confidence=23, matches=23, residual_seconds=0.034
    ),
  )
  monkeypatch.setattr(
    convert, "probe_capture", lambda p: SimpleNamespace(duration=60, rear_path=p)
  )
  monkeypatch.setattr(convert, "normalize_h264", lambda _src, dst: dst)
  calls = []

  def video(mp4, entity, time_shift, window):
    calls.append(("video", time_shift, window))
    return None

  def telemetry(slot, insv, time_shift, window):
    calls.append(("telemetry", time_shift, window))
    return []

  monkeypatch.setattr(convert, "video_chunks", video)
  monkeypatch.setattr(convert, "imu_chunks", telemetry)
  monkeypatch.setattr(convert, "exposure_chunks", telemetry)
  monkeypatch.setattr(convert, "calibration_chunks", lambda *_a: [])
  clocks = []
  monkeypatch.setattr(convert, "capture_chunk", lambda *args: clocks.append(args[3:]))
  monkeypatch.setattr(convert, "write_review", lambda *_a: None)
  monkeypatch.setattr(
    convert, "write_base_recording", lambda path, *_a: path.write_bytes(b"test")
  )
  convert.main(
    convert.Args(
      input_insvs=(source,) if sync_method == "single" else (source, other),
      output_dir=tmp_path,
      episodes=review,
      imu_sync_fallback=sync_method == "imu",
    )
  )
  assert len(calls) == (4 if sync_method == "single" else 8)
  assert all(shift == 0 and window == (3, 13) for _, shift, window in calls[:4])
  assert all(shift == 1.87 and window == (3, 13) for _, shift, window in calls[4:])
  assert clocks[0][2] == "reference"
  if sync_method != "single":
    assert clocks[1] == (
      1.87,
      23 if sync_method == "imu" else 3,
      "imu_triple_taps" if sync_method == "imu" else "audio",
    )
  metadata = json.loads((tmp_path / "base" / f"{name}_ep0001.episode.json").read_text())
  assert metadata["task_end"] == 12
  assert metadata["task"] == "stack"
  assert metadata["source_session"] == name
  assert not (tmp_path / "base" / f"{name}.rrd").exists()
