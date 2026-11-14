"""Run the real worker state machine with small input files and stubbed GPU stages."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fcntl")

from scripts import process_import  # noqa: E402

from specialist import ingest  # noqa: E402
from specialist.loader import parse_insv  # noqa: E402


def prepare(tmp_path: Path) -> Path:
  for hand in ("left", "right"):
    source = tmp_path / hand
    source.mkdir()
    master = source / "VID_20261113_120000_00_001.insv"
    master.write_bytes(hand.encode())
    master.with_name(master.name.replace("_00_", "_10_")).write_bytes(
      (hand + "rear").encode()
    )
    ingest.import_hand(tmp_path / "jobs", "take1", hand, master)
  return tmp_path / "jobs/take1"


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("reject_first", [False, True])
def test_worker_waits_for_review_then_processes_and_checkpoints(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, reject_first: bool
) -> None:
  folder = prepare(tmp_path)
  calls: list[list[str]] = []

  def run(command: list[str], **_kwargs: object) -> None:
    calls.append(command)
    if "--detect-episodes" in command:
      ingest.write_json(
        folder / "output/episodes/session.json",
        dict(
          version=1,
          session_id="session",
          reviewed=False,
          episodes=[
            dict(
              episode_id=f"ep{i + 1:04d}",
              capture_start=3 + i * 20,
              task_end=12 + i * 20,
              capture_end=13 + i * 20,
            )
            for i in range(count)
          ],
        ),
      )
      (folder / "output/base").mkdir(parents=True, exist_ok=True)
      (folder / "output/base/session.rrd").write_bytes(b"review video")
    elif "--episode-id" in command:
      identifier = command[command.index("--episode-id") + 1]
      assert not (folder / "output/base/session.rrd").exists()
      assert not list((folder / "output/base").glob("*.rrd"))
      (folder / f"output/base/session_{identifier}.rrd").write_bytes(b"episode video")
    elif command[1] == "scripts/state.py" and reject_first:
      base = Path(command[command.index("--base") + 1])
      if base.stem == "session_ep0001":
        report = Path(command[command.index("--rejection-report") + 1])
        ingest.write_json(
          report,
          dict(
            version=1,
            session_id=base.stem,
            source_session="session",
            episode_id="ep0001",
            check="initial_hold",
            slot="cam00",
            reason="hold is shorter than one second",
          ),
        )
        raise subprocess.CalledProcessError(3, command)

  monkeypatch.setattr(process_import.subprocess, "run", run)
  monkeypatch.setattr(
    parse_insv,
    "load_insv",
    lambda p: SimpleNamespace(
      capture={"serial_number": "a" if "left" in p.parts else "b"}
    ),
  )
  process_import.process(folder, resume=False)
  assert ingest.read_json(folder / "status.json")["state"] == "awaiting_review"
  assert len(calls) == 1
  assert "--imu-sync-fallback" in calls[0]
  assert (folder / "output/base/session.rrd").exists()
  process_import.process(folder, resume=True)
  assert len(calls) == 1  # --resume cannot bypass unreviewed boundaries.
  manifest = folder / "output/episodes/session.json"
  data = ingest.read_json(manifest)
  data["reviewed"] = True
  ingest.write_json(manifest, data)
  process_import.process(folder, resume=True)
  assert [c[1] for c in calls] == ["scripts/convert.py"] + [
    "scripts/convert.py",
    "scripts/estimate.py",
    "scripts/state.py",
  ] * count
  assert calls[-1][calls[-1].index("--left-slot") + 1] == "cam00"
  status = ingest.read_json(folder / "status.json")
  assert status["state"] == ("complete_with_rejections" if reject_first else "complete")
  assert status["episodes"] == count - int(reject_first)
  assert status["rejected"] == int(reject_first)
  progress = ingest.read_json(folder / "progress.json")
  if reject_first:
    assert "ep0001" not in progress["completed"]
    assert "ep0001" in progress["rejected"]
  process_import.process(folder, resume=True)
  assert len(calls) == 1 + count * 3
  assert not list((folder / "output/base").glob("*.rrd"))
  ingest.verify_job(folder, ingest.read_json(folder / "import.json"))
  data["episodes"][0]["task_end"] = 11
  ingest.write_json(manifest, data)
  with pytest.raises(ValueError, match="Boundaries changed"):
    process_import.process(folder, resume=True)


def test_worker_records_corrupt_upload_failure(tmp_path: Path) -> None:
  folder = prepare(tmp_path)
  job = ingest.read_json(folder / "import.json")
  (folder / job["hands"]["left"]["master"]).write_bytes(b"bad")
  with pytest.raises(ValueError, match="verification failed"):
    process_import.process(folder, resume=False)
  status = ingest.read_json(folder / "status.json")
  assert status["state"] == "failed"
  assert "verification failed" in status["error"]


def test_low_disk_check_stops_before_running_next_stage(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(
    process_import.shutil, "disk_usage", lambda _p: SimpleNamespace(free=100)
  )
  with pytest.raises(ValueError, match="Free space and resume"):
    process_import.require_space(tmp_path, 101)


def test_cannot_discard_raw_as_generated_base(tmp_path: Path) -> None:
  raw = tmp_path / "raw.insv"
  raw.write_bytes(b"keep")
  with pytest.raises(ValueError, match="outside"):
    process_import.discard_base(raw, tmp_path / "output")
  assert raw.read_bytes() == b"keep"


@pytest.mark.parametrize("failure", ["technical", "missing_report", "wrong_episode"])
def test_worker_does_not_mislabel_other_errors_as_quality_rejections(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  failure: str,
) -> None:
  folder = prepare(tmp_path)
  ingest.write_json(
    folder / "output/episodes/session.json",
    dict(
      version=1,
      session_id="session",
      reviewed=True,
      episodes=[
        dict(episode_id="ep0001", capture_start=3, task_end=12, capture_end=13)
      ],
    ),
  )
  ingest.write_json(
    folder / "progress.json", dict(detected=True, manifest="session.json")
  )
  (folder / "output/base").mkdir()

  def run(command: list[str], **_kwargs: object) -> None:
    if "--episode-id" in command:
      (folder / "output/base/session_ep0001.rrd").write_bytes(b"episode")
    elif command[1] == "scripts/state.py":
      if failure == "wrong_episode":
        report = Path(command[command.index("--rejection-report") + 1])
        ingest.write_json(
          report,
          dict(
            version=1,
            session_id="other",
            source_session="session",
            episode_id="ep0001",
            check="initial_hold",
            reason="short hold",
          ),
        )
      raise subprocess.CalledProcessError(1 if failure == "technical" else 3, command)

  monkeypatch.setattr(process_import.subprocess, "run", run)
  monkeypatch.setattr(
    parse_insv,
    "load_insv",
    lambda p: SimpleNamespace(
      capture={"serial_number": "a" if "left" in p.parts else "b"}
    ),
  )
  with pytest.raises(subprocess.CalledProcessError):
    process_import.process(folder, resume=True)
  assert ingest.read_json(folder / "status.json")["state"] == "failed"
  assert not ingest.read_json(folder / "progress.json")["completed"]
  assert (folder / "output/base/session_ep0001.rrd").exists()
