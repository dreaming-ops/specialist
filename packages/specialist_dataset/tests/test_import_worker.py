"""Run the real worker state machine with small input files and stubbed GPU stages."""

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


def test_worker_waits_for_review_then_processes_and_checkpoints(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
            dict(episode_id="ep0001", capture_start=3, task_end=12, capture_end=13)
          ],
        ),
      )

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
  process_import.process(folder, resume=True)
  assert len(calls) == 1  # --resume cannot bypass unreviewed boundaries.
  manifest = folder / "output/episodes/session.json"
  data = ingest.read_json(manifest)
  data["reviewed"] = True
  ingest.write_json(manifest, data)
  process_import.process(folder, resume=True)
  assert [c[1] for c in calls] == [
    "scripts/convert.py",
    "scripts/convert.py",
    "scripts/estimate.py",
    "scripts/state.py",
  ]
  assert calls[-1][calls[-1].index("--left-slot") + 1] == "cam00"
  assert ingest.read_json(folder / "status.json")["state"] == "complete"
  process_import.process(folder, resume=True)
  assert len(calls) == 4
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
