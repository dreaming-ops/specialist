"""Intake tests use small stand-ins, never camera data or a real SSH server."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from specialist import ingest


def camera(root: Path, body: bytes = b"camera-one") -> Path:
  root.mkdir()
  master = root / "VID_20261113_120000_00_001.insv"
  master.write_bytes(body)
  master.with_name(master.name.replace("_00_", "_10_")).write_bytes(body + b"rear")
  return master


def test_sequential_import_retains_names_and_originals(tmp_path: Path) -> None:
  left, right = camera(tmp_path / "left"), camera(tmp_path / "right", b"camera-two")
  root = tmp_path / "library"
  job = ingest.import_hand(root, "take1", "left", left)
  assert job["state"] == "waiting"
  job = ingest.import_hand(root, "take1", "right", right)
  assert job["state"] == "ready"
  ingest.verify_job(root / "take1", job)
  assert left.is_file() and right.is_file()
  assert (root / "take1/raw/left" / left.name).read_bytes() == left.read_bytes()
  # Retry after a failed network transfer must not recopy or create a new session.
  assert ingest.import_hand(root, "take1", "left", left)["job_id"] == job["job_id"]


def test_reject_duplicate_hand_recording(tmp_path: Path) -> None:
  source = camera(tmp_path / "cam")
  root = tmp_path / "library"
  ingest.import_hand(root, "take1", "left", source)
  with pytest.raises(ValueError, match="same recording"):
    ingest.import_hand(root, "take1", "right", source)
  assert not (root / "take1/raw/right").exists()


def test_refuse_missing_pair(tmp_path: Path) -> None:
  source = camera(tmp_path / "cam")
  source.with_name(source.name.replace("_00_", "_10_")).unlink()
  with pytest.raises(ValueError, match="Both nonempty"):
    ingest.import_hand(tmp_path / "library", "take1", "left", source)


def test_no_submission_until_both_hands(tmp_path: Path) -> None:
  source = camera(tmp_path / "cam")
  root = tmp_path / "library"
  ingest.import_hand(root, "take1", "left", source)
  with pytest.raises(ValueError, match="Both hands"):
    ingest.submit(root, "take1", {})


def test_modified_copy_fails_verification(tmp_path: Path) -> None:
  root = tmp_path / "library"
  ingest.import_hand(root, "take1", "left", camera(tmp_path / "left"))
  job = ingest.import_hand(root, "take1", "right", camera(tmp_path / "right", b"other"))
  (root / "take1" / job["hands"]["left"]["master"]).write_bytes(b"corrupt")
  with pytest.raises(ValueError, match="verification failed"):
    ingest.verify_job(root / "take1", job)


@pytest.mark.parametrize("name", ["../escape", "a/b", "", "x; echo hi"])
def test_session_path_validation(tmp_path: Path, name: str) -> None:
  with pytest.raises(ValueError):
    ingest.session_dir(tmp_path, name)


def test_upload_failure_is_retryable(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  root = tmp_path / "library"
  ingest.import_hand(root, "take1", "left", camera(tmp_path / "left"))
  ingest.import_hand(root, "take1", "right", camera(tmp_path / "right", b"other"))
  config = dict(
    host="test-host",
    repo="/project",
    python="/project/.venv/bin/python",
    remote_root="/jobs",
  )
  monkeypatch.setattr(
    ingest,
    "ssh",
    lambda _config, command: str(100 * 1024**3) if command.startswith("df ") else "",
  )

  def fail(*_args: object, **_kwargs: object) -> None:
    raise OSError("network unavailable")

  monkeypatch.setattr(ingest.subprocess, "run", fail)
  with pytest.raises(OSError):
    ingest.submit(root, "take1", config)
  assert ingest.read_json(root / "take1/import.json")["state"] == "ready"
  monkeypatch.setattr(
    ingest.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0)
  )
  assert ingest.submit(root, "take1", config)["state"] == "submitted"


def test_server_space_gate_runs_before_scp(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  root = tmp_path / "library"
  ingest.import_hand(root, "take1", "left", camera(tmp_path / "left"))
  ingest.import_hand(root, "take1", "right", camera(tmp_path / "right", b"other"))
  config = dict(
    host="test-host",
    repo="/project",
    python="/project/.venv/bin/python",
    remote_root="/jobs",
  )
  monkeypatch.setattr(
    ingest, "ssh", lambda _config, command: "10" if command.startswith("df ") else ""
  )
  with pytest.raises(ValueError, match="no upload"):
    ingest.submit(root, "take1", config)
