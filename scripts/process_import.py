"""Linux worker: verify an uploaded capture, detect episodes, then await review."""

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specialist.ingest import read_json, verify_job, write_json  # noqa: E402


def process(folder: Path, resume: bool) -> None:
  import fcntl

  folder = folder.resolve()
  repo = Path(__file__).resolve().parents[1]
  status_path = folder / "status.json"
  progress_path = folder / "progress.json"

  def status(state: str, **fields: object) -> None:
    write_json(status_path, dict(state=state, pid=os.getpid(), **fields))

  with (folder / "worker.lock").open("w") as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      return
    try:
      status("queued")
      # A single worker holds the GPU/disk-intensive processing slot per checkout.
      with (repo / ".import-processing.lock").open("w") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX)
        status("verifying")
        job = read_json(folder / "import.json")
        verify_job(folder, job)
        progress = read_json(progress_path) if progress_path.exists() else {}
        output = folder / "output"
        masters = [
          str(folder / job["hands"][hand]["master"]) for hand in ("left", "right")
        ]

        def run(script: str, *arguments: str) -> None:
          print(f"Running {script}", flush=True)
          subprocess.run(
            [sys.executable, f"scripts/{script}.py", *arguments], cwd=repo, check=True
          )

        conversion = ["--input-insvs", *masters, "--output-dir", str(output)]
        if not progress.get("detected"):
          status("detecting")
          run("convert", *conversion, "--detect-episodes", "--overwrite")
          candidates = list((output / "episodes").glob("*.json"))
          if len(candidates) != 1:
            raise ValueError("Expected exactly one episode manifest")
          progress.update(detected=True, manifest=candidates[0].name)
          write_json(progress_path, progress)
        manifest_path = output / "episodes" / progress["manifest"]
        manifest = read_json(manifest_path)
        if not resume or manifest.get("reviewed") is not True:
          status(
            "awaiting_review",
            manifest=str(manifest_path),
            candidates=len(manifest.get("episodes", [])),
          )
          return
        from specialist.ingest import digest

        fingerprint = digest(manifest_path)
        from specialist_dataset.schema import camera_slots

        from specialist.loader.parse_insv import load_insv

        serials = [
          str(load_insv(Path(p)).capture.get("serial_number", "")) for p in masters
        ]
        left_slot = camera_slots(serials)[0]
        if progress.get("split") and progress.get("manifest_sha256") != fingerprint:
          raise ValueError(
            "Boundaries changed after export; create a new job to reprocess"
          )
        if not progress.get("split"):
          status("splitting")
          run("convert", *conversion, "--episodes", str(manifest_path), "--overwrite")
          progress.update(split=True, manifest_sha256=fingerprint, completed=[])
          write_json(progress_path, progress)
        for episode in manifest["episodes"]:
          identifier = episode["episode_id"]
          if identifier in progress["completed"]:
            continue
          base = output / "base" / f"{manifest['session_id']}_{identifier}.rrd"
          status("estimating", episode=identifier)
          # Default anchor IDs match Specialist's supplied tags.
          run(
            "estimate",
            "--base",
            str(base),
            "--output-dir",
            str(output),
            "--anchor.tag-ids",
            *[str(i) for i in job.get("anchor_ids", [0, 1])],
            "--overwrite",
          )
          status("writing_state", episode=identifier)
          run(
            "state",
            "--base",
            str(base),
            "--output-dir",
            str(output),
            "--left-slot",
            left_slot,
            "--overwrite",
          )
          progress["completed"].append(identifier)
          write_json(progress_path, progress)
        status("complete", episodes=len(progress["completed"]), output=str(output))
    except Exception as exc:
      status("failed", error=str(exc), log=str(folder / "worker.log"))
      traceback.print_exc()
      raise


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--job-dir", type=Path, required=True)
  parser.add_argument("--resume", action="store_true")
  args = parser.parse_args()
  process(args.job_dir, args.resume)
