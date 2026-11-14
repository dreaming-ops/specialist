"""Linux worker: verify an uploaded capture, detect episodes, then await review."""

import argparse
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specialist.ingest import (  # noqa: E402
  DISK_RESERVE,
  conversion_space,
  read_json,
  verify_job,
  write_json,
)


def require_space(folder: Path, required: int) -> None:
  free = shutil.disk_usage(folder).free
  if free < required:
    raise ValueError(
      f"Need {required / 1024**3:.1f} GiB free for the next stage; "
      f"only {free / 1024**3:.1f} GiB available. Free space and resume."
    )


def discard_base(path: Path, output: Path) -> None:
  """Release only a reconstructible generated base, never raw camera data."""
  if path.parent != output / "base" or not path.resolve().is_relative_to(
    output.resolve()
  ):
    raise ValueError("Refusing to remove a base outside this job's output")
  path.unlink(missing_ok=True)


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
        temporary = folder / ".tmp"
        temporary.mkdir(exist_ok=True)
        masters = [
          str(folder / job["hands"][hand]["master"]) for hand in ("left", "right")
        ]

        def run(script: str, *arguments: str) -> None:
          print(f"Running {script}", flush=True)
          subprocess.run(
            [sys.executable, f"scripts/{script}.py", *arguments],
            cwd=repo,
            check=True,
            env={**os.environ, "TMPDIR": str(temporary)},
          )

        conversion = [
          "--input-insvs",
          *masters,
          "--output-dir",
          str(output),
          "--low-disk",
          "--imu-sync-fallback",
        ]
        if not progress.get("detected"):
          require_space(folder, conversion_space(job))
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
        from specialist_dataset.episodes import read_manifest

        session_id = str(manifest["session_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
          raise ValueError("Invalid source session ID")
        # Validate all reviewed paths/ranges before deleting any generated file.
        read_manifest(manifest_path, session_id, (float("-inf"), float("inf")))
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
          progress.update(split=True, manifest_sha256=fingerprint, completed=[])
          write_json(progress_path, progress)
        progress.setdefault("rejected", {})
        # Full review video is reconstructible from the retained INSV originals.
        discard_base(output / "base" / f"{session_id}.rrd", output)
        for episode in manifest["episodes"]:
          identifier = episode["episode_id"]
          base = output / "base" / f"{session_id}_{identifier}.rrd"
          if identifier in progress["completed"] or identifier in progress["rejected"]:
            discard_base(base, output)
            continue
          discard_base(base, output)
          require_space(folder, conversion_space(job))
          status("splitting", episode=identifier)
          run(
            "convert",
            *conversion,
            "--episodes",
            str(manifest_path),
            "--episode-id",
            identifier,
            "--overwrite",
          )
          # The solver stages lens video from this one episode's base.
          require_space(folder, base.stat().st_size + DISK_RESERVE)
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
          require_space(folder, base.stat().st_size + DISK_RESERVE)
          report_path = output / "rejections" / f"{session_id}_{identifier}.json"
          report_path.unlink(missing_ok=True)
          try:
            run(
              "state",
              "--base",
              str(base),
              "--output-dir",
              str(output),
              "--left-slot",
              left_slot,
              "--rejection-report",
              str(report_path),
              "--overwrite",
            )
          except subprocess.CalledProcessError as exc:
            if exc.returncode != 3 or not report_path.is_file():
              raise
            report = read_json(report_path)
            if not (
              report.get("version") == 1
              and report.get("check") == "initial_hold"
              and report.get("session_id") == f"{session_id}_{identifier}"
              and report.get("source_session") == session_id
              and report.get("episode_id") == identifier
              and isinstance(report.get("reason"), str)
            ):
              raise
            progress["rejected"][identifier] = report
            write_json(progress_path, progress)
            print(f"Rejected {identifier}: {report['reason']}", flush=True)
            discard_base(base, output)
            continue
          progress["completed"].append(identifier)
          write_json(progress_path, progress)
          discard_base(base, output)
        status(
          "complete_with_rejections" if progress["rejected"] else "complete",
          episodes=len(progress["completed"]),
          rejected=len(progress["rejected"]),
          output=str(output),
        )
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
