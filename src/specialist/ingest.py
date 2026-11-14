"""Dependency-free Windows/Linux intake and SSH submission of two-camera captures."""

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

NAME = re.compile(r"VID_\d{8}_\d{6}_00_\d+\.insv", re.IGNORECASE)


def read_json(path: Path) -> dict[str, Any]:
  return dict(json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, data: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
  try:
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)
  finally:
    temporary.unlink(missing_ok=True)


def digest(path: Path) -> str:
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def locked(root: Path) -> Iterator[None]:
  root.mkdir(parents=True, exist_ok=True)
  lock = root / ".import-lock"
  try:
    lock.mkdir()
  except FileExistsError:
    raise ValueError(f"Another operation is active: {lock}") from None
  try:
    yield
  finally:
    lock.rmdir()


def pair(source: Path) -> tuple[Path, Path]:
  if source.is_dir():
    masters = [p for p in source.iterdir() if p.is_file() and NAME.fullmatch(p.name)]
    if len(masters) != 1:
      raise ValueError(
        "Select one _00_ file; folder must contain exactly one recording"
      )
    source = masters[0]
  if not source.is_file() or not NAME.fullmatch(source.name):
    raise ValueError("Select an original VID_<date>_<time>_00_<number>.insv file")
  rear = source.with_name(source.name.replace("_00_", "_10_", 1))
  if not rear.is_file() or min(source.stat().st_size, rear.stat().st_size) == 0:
    raise ValueError("Both nonempty _00_ and _10_ files must be in the same folder")
  return source, rear


def session_dir(root: Path, name: str) -> Path:
  if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
    raise ValueError("Session name: 1–80 letters, digits, underscores or hyphens")
  return root / name


def import_hand(root: Path, name: str, hand: str, source: Path) -> dict[str, Any]:
  if hand not in ("left", "right"):
    raise ValueError("hand must be left or right")
  sources = pair(source)
  folder = session_dir(root, name)
  with locked(folder):
    path = folder / "import.json"
    job: dict[str, Any] = (
      read_json(path)
      if path.exists()
      else dict(
        version=1,
        name=name,
        job_id=f"{name}-{uuid.uuid4().hex[:12]}",
        hands={},
        state="waiting",
      )
    )
    if hand in job["hands"]:
      previous = job["hands"][hand]["files"]
      if all(
        src.stat().st_size == f["size"]
        and digest(src) == f["sha256"]
        and digest(folder / f["path"]) == f["sha256"]
        for src, f in zip(sources, previous, strict=True)
      ):
        return job
      raise ValueError(
        "This hand already contains different files; use a new session name"
      )
    if job["state"] not in ("waiting", "ready"):
      raise ValueError(
        "This hand is already imported or the session has been submitted"
      )
    required = sum(p.stat().st_size for p in sources)
    if shutil.disk_usage(folder).free < required + 1024**3:
      raise ValueError(
        "Not enough local disk space for a verified copy plus 1 GiB reserve"
      )
    staging = folder / f".copy-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
      files = []
      for src in sources:
        before = src.stat()
        dst = staging / src.name
        shutil.copyfile(src, dst)
        original_hash, copied_hash = digest(src), digest(dst)
        after = src.stat()
        if (before.st_size, before.st_mtime_ns) != (
          after.st_size,
          after.st_mtime_ns,
        ) or original_hash != copied_hash:
          raise ValueError(
            "Source changed during copying; finish recording/copying first"
          )
        files.append(
          dict(path=f"raw/{hand}/{src.name}", size=before.st_size, sha256=copied_hash)
        )
      fingerprints = [f["sha256"] for f in files]
      if any(
        fingerprints == [f["sha256"] for f in entry["files"]]
        for entry in job["hands"].values()
      ):
        raise ValueError("The same recording was selected for both hands")
      target = folder / "raw" / hand
      target.parent.mkdir(parents=True, exist_ok=True)
      if target.exists():
        raise ValueError(
          f"Unregistered files already exist at {target}; inspect before retrying"
        )
      staging.replace(target)
      job["hands"][hand] = dict(master=files[0]["path"], files=files)
      job["state"] = "ready" if len(job["hands"]) == 2 else "waiting"
      write_json(path, job)
      return job
    finally:
      if staging.exists() and staging.resolve().is_relative_to(folder.resolve()):
        shutil.rmtree(staging)


def verify_job(folder: Path, job: dict[str, Any]) -> None:
  if job.get("version") != 1 or set(job.get("hands", {})) != {"left", "right"}:
    raise ValueError("Both hands are required")
  if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", job.get("job_id", "")):
    raise ValueError("Invalid job ID")
  for hand, entry in job["hands"].items():
    files = entry["files"]
    if len(files) != 2 or entry["master"] != files[0]["path"]:
      raise ValueError("Invalid recording pair")
    master = folder / entry["master"]
    if not NAME.fullmatch(master.name):
      raise ValueError("Invalid master filename")
    expected = [
      f"raw/{hand}/{master.name}",
      f"raw/{hand}/{master.name.replace('_00_', '_10_', 1)}",
    ]
    for f, relative in zip(files, expected, strict=True):
      if f["path"] != relative:
        raise ValueError("File path does not belong to this hand")
      path = folder / relative
      if not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError("Input escapes session folder")
      if (
        not path.is_file()
        or path.stat().st_size != f["size"]
        or digest(path) != f["sha256"]
      ):
        raise ValueError(f"File verification failed: {relative}")


def validate_config(config: dict[str, Any]) -> None:
  if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", config["host"]):
    raise ValueError("Use an SSH alias or user@hostname")
  for key in ("repo", "remote_root", "python"):
    value = config[key]
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", value) or ".." in Path(value).parts:
      raise ValueError(f"{key} must be an absolute Linux path without spaces or '..'")


def ssh(config: dict[str, Any], command: str) -> str:
  validate_config(config)
  return subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", config["host"], command],
    check=True,
    text=True,
    capture_output=True,
    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
  ).stdout.strip()


def launch(config: dict[str, Any], remote: str, resume: bool = False) -> None:
  args = [config["python"], "scripts/process_import.py", "--job-dir", remote]
  if resume:
    args.append("--resume")
  command = f"cd {shlex.quote(config['repo'])} && "
  # The detached child keeps its log; the job file is the durable source of status.
  command += f"nohup {shlex.join(args)} >> {shlex.quote(remote + '/worker.log')} 2>&1 < /dev/null &"
  ssh(config, command)


def submit(root: Path, name: str, config: dict[str, Any]) -> dict[str, Any]:
  folder = session_dir(root, name)
  with locked(folder):
    job = read_json(folder / "import.json")
    if job["state"] == "submitted":
      return job
    verify_job(folder, job)
    validate_config(config)
    remote = f"{config['remote_root']}/{job['job_id']}"
    # Check the configured worker and interpreter before uploading any large files.
    ssh(
      config,
      f"test -x {config['python']} && test -f {config['repo']}/scripts/process_import.py && mkdir -p {config['remote_root']}",
    )
    free = int(ssh(config, f"df -B1 --output=avail {config['remote_root']} | tail -1"))
    size = sum(f["size"] for h in job["hands"].values() for f in h["files"])
    if free < size * 4 + 5 * 1024**3:
      raise ValueError(
        "Server needs at least 4× raw file size + 5 GiB free; no upload started"
      )
    # Retry reuses a staging directory; it never overwrites a published remote job.
    incoming = remote + ".uploading"
    exists = ssh(config, f"if test -f {remote}/import.json; then echo yes; fi")
    if not exists:
      ssh(config, f"mkdir -p {incoming}")
      subprocess.run(
        [
          "scp",
          "-B",
          "-r",
          str(folder / "raw"),
          str(folder / "import.json"),
          f"{config['host']}:{incoming}/",
        ],
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
      )
      # Remote validation is performed before the worker touches any video.
      ssh(config, f"test ! -e {remote} && mv {incoming} {remote}")
    launch(config, remote)
    job.update(state="submitted", remote=remote, host=config["host"])
    write_json(folder / "import.json", job)
    return job
