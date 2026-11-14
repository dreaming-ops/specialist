"""Import left/right camera files on Windows, then submit over SSH. No pip install."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specialist.ingest import (  # noqa: E402
  import_hand,
  launch,
  read_json,
  session_dir,
  ssh,
  submit,
  validate_config,
  write_json,
)

DEFAULT_CONFIG = Path.home() / ".specialist-import.json"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
  commands = parser.add_subparsers(dest="command", required=True)
  configure = commands.add_parser("configure")
  configure.add_argument("--host", required=True)
  configure.add_argument("--repo", required=True)
  configure.add_argument("--remote-root", required=True)
  configure.add_argument("--python", required=True)
  configure.add_argument("--local-root", type=Path, required=True)
  add = commands.add_parser("add")
  add.add_argument("--session", required=True)
  add.add_argument("--hand", choices=["left", "right"], required=True)
  add.add_argument("--source", type=Path, required=True)
  add.add_argument("--no-submit", action="store_true")
  commands.add_parser("gui")
  for action in ("submit", "status", "resume", "fetch-review", "upload-review"):
    commands.add_parser(action).add_argument("--session", required=True)
  args = parser.parse_args()
  if args.command == "configure":
    config = dict(
      host=args.host,
      repo=args.repo,
      remote_root=args.remote_root,
      python=args.python,
      local_root=str(args.local_root.resolve()),
    )
    validate_config(config)
    write_json(args.config, config)
    print(f"Saved {args.config}")
    return
  config = read_json(args.config)
  validate_config(config)
  root = Path(config["local_root"])
  if args.command == "gui":
    gui(config)
    return
  folder = session_dir(root, args.session)
  if args.command == "add":
    job = import_hand(root, args.session, args.hand, args.source)
    print(f"Imported {args.hand}: {job['state']}")
    if job["state"] == "ready" and not args.no_submit:
      print(json.dumps(submit(root, args.session, config), indent=2))
    return
  if args.command == "submit":
    print(json.dumps(submit(root, args.session, config), indent=2))
    return
  job = read_json(folder / "import.json")
  if job.get("host") != config["host"] or not job.get("remote"):
    raise ValueError("Session is not submitted to this server")
  remote = f"{config['remote_root']}/{job['job_id']}"
  if remote != job["remote"]:
    raise ValueError("Server path changed since submission")
  if args.command == "status":
    print(ssh(config, f"cat {remote}/status.json"))
  elif args.command == "resume":
    launch(config, remote, resume=True)
    print("Processing requested; use status to follow progress")
  else:
    progress = json.loads(ssh(config, f"cat {remote}/progress.json"))
    filename = Path(progress["manifest"]).name
    if filename != progress["manifest"] or not re.fullmatch(
      r"[A-Za-z0-9_-]+\.json", filename
    ):
      raise ValueError("Invalid remote manifest name")
    review = folder / "review"
    review.mkdir(exist_ok=True)
    target = f"{remote}/output/episodes/{filename}"
    if args.command == "fetch-review":
      subprocess.run(
        ["scp", "-B", f"{config['host']}:{target}", str(review / filename)], check=True
      )
      print(f"Saved {review / filename}; inspect videos in Rerun on the Linux server")
    else:
      data = read_json(review / filename)
      if data.get("reviewed") is not True:
        raise ValueError("Review boundaries and set reviewed to true first")
      subprocess.run(
        ["scp", "-B", str(review / filename), f"{config['host']}:{target}.new"],
        check=True,
      )
      ssh(config, f"mv {target}.new {target}")
      print("Reviewed boundaries uploaded; run resume to process")


def gui(config: dict[str, str]) -> None:
  import threading
  import tkinter as tk
  from datetime import datetime
  from functools import partial
  from queue import Empty, Queue
  from tkinter import filedialog, messagebox, ttk

  window = tk.Tk()
  window.title("Specialist — Camera Import")
  window.geometry("690x390")
  window.minsize(620, 390)
  session = tk.StringVar(value=datetime.now().strftime("capture_%Y%m%d_%H%M%S"))
  paths = {hand: tk.StringVar() for hand in ("left", "right")}
  frame = ttk.Frame(window, padding=22)
  frame.pack(fill="both", expand=True)
  ttk.Label(
    frame, text="Import a camera recording", font=("Segoe UI", 15, "bold")
  ).pack(anchor="w")
  ttk.Label(
    frame,
    text="Select the original _00_.insv files. Each _10_ file must be in the same folder.",
  ).pack(anchor="w", pady=(4, 14))
  ttk.Label(frame, text="Session name").pack(anchor="w")
  ttk.Entry(frame, textvariable=session).pack(fill="x", pady=(3, 10))

  def browse(side: str) -> None:
    selected = filedialog.askopenfilename(
      title="Select the _00_.insv file", filetypes=[("Insta360", "*.insv")]
    )
    if selected:
      paths[side].set(selected)

  for side, label in [("left", "Left hand"), ("right", "Right hand")]:
    ttk.Label(frame, text=label).pack(anchor="w")
    row = ttk.Frame(frame)
    row.pack(fill="x", pady=(3, 10))
    ttk.Entry(row, textvariable=paths[side]).pack(side="left", fill="x", expand=True)
    ttk.Button(row, text="Browse…", command=partial(browse, side)).pack(
      side="right", padx=(8, 0)
    )
  status = tk.StringVar(
    value=f"Server: {config['host']}. Original camera files are kept."
  )
  events: Queue[tuple[str, str]] = Queue()
  activity = ttk.Progressbar(frame, mode="indeterminate")

  def start() -> None:
    name = session.get().strip()
    selected = {side: Path(value.get()) for side, value in paths.items()}
    if any(not value.get().strip() for value in paths.values()):
      messagebox.showinfo(
        "Select recordings", "Select both the left-hand and right-hand recordings."
      )
      return
    try:
      session_dir(Path(config["local_root"]), name)
      from specialist.ingest import pair

      for path in selected.values():
        pair(path)
    except ValueError as exc:
      messagebox.showerror("Check input files", str(exc))
      return
    button.configure(state="disabled")
    activity.start(12)

    def work() -> None:
      try:
        for side, label in [("left", "left"), ("right", "right")]:
          events.put(
            (
              "progress",
              f"Copying and verifying the {label} hand… Keep the camera connected.",
            )
          )
          import_hand(Path(config["local_root"]), name, side, selected[side])
        events.put(
          (
            "progress",
            "Uploading to the server… Episode detection will start after the upload.",
          )
        )
        submit(Path(config["local_root"]), name, config)
        events.put(
          (
            "done",
            "Submitted. Review the episode boundaries after detection completes.",
          )
        )
      except Exception as exc:
        detail = str(exc)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
          detail += "\n" + str(exc.stderr)
        events.put(("error", detail))

    threading.Thread(target=work, daemon=True).start()

  button = ttk.Button(frame, text="Send for processing", command=start)
  button.pack(fill="x", pady=(6, 10), ipady=5)
  activity.pack(fill="x")
  ttk.Label(frame, textvariable=status, wraplength=640).pack(anchor="w", pady=10)

  def poll() -> None:
    try:
      while True:
        kind, message = events.get_nowait()
        status.set(message)
        if kind != "progress":
          activity.stop()
          button.configure(state="normal")
          if kind == "error":
            messagebox.showerror("Camera import", message)
    except Empty:
      pass
    window.after(200, poll)

  def close() -> None:
    if str(button.cget("state")) == "disabled":
      messagebox.showinfo(
        "Camera import", "Please wait until copying and uploading have finished."
      )
    else:
      window.destroy()

  window.protocol("WM_DELETE_WINDOW", close)
  poll()
  window.mainloop()


if __name__ == "__main__":
  try:
    main()
  except (ValueError, OSError, subprocess.CalledProcessError) as exc:
    print(f"Import failed: {exc}", file=sys.stderr)
    if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
      print(exc.stderr, file=sys.stderr)
    sys.exit(1)
