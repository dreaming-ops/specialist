# Camera import on Windows

The small English-language importer has a **Left hand** file field, a **Right
hand** file field and **Send for processing**. Select each original `_00_.insv`;
its matching `_10_.insv` must be in the same folder. Copy both pairs to your PC
first if you connect cameras one at a time. X3's USB **U-Disk Mode** exposes the
original files; exporting an MP4 in Insta360 Studio is unnecessary.

Launch the configured **Specialist Import** desktop shortcut, or:

```powershell
python scripts/import_camera.py gui
```

The importer keeps original filenames and separates files into `raw/left` and
`raw/right`, so equal filenames from two cameras do not collide. It verifies
copied bytes with SHA-256, leaves the originals alone, and rejects selecting the
same recording for both hands. There is no background USB-device watcher.

## One-time configuration

The Windows importer uses Python's standard library and OpenSSH, without the GPU
dependencies. The server needs a Linux installation of this branch and its full
Specialist dependencies, including the gripper description package. Authenticate
SSH normally first (including host-key trust). Password prompts are disabled in
the importer; use an SSH key or agent. Do not put passwords/tokens in configuration.

```powershell
python scripts/import_camera.py configure --host my-linux-host `
  --repo /home/user/specialist-import `
  --python /home/user/specialist-import/.venv/bin/python `
  --remote-root /home/user/specialist-data/imports `
  --local-root C:/Users/me/Videos/Specialist
```

Configuration is in `~/.specialist-import.json`; `--config PATH` selects another.
Remote paths currently must be absolute paths without spaces. The importer does
not install or update the server checkout automatically.

## What happens after Send

1. Copy and verify both original pairs in the local library.
2. Check SSH access and available server disk space. The low-disk workflow reserves
   2× raw size + the largest single lens file + 5 GiB before upload. For four
   similarly sized files this is about 2.25× raw size + 5 GiB. It rechecks disk
   space before each conversion, solve and state export.
3. Upload to a staging directory, then publish the complete job. Failed uploads
   can be retried; a submitted session does not create a second remote job.
4. The Linux worker verifies all hashes again, synchronizes cameras and runs
   episode detection. Only one job per server checkout processes at a time.
5. Stop at **awaiting_review**. Review the green spans in Rerun using the full
   base and review layer on the server, as described in
   [Continuous episodes](continuous-episodes.md).
6. After reviewed boundaries are uploaded and processing resumed, release the full
   review base and convert, estimate and write state for **one episode at a time**.
   Release each episode's base only after successful state export and a checkpoint.
   The hand choices from import are used explicitly in the final state; they are
   not inferred from motion.

Conversion keeps at most one temporary normalized lens MP4 on disk. Compressed
video chunks are collected in memory, as in the existing base writer; the worker
is intended for a server with sufficient RAM. Original INSV files are always
retained, along with manifests, estimates, visualizations and final training state.
The full Rerun review video is available until review is confirmed; afterwards it
can be regenerated from the originals. A very large individual episode or growing
outputs can still trigger a later space check. Free space and use `resume` to retry.

The interface shows copy/upload progress, not percentage or live remote job
progress. To inspect the server job:

```powershell
python scripts/import_camera.py status --session capture_20261113_120000
python scripts/import_camera.py fetch-review --session capture_20261113_120000
```

Edit the downloaded `review/<session>.json` only after checking the videos. Set
`reviewed` to `true`, correct boundaries and task labels, then:

```powershell
python scripts/import_camera.py upload-review --session capture_20261113_120000
python scripts/import_camera.py resume --session capture_20261113_120000
```

Remote job files: `status.json`, `progress.json`, `worker.log`, and `output/`.
Original and final processed files remain on the server; reconstructible base RRDs
are released as described above. No automatic dataset registration is performed.
The default anchor IDs are 0 and 1. For custom IDs,
set `anchor_ids` in the local `import.json` before submission.

## Sequential import and recovery

The CLI can copy the first hand directly from a mounted USB camera, wait for the
second camera, and automatically submit once both arrive:

```powershell
python scripts/import_camera.py add --session take01 --hand left --source E:/DCIM/Camera01/VID_20261113_120000_00_001.insv
python scripts/import_camera.py add --session take01 --hand right --source E:/DCIM/Camera01/VID_20261113_120002_00_001.insv
```

`--no-submit` copies only. `submit --session take01` retries a failed upload.
Identical same-hand reimports are idempotent; different files require a new session.
For a remote processing failure, inspect `worker.log`, correct the cause and use
`resume`. Completed stages are checkpointed. Do not change reviewed boundaries
after splitting has begun; use a new job for a new segmentation.

Do not disconnect cameras or close the importer during copying. If the process
was forcibly terminated, inspect the session's `.import-lock` directory before
removing that stale lock. Camera-created multi-file chapters and camera clock
drift remain limitations of the underlying conversion pipeline.

macOS packaging is deferred; this delivery targets Windows intake and Linux
processing.
