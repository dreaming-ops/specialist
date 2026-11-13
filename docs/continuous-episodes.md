# Continuous recording with triple-tap episode markers

This experimental workflow keeps the cameras recording while the operator resets
the scene. It creates separate base recordings before trajectory estimation, so
the solver does not need to optimize a whole 30-minute capture at once.

## Operator protocol

1. Start both cameras within 30 seconds of each other, with audio enabled.
2. Tap **both camera-mounted grippers** on the table three times together. Space
   taps about 0.3 seconds apart. A clap near stationary cameras is not a marker.
3. Show the anchor tags to the auxiliary lenses and hold still for at least one
   second (aim for 1.2 seconds to allow sampling tolerance), within five seconds
   of the triple. Keep gripper tags visible too.
4. Leave one second of padding, then perform the task.
5. Leave one second of padding after the task, then tap both grippers on the table
   three times again.
6. Change the scene without stopping recording. Leave at least one second between
   triple-tap groups. Repeat from step 2.

Do not move the anchor tags during an episode. If the layout changes between
episodes, show the anchors again at the next start. Each episode is solved
independently; moving anchors changes the meaning of its world frame.

## Detect and review

Keep both `_00_` and `_10_` original INSV files for each camera. Run on the Linux
environment required by Specialist:

```bash
uv run python scripts/convert.py \
  --input-insvs path/to/left_00.insv path/to/right_00.insv \
  --output-dir logs --detect-episodes
```

The converter writes the usual full `base/<session>.rrd`, plus:

- `episodes/<session>.json`: candidate pairs, warnings and a task label per pair.
- `episodes/<session>.rrd`: a review layer with translucent green spans on a
  dedicated time plot beside the video, sharing the `video_time` playback cursor.
  This is a separate episode track, not a recoloring of Rerun's built-in time bar.

```bash
uv run rerun logs/base/<session>.rrd logs/episodes/<session>.rrd
```

Inspect every candidate. Green spans run from just after the starting triple to
the first ending strike; they include the tag-showing prelude. The final training
window starts after the detected hold and excludes one second at each end by
default. Exact training boundaries are printed by `state.py`.

In the JSON, correct `capture_start` (after the starting triple, before the hold),
`task_end` (the first ending strike), and `capture_end` (after the ending triple).
All values are seconds on the original synchronized session clock, not clip-relative
times. Add each task's name to `task`, remove invalid episodes or add missed ones,
then set `reviewed` to `true`. Episode IDs must be unique letters/digits/underscores/
hyphens. Do not change `session_id`. The recorded marker list is diagnostic;
export uses the `episodes` list.

Review is necessary because identical start/end markers are inherently ambiguous:
an even number of missed or extra triples can still produce plausible wrong pairs.
An odd number of markers produces a warning and no proposed episodes. Unmatched
camera triples are reported. Ordinary impacts, rebounds, weak taps and camera
clock drift need validation on actual recordings. There is no claim of perfect
automatic segmentation or automatic visual verification of tag visibility.

## Export and process each episode

```bash
uv run python scripts/convert.py \
  --input-insvs path/to/left_00.insv path/to/right_00.insv \
  --output-dir logs --episodes logs/episodes/<session>.json

uv run python scripts/estimate.py \
  --base logs/base/<session>_ep0001.rrd --anchor.tag-ids 0 1 --output-dir logs
uv run python scripts/state.py \
  --base logs/base/<session>_ep0001.rrd --output-dir logs
```

Repeat the last two commands for each exported episode. Video, IMU, exposure,
calibration and camera clock shifts are preserved. Video trimming retains decoder
GOP context, while IMU/exposure are trimmed to the selected interval. Training
windows exclude this extra context. Times remain on the original session clock.
Export also writes `episodes/<session>.reviewed.rrd` with the corrected green
spans; load that alongside the original full base to review your edited boundaries.

Keep each `base/<session>_epNNNN.episode.json` sidecar with its RRD. `state.py`
automatically uses it to find a one-second initial hold in the first five seconds,
use the reviewed ending boundary, and preserve the source session and task label
in output properties. `--episode-hold-seconds` and `--episode-hold-window` adjust
that check. The existing single-episode behavior is unchanged without a sidecar.
The initial hold still uses Specialist's velocity/proximity heuristic; tag
visibility must be checked in the video and successful pose estimation.

The splitter does not concatenate camera-created chapters. If a long recording
produces multiple INSV pairs per camera, those need separate handling first.
Audio synchronization still estimates one constant offset; check alignment near
both ends of long recordings. Start with a 2–3 minute pilot containing several
episodes, then validate a longer capture. No real camera capture was available
when this implementation was authored.
