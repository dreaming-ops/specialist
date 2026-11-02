# specialist

6DoF tracking of gripper-mounted Insta360 dual-fisheye + IMU cameras in an
AprilTag wall-anchor frame, for human fingertip data collection.

Each gripper carries an Insta360 (two back-to-back fisheye cameras) and an IMU.
AprilTags sit rigidly on the gripper body. A wall- or ceiling-mounted
anchor tag pins the world frame and the metric scale. One GTSAM factor graph
fuses each camera's visual odometry, IMU preintegration and tag-corner
reprojection. With more than one gripper, sightings of each other's tags tie
the cameras together. The output is globally consistent metric trajectories and
finger joint angles.

A transform is a_T_b: p_a = a_T_b · p_b, the pose of frame b in frame a. a_R_b
is its rotation, and a_t_b is b's origin in frame a.

## Install

```
uv sync
```

`uv sync`, and any `uv run`, installs the default `dev` and `gpu` groups. The
`gpu` group is the CUDA solve stack. It installs on Linux only, and
`scripts/estimate.py` exits without it.

`uv sync` also compiles the native MEI factor (`_mei_factor_cpp`). It needs a
C++17 compiler and `git`. The gtsam wheel ships no headers, so the first build
clones them into a gitignored `.build/` cache. It also downloads Boost headers
there when the system has none.

The `.so` is specific to the architecture and gitignored, so run `uv sync` on
each host. To rebuild after you edit the C++ source:

```
uv run build-mei-factor
```

## Test

```
uv run pytest
```

## Run

A capture is one camera's `.insv` pair. A session is the simultaneous captures,
one per camera, converted into one base recording. The session id names the
first capture's start instant and camera serial. It is also the Rerun recording
id, and so the segment id, of each layer file of the session.

A session goes through four steps:

- The raw captures convert once into a base recording.
- Every solve reads that recording and writes its two layers beside it.
- The state layer distills both into what a policy trains on.
- A catalog serves the state layer.

Records are laid out layer-first: one directory per layer, one file per session.
So you can fetch or regenerate any layer again without touching the rest:

```
base/<session-id>.rrd          # remuxed per-lens video, IMU, calibration, clock shifts
estimate/<session-id>.rrd      # the run's answers: poses, joint angles
viz/<session-id>.rrd           # what is only drawn: ribbon, overlays
assets/gripper-<sha>-<slot>.rrd # one slot's gripper model, shared by every session
state_s8_r100/<session-id>.rrd  # what the policy sees: 360x360 images, 100 Hz state
mocap/<session-id>.rrd         # ground-truth layer (eval branch)
```

**1. Convert.** Pass one `_00_` master `.insv` per camera; the converter finds
its `_10_` rear sibling. The first capture's clock is the session clock. The
converter audio-syncs the other captures onto it. The converter trims the
recording to the span that every capture covers.

```
uv run python scripts/convert.py \
  --input-insvs path/to/g0_00.insv path/to/g1_00.insv --output-dir logs
```

This writes `logs/base/<session-id>.rrd`. The converter stops when the
audio-sync peak is ambiguous.

**2. Estimate.** `estimate.py` reads the base recording. The recording carries
the camera slots, the session window and the session id, so you pass none of
them again. The solve discovers each camera's own base and finger tags from its
rear lens against the gripper URDF, so you name only the anchors. The first
`--anchor.tag-ids` entry is the reference anchor, which pins the world gauge.
`--urdf-path` overrides the packaged gripper description.

```
uv run python scripts/estimate.py --base logs/base/<session-id>.rrd \
  --anchor.tag-ids <anchor-tag-id> --output-dir logs
```

A single camera uses the same command, with no cross-camera factors.

The run writes the estimate and viz layers under the base recording's id. A
re-run refuses to overwrite either layer unless you pass `--overwrite`, so point
experimental runs at their own `--output-dir`. Every solve also writes
`logs/blueprints/specialist.rbl`. To look at a session, load its layers and the
gripper asset. The shared id merges them:

```
uv run rerun logs/blueprints/specialist.rbl logs/base/<id>.rrd \
  logs/estimate/<id>.rrd logs/viz/<id>.rrd \
  --asset logs/assets/gripper-<sha>.rrd
```

**3. State.** A trainer builds the policy's observation from the images and
past rows, and its action from future rows.

```
state ──┬── images ───── augment ── normalize ──────────────┐
        │                                                   ├──►  obs
        ├── past rows ── transform (egocentric, …) ─────────┘
        │
        └── future rows ─ transform (egocentric, …) ──►  action   (exact, unaugmented)
```

`state.py` writes the layer from the base recording and the estimate, keyed by
hand under `state/left` and `state/right`. A lone camera rides the right hand.
Per hand, the layer holds:

- the rear lens, block-averaged to 360x360 (`--scale 8`) and blanked outside
  the fisheye disc, as one JPEG for every frame inside the usable window;
- the gripper's base-tag pose and twist in the world frame on every row,
  interpolated between the estimate's samples;
- the coupled finger angle.

The window comes from the operator's bracket gestures: a hold at the start and
a tap at the end. `--margin` seconds stay clear of each. The writer finds which
camera rode which hand.

A trainer indexes the layer by the integer `row` timeline, `round(t * --f-row)`.
Every row keeps its true time beside that label. Both hands cover one dense row
range.

```
uv run python scripts/state.py --base logs/base/<session-id>.rrd --output-dir logs
```

The layer lands in `logs/state_s8_r100/<session-id>.rrd`. The directory names
the configuration, so several configurations sit side by side in one dataset.

**4. Register** (optional) one state configuration on a Rerun catalog, for
dataset inspection and training. The one argument is the configuration's
directory. The catalog dataset takes the name of the dataset root, the parent of
that directory. It holds only the state layer and refuses a second
configuration. It installs the state dashboard as its default blueprint and also
writes it to `logs/blueprints/state.rbl`.

```
uv run python scripts/catalog.py logs/state_s8_r100 \
  --catalog-url rerun+http://host:port
```

## Layout

```
packages/
├── specialist_camera/   mei (MEI projection both ways, rescaling), image (the
│                        lens as the policy sees it)
└── specialist_dataset/  the dataset contract: schema (entity paths, timelines,
                         sessions), read (layers as numpy), frames (lens frame
                         times, H.264 decoding), poses, fiducials, state
                         (state-stream helpers)

src/specialist/
├── vi_ba/       the VI-BA solve: frontend, initialization, factor graph, the
│                native cpp/ MEI factor, LM, the joint multi-camera solve
├── record/      the rrd writers and the solve's reader: base + video +
│                offset_v3 (base recording), layer (estimate and viz
│                layers), asset (gripper model), session (a base recording
│                as the solve reads it)
└── loader/      parse_insv, capture, audio_sync

scripts/
├── convert.py          captures → one session base recording
├── estimate.py         the VI-BA driver (N ≥ 1 cameras) → estimate + viz layers
├── state.py            base + estimate → the state layer
├── catalog.py          a dataset's state layer → a Rerun catalog dataset
├── imu.py              IMU rate / noise / Allan-deviation probe
├── video_imu_timing.py video↔IMU clock + rolling-shutter probe
└── _probe.py           summary output the probes share
```
