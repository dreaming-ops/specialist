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

A session goes through three steps. The raw captures convert once into a base
recording; every solve reads that recording and writes its two layers beside
it; registration puts them all on a catalog. Records are laid out layer-first —
one directory per layer, one file per session — so a whole layer registers
with one glob and any layer can be refetched or regenerated without touching
the rest:

```
base/<session-id>.rrd    # remuxed per-lens video, IMU, calibration, clock shifts
estimate/<session-id>.rrd # the run's answers: poses, joint angles
viz/<session-id>.rrd     # what is only drawn: ribbon, overlays, meshes
mocap/<session-id>.rrd   # ground-truth layer (eval branch)
```

**1. Convert.** Pass one `_00_` master `.insv` per camera; the converter finds
its `_10_` rear sibling. The first capture's clock is the session clock. The
converter audio-syncs the other captures onto it. The converter trims the
recording to the span that every capture covers.

```
uv run python scripts/convert.py \
  --input-insvs path/to/g0_00.insv path/to/g1_00.insv \
  --session-id <session-id> --output-dir logs
```

This writes `logs/base/<session-id>.rrd`: the remuxed per-lens video with its
real PTS, the native-rate IMU, the calibration, and each capture's clock shift,
found by correlating the captures' audio; conversion stops when the audio peak is
ambiguous.

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
`--keyframe-every` overrides the automatic ~0.10 s keyframe stride, and
`--no-open-viewer` keeps a headless box from spawning one.

The run writes two layers sharing the base recording's id:
`logs/estimate/<session-id>.rrd` carries what a consumer reads — body poses,
keyframe poses, finger joint angles, the solved anchor poses, the landmarks,
the diagnostic series and the run summary — and `logs/viz/<session-id>.rrd`
carries what is only drawn: the trajectory ribbon, the image overlays, the
factor rays, the axis triads and the robot meshes. Re-running a session
refuses to overwrite either unless `--overwrite` is passed, so point
experimental runs at their own `--output-dir`.
It spawns a viewer on the way out; to re-open a session later,
load the layers beside their base — the shared id merges them — with
`uv run rerun logs/base/<id>.rrd logs/estimate/<id>.rrd logs/viz/<id>.rrd`.

**3. Register** (optional) the pair on a Rerun catalog, which installs the
standard dashboard as the catalog dataset default. A layer whose recording id matches
no base segment is refused; a layer attaches only to the base segment that
shares its recording id.

```
uv run python scripts/register.py --catalog-url rerun+http://host:port \
  --dataset <catalog-dataset> --root logs
```

## Layout

```
packages/
├── specialist_camera/   mei (MEI projection both ways, rescaling)
└── specialist_dataset/  the dataset contract: schema (entity paths, timelines,
                         sessions), read (layers as numpy), frames (lens frame
                         times, H.264 decoding), poses, fiducials

src/specialist/
├── vi_ba/       the solve: single_camera (per-camera prep), tracks +
│                features + gpu_flow (frontend), init (incremental SfM + VI
│                alignment), build / compose (graph) + the native cpp/ MEI
│                factor, optimize + gpu_lm (LM), reference_solve (the
│                multi-tracker world gauge), finger_kinematics + urdf_joints
│                (gripper joint recovery), imu,
│                tag_detection, landmark_cull, highrate,
│                geometry, configs, summary, viz + viz_build
├── record/      the rrd writers and the solve's reader: base + video +
│                offset_v3 (base recording), layer (estimate and viz
│                layers), session (a base recording as the solve reads it)
└── loader/      parse_insv, capture, audio_sync

scripts/
├── convert.py          captures → one session base recording
├── estimate.py         the VI-BA driver (N ≥ 1 cameras) → estimate + viz layers
├── register.py         a layer-first dataset root → a Rerun catalog dataset
├── imu.py              IMU rate / noise / Allan-deviation probe
├── video_imu_timing.py video↔IMU clock + rolling-shutter probe
└── _probe.py           summary output the probes share
```
