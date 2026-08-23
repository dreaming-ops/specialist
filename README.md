# specialist

6DoF tracking of gripper-mounted Insta360 dual-fisheye + IMU cameras in an
AprilTag wall-anchor frame, for human fingertip data collection.

Each gripper carries an Insta360 (two back-to-back fisheye cameras) plus an IMU,
with AprilTags rigidly mounted on the gripper body. A wall- or ceiling-mounted
anchor tag pins the world frame and metric scale. One GTSAM factor graph fuses
each camera's visual odometry, IMU preintegration, and tag-corner reprojection;
with multiple grippers, sightings of each other's tags tie the cameras
together. The output is globally-consistent metric trajectories and finger
joint angles.

A transform is a_T_b: p_a = a_T_b · p_b, the pose of frame b in frame a. a_R_b
is its rotation, and a_t_b is b's origin in frame a.

## Install

```
uv sync
```

`uv sync`, and any `uv run`, installs the default `dev` and `gpu` groups. The
`gpu` group is the CUDA solve stack. It installs on Linux only, and
`scripts/track.py` exits without it.

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
track/<session-id>.rrd   # the solve's answers: poses, joint angles
viz/<session-id>.rrd     # what is only drawn: ribbon, overlays, meshes
mocap/<session-id>.rrd   # ground-truth layer (eval branch)
```

**1. Convert.** One `_00_` master `.insv` per tracker (its `_10_` rear sibling
is found automatically). The first capture's clock is the session clock; the
rest are audio-synced onto it, and the recording is trimmed to the span every
capture covers.

```
uv run python scripts/convert.py \
  --input-insvs path/to/g0_00.insv path/to/g1_00.insv \
  --session-id <session-id> --output-dir logs
```

This writes `logs/base/<session-id>.rrd`: the remuxed per-lens video with its
real PTS, the native-rate IMU, the calibration, and the audio offsets. Pass
`--audio-offsets` to supply the shifts by hand when a take's clips share no
audible event.

**2. Solve.** `track.py` reads the base recording — it carries the tracker
names, the session window, and the session id, so none of them are passed
again. Each tracker's own base and finger tags are discovered from its rear
lens against the gripper URDF, so only the anchors are named; the first
`--anchor.tag-ids` entry is the reference anchor that pins the world gauge. The
URDF is required: `--urdf-path` overrides the packaged gripper description.

```
uv run python scripts/track.py --base logs/base/<session-id>.rrd \
  --anchor.tag-ids <anchor-tag-id> --output-dir logs
```

A single tracker is the same command with no cross-tracker factors.
`--keyframe-every` overrides the automatic ~0.10 s keyframe stride, and
`--no-open-viewer` keeps a headless box from spawning one.

The run writes two layers sharing the base recording's id:
`logs/track/<session-id>.rrd` carries what a consumer reads — body poses,
keyframe poses, finger joint angles, the solved anchor poses
and the run summary — and `logs/viz/<session-id>.rrd` carries what is only
drawn, the trajectory ribbon, the image overlays, the landmarks, the robot
meshes and the diagnostic scalars. Re-running a session refuses to overwrite
either unless `--overwrite` is passed, so point experimental runs at their own
`--output-dir`. It spawns a viewer on the way out; to re-open a session later,
load the layers beside their base — the shared id merges them — with
`uv run rerun logs/base/<id>.rrd logs/track/<id>.rrd logs/viz/<id>.rrd`.

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
src/specialist/
├── vi_ba/       the solve: single_camera (per-camera prep), tracks +
│                features + gpu_flow (frontend), init (incremental SfM + VI
│                alignment), build / compose (graph) + the native cpp/ MEI
│                factor, optimize + gpu_lm (LM), reference_solve (the
│                multi-tracker world gauge), finger_kinematics + urdf_joints
│                (gripper joint recovery), imu,
│                tag_detection, landmark_cull, highrate,
│                geometry, configs, summary, viz + viz_build
├── record/      the rrd data layer — schema (entity paths), base + video
│                (session recording), layer (track and viz layers), session +
│                frames + read (reading them back), offset_v3 (factory lens
│                calibration)
└── loader/      parse_insv, mei_model, capture, audio_sync

scripts/
├── convert.py          captures → one session base recording
├── track.py            the VI-BA driver (N ≥ 1 cameras) → track + viz layers
├── register.py         a layer-first dataset root → a Rerun catalog dataset
├── imu.py              IMU rate / noise / Allan-deviation probe
├── video_imu_timing.py video↔IMU clock + rolling-shutter probe
└── _probe.py           summary output the probes share
```
