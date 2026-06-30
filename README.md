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
uv sync --extra dev
```

Pulls `gtsam>=4.3a0` (the alpha is the only release with cp312 wheels), `numpy`,
`scipy`, `rerun-sdk`, `tyro`, `telemetry-parser`, `opencv-python`,
`pupil-apriltags`, `allantools`, plus dev tools `pytest`, `ruff`,
`mypy`, `pre-commit`.

`uv sync` also compiles the native MEI bundle-adjustment factor
(`_mei_factor_cpp`). This C++ port of the reprojection factor makes bundle
adjustment about 88× faster (about 35% off the total wall clock), with a
bit-identical optimum. It needs a C++17 compiler, `git` and Boost headers
(`libboost-dev`, or `brew install boost` on macOS). The gtsam wheel ships no
headers. So the first build clones the matching gtsam headers into a gitignored
`.build/` cache.

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

`scripts/track.py` is the pipeline. One `.insv` per tracker; with `N ≥ 2` the
clips are audio-synced via clap detection. Base tag ids must be distinct per
tracker (cross-tracker disambiguation).

Single tracker against a wall anchor:

```
uv run python scripts/track.py \
  --input-insvs path/to/clip.insv \
  --base-tag-ids <base-tag-id> --anchor.tag-id <anchor-tag-id> \
  --window.duration None --output-dir logs/run
```

Two grippers (cross-tag multi-tracker):

```
uv run python scripts/track.py \
  --input-insvs path/to/g0.insv path/to/g1.insv \
  --base-tag-ids <g0-base-tag-id> <g1-base-tag-id> --anchor.tag-id <anchor-tag-id> \
  --left-finger-tag-ids <g0-left-finger-tag-id> <g1-left-finger-tag-id> --right-finger-tag-ids <g0-right-finger-tag-id> <g1-right-finger-tag-id> \
  --window.duration None --output-dir logs/run
```

`track.py` writes a per-keyframe trajectory JSON and a Rerun `.rrd`.
`--log-frames-in-rrd` adds the camera image panels and the gripper URDF mesh.
View a result with `uv run rerun logs/run/track_<stamp>.rrd`.

## Layout

```
src/specialist/
├── vi_ba/       the pipeline — single_camera (per-camera prep), pipeline +
│                features (frontend), init (incremental SfM + VI alignment),
│                build / compose (graph) + the native cpp/ MEI factor,
│                finger_kinematics + urdf_joints (gripper joint recovery), imu,
│                tag_detection, optimize, landmark_cull,
│                configs, viz
├── record/      offset_v3 (factory lens intrinsics and body_T_lens)
└── loader/      parse_insv, mei_model, audio_sync

scripts/
├── track.py            the VI-BA driver (N ≥ 1 cameras)
├── imu.py              IMU rate / noise / Allan-deviation probe
├── video_imu_timing.py video↔IMU clock + rolling-shutter probe
└── _probe.py           summary output the probes share
```
