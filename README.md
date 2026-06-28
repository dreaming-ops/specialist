# specialist

Multi-camera AprilTag SLAM for human data collection.

Each hand-held gripper carries an Insta360 (two back-to-back fisheye cameras) plus an AprilTag mounted on the body. A large wall- or ceiling-mounted anchor tag provides the world gauge. Per-tracker VIO supplies odometry; the lenses see the anchor and the other cameras' tags. A GTSAM `NonlinearFactorGraph` fuses all three signals into globally-consistent, inter-tracker-consistent trajectories.

The pipeline runs on real Insta360 X3 captures end to end: the loader parses `.insv` video + telemetry, and `scripts/track.py` runs per-tracker visual-inertial bundle adjustment against the wall anchor. See **Probes** below for the real-data characterization scripts the pipeline is built on.

A transform is a_T_b: p_a = a_T_b · p_b, the pose of frame b in frame a. a_R_b
is its rotation, and a_t_b is b's origin in frame a.

## Install

```
uv sync --extra dev
```

This pulls `gtsam>=4.3a0` (the alpha — it is the only release with cp312 wheels), `numpy`, `scipy`, `rerun-sdk`, `tyro`, `telemetry-parser`, `opencv-python`, `pupil-apriltags`, `allantools`, and `imageio-ffmpeg` (wheel-bundled ffmpeg for audio clap-sync), plus the dev tools `pytest`, `ruff`, `mypy`, and `pre-commit`. `specialist-description` (gripper URDF assets) is a private SSH git dependency.

## Test

```
uv run pytest
```

Unit tests per `loader` / `vi_ba` module, plus integration tests covering the VI-BA pipeline on a synthetic scene and IMU preintegration on a real 30-second stationary window. VI-BA real-data assertions use tiny fixtures (40-float `offset_v3` JSON, 30-s IMU `.npz`) committed under `tests/fixtures/vi_ba/`.

## Visualize

Open a saved `.rrd` in the Rerun viewer:

```
uv run rerun logs/track_<stamp>.rrd
```

## Layout

```
src/specialist/
├── loader/      parse_insv, mei_model, audio_sync
├── record/      offset_v3 (factory lens intrinsics and body_T_lens)
├── vi_ba/       per-tracker visual-inertial bundle adjustment — single_camera
│                (per-camera prep), pipeline + features (frontend), init
│                (incremental SfM + VI alignment), build / compose (graph),
│                mei_factor, imu, tag_detection, optimize,
│                configs, viz
├── graph/       factors (shared GTSAM factor constructors)
└── synthesis/   tag_detection (the cross-tracker observation record)

tests/
├── unit/        loader + vi_ba module tests
└── integration/ VI-BA pipeline on a synthetic scene, IMU preintegration

scripts/
├── imu.py              sample rate, jitter, gaps, Allan deviation
├── video_imu_timing.py LK-flow / gyro cross-correlation + rolling-shutter shear
└── track.py            VI-BA driver for any number of cameras (N ≥ 1)
```

## Probes

`scripts/` holds real-data probes that answer open questions about Insta360 X3 captures before the SLAM pipeline is built on top. Each probe is a standalone `tyro`-driven CLI that writes a Rerun `.rrd` and a JSON summary to `logs/` and prints a one-line verdict. Pass `--input-insv <path>` per run.

```
uv run python scripts/imu.py --input-insv path/to/stationary.insv
uv run python scripts/video_imu_timing.py --input-insv path/to/yaw.insv
uv run python scripts/track.py --input-insvs path/to/manipulation.insv --base-tag-ids <base-tag-id> --anchor.tag-id <anchor-tag-id> --window.master-start 2.0 --window.duration 10.0
```

