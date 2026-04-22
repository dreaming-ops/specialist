# specialist

Multi-camera AprilTag SLAM for human data collection.

Each hand-held gripper carries an Insta360 (two back-to-back fisheye cameras) plus an AprilTag mounted on the body. A large wall- or ceiling-mounted anchor tag provides the world gauge. Per-tracker VIO supplies odometry; the lenses see the anchor and the other cameras' tags. A GTSAM `NonlinearFactorGraph` fuses all three signals into globally-consistent, inter-tracker-consistent trajectories.

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

## Layout

```
src/specialist/
├── loader/      parse_insv, mei_model, audio_sync
├── record/      offset_v3 (factory lens intrinsics and body_T_lens)
└── vi_ba/       per-tracker visual-inertial bundle adjustment — single_camera
                 (per-camera prep), pipeline + features (frontend), init
                 (two-view bootstrap + VI alignment), build / compose (graph),
                 mei_factor, imu, tag_detection, optimize,
                 configs, viz

tests/
├── unit/        loader + vi_ba module tests
└── integration/ VI-BA pipeline on a synthetic scene, IMU preintegration
```

