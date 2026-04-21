# specialist

Multi-camera AprilTag SLAM for human data collection.

Each hand-held gripper carries an Insta360 (two back-to-back fisheye cameras) plus an AprilTag mounted on the body. A large wall- or ceiling-mounted anchor tag provides the world gauge. Per-tracker VIO supplies odometry; the lenses see the anchor and the other cameras' tags. A GTSAM `NonlinearFactorGraph` fuses all three signals into globally-consistent, inter-tracker-consistent trajectories.

## Install

```
uv sync --extra dev
```

This pulls `gtsam>=4.3a0` (the alpha — it is the only release with cp312 wheels), `numpy`, `scipy`, `rerun-sdk`, `tyro`, `telemetry-parser`, `opencv-python`, `pupil-apriltags`, and `allantools`, plus the dev tools `pytest`, `ruff`, `mypy`, and `pre-commit`. `specialist-description` (gripper URDF assets) is a private SSH git dependency.

## Test

```
uv run pytest
```
