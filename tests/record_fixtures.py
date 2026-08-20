"""Capture metadata, a video encoder and a solved run for the record and viz tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.audio_sync import resolve_ffmpeg
from specialist.loader.parse_insv import InsvData, LensIntrinsics
from specialist.record.layer import write_solution_layers
from specialist.vi_ba.summary import CameraSummary, Summary
from specialist.vi_ba.viz import CameraVizData, LensVizData
from specialist.vi_ba.viz_build import RunOutputs

BASE_TAG = gtsam.Pose3(gtsam.Rot3.Rz(0.3), gtsam.Point3(0.01, -0.02, 0.05))


def synthetic_insv() -> InsvData:
  """Both lenses share one ``offset_v3`` block, and the IMU rolls at 90 deg/s."""
  block = [
    1.9,  # xi
    5000.0,
    5001.0,
    2970.0,
    2971.0,  # fx fy cx cy
    0.5,
    -0.3,
    90.0,  # yaw pitch roll deg
    0.01,
    0.02,
    0.03,  # t
    0.1,
    -0.05,
    0.01,
    0.001,
    -0.002,  # k1..k3 p1 p2
    11904.0,
    5952.0,
    0.0,  # frame w h unknown
  ]
  return InsvData(
    imu_timestamps=np.array([-0.5, 0.0, 0.001, 0.002]),
    gyro=np.radians([[90.0, 0.0, 0.0]] * 4),
    accel=np.array([[0.0, 0.0, 9.81]] * 4),
    intrinsics=LensIntrinsics(
      camera_matrix=np.eye(3),
      distortion_coeffs=np.zeros(6),
      calib_width=11904,
      calib_height=5952,
      distortion_model="mei",
    ),
    rolling_shutter_readout=0.0157,
    offset_v3=np.array([2.0, *block, *block, 0.0]),
    gyro_calib_numbers=np.array([0.001, 0.002, 0.003, 0.01, 0.02, 0.03]),
    capture={
      "serial_number": "SYNTHCAM01",
      "camera_type": "Insta360 X3",
      "fw_version": "v0.0.0",
      "photo_rot": np.eye(3).ravel(),
    },
    exposure_timestamps=np.array([0.0, 1.0 / 30.0, 2.0 / 30.0]),
    exposure_durations=np.array([0.002, 0.002, 0.004]),
    ae_timestamps=np.array([-0.5, 0.1]),
    ae_stats={"iso_value": np.array([400.0, 800.0])},
  )


def encode_clip(
  path: Path,
  n_frames: int,
  *,
  rate: float = 30.0,
  gop: int | None = None,
  bframes: bool = False,
) -> Path:
  """A tiny H.264 lavfi clip; with ``bframes`` its decode order is not display order."""
  options = ([] if gop is None else ["-g", str(gop)]) + (
    [] if bframes else ["-bf", "0"]
  )
  subprocess.run(
    [
      resolve_ffmpeg(),
      "-y",
      "-loglevel",
      "error",
      "-f",
      "lavfi",
      "-i",
      f"testsrc=duration={n_frames / rate}:size=64x64:rate={rate:g}",
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      *options,
      str(path),
    ],
    check=True,
  )
  return path


def viz_tracker(
  name: str, *, offset: float = 0.0, highrate: bool = False
) -> CameraVizData:
  """One solved tracker with one of everything a layer holds."""
  n = 4
  times: npt.NDArray[np.float64] = np.arange(n, dtype=np.float64) * 0.1
  poses = [
    gtsam.Pose3(gtsam.Rot3.Ry(0.05 * i), gtsam.Point3(offset + 0.1 * i, 0.02 * i, 1.0))
    for i in range(n)
  ]
  data = CameraVizData(
    name=name,
    body_poses=poses,
    keyframe_times=times,
    keyframe_frames=np.arange(n, dtype=np.int64) * 3,
    velocities=[np.array([0.1, 0.0, 0.0])] * n,
    body_T_base_tag=BASE_TAG,
  )
  data.lenses = {
    "front": LensVizData(points={0: np.zeros(3), 1: np.ones(3)}),
    "rear": LensVizData(points={2: np.full(3, 0.5)}),
  }
  if highrate:
    hr_t = np.linspace(0.0, times[-1], 10 * n + 1)
    zeros = np.zeros_like(hr_t)
    data.highrate_times = hr_t
    data.highrate_positions = np.stack([hr_t, zeros, zeros], axis=1)
    data.highrate_quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(hr_t), 1))
    data.highrate_vels = np.ones((len(hr_t), 3))
  return data


def make_summary(*names: str, cross_tracker_obs: int = 7) -> Summary:
  return Summary(
    sources=tuple(f"{name}.insv" for name in names),
    slots=names,
    cross_tracker_obs=cross_tracker_obs,
    initial_graph_error=10.0,
    final_graph_error=1.0,
    solver_iterations=5,
    # An empty tuple writes no column; one flag proves the transport.
    non_default_flags=("frontend.flow_scale=0.5",),
  )


def write_run(
  root: Path,
  session_id: str,
  cameras: list[CameraVizData],
  *,
  per_tracker: list[CameraSummary] | None = None,
  summary: Summary | None = None,
  mask_rects: tuple[tuple[int, int, int, int], ...] = (),
  urdf_path: Path | None = None,
  open_viewer: bool = False,
  frame_period: float | None = None,
) -> tuple[Path, Path]:
  """Write both layers under ``root`` and return the estimate and viz paths."""
  track_path = root / "track" / f"{session_id}.rrd"
  viz_path = root / "viz" / f"{session_id}.rrd"
  for tracker in cameras:
    if mask_rects:
      tracker.lenses.setdefault("rear", LensVizData()).static_mask_rects = list(
        mask_rects
      )
    if urdf_path is not None:
      tracker.urdf_root_poses = list(tracker.body_poses)
  write_solution_layers(
    track_path=track_path,
    viz_path=viz_path,
    outputs=RunOutputs(
      viz_trackers=cameras,
      per_tracker=per_tracker or [],
      world_T_anchor={0: gtsam.Pose3()},
    ),
    summary=summary or make_summary(*(t.name for t in cameras)),
    reference_tag_id=0,
    anchor_tag_size=0.1,
    session_id=session_id,
    # -1 disables tag masking; these cameras carry no tag detections anyway.
    mask_pad_px=-1,
    urdf_path=urdf_path,
    frame_period=frame_period,
    open_viewer=open_viewer,
  )
  return track_path, viz_path


def urdf_file(tmp_path: Path) -> Path:
  """A two-link robot whose visual geometry needs no mesh file on disk."""
  path = tmp_path / "robot.urdf"
  link = (
    '<link name="{n}"><visual><geometry><box size="0.1 0.1 0.1"/></geometry>'
    "</visual></link>"
  )
  path.write_text(
    '<?xml version="1.0"?><robot name="gripper">'
    + link.format(n="base")
    + link.format(n="tip")
    + '<joint name="j0" type="revolute"><parent link="base"/><child link="tip"/>'
    '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>'
    "</joint></robot>"
  )
  return path
