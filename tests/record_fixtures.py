"""Capture metadata, a video encoder and a solved run for the record and viz tests."""

import subprocess
from pathlib import Path

import gtsam
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from specialist.loader.audio_sync import resolve_ffmpeg
from specialist.loader.parse_insv import InsvData
from specialist.record.layer import write_estimate_layers
from specialist.vi_ba.summary import CameraSummary, Summary
from specialist.vi_ba.tag_detection import TagDetection
from specialist.vi_ba.viz import CameraEstimate, LensEstimate
from specialist.vi_ba.viz_build import RunEstimate

BASE_TAG = gtsam.Pose3(gtsam.Rot3.Rz(0.3), gtsam.Point3(0.01, -0.02, 0.05))


def synthetic_insv() -> InsvData:
  """Both lenses share one ``offset_v3`` block, and the IMU rolls at 90 deg/s."""
  block = [
    1.9,  # xi
    5000.0, 5001.0, 2970.0, 2971.0,  # fx fy cx cy
    0.5, -0.3, 90.0,  # yaw pitch roll deg
    0.01, 0.02, 0.03,  # t
    0.1, -0.05, 0.01, 0.001, -0.002,  # k1..k3 p1 p2
    11904.0, 5952.0, 0.0,  # frame w h unknown
  ]  # fmt: skip
  return InsvData(
    imu_timestamps=np.array([-0.5, 0.0, 0.001, 0.002]),
    gyro=np.radians([[90.0, 0.0, 0.0]] * 4),
    accel=np.array([[0.0, 0.0, 9.81]] * 4),
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


# The trajectory turns SPIN rad/s about world z, its body tilted TILT rad about its x.
SPIN = 0.7
TILT = 0.3


def camera_estimate(
  slot: str,
  *,
  offset: float = 0.0,
  hold: float = 0.0,
  rate: float = 1000.0,
) -> CameraEstimate:
  """One solved camera with every layer field, sliding along x after ``hold`` s."""
  n = 4
  times: npt.NDArray[np.float64] = np.arange(n, dtype=np.float64) * 0.1
  poses = [
    gtsam.Pose3(gtsam.Rot3.Ry(0.05 * i), gtsam.Point3(offset + 0.1 * i, 0.02 * i, 1.0))
    for i in range(n)
  ]
  hr_t = np.arange(round(times[-1] * rate) + 1) / rate
  if hold:
    hr_t = np.concatenate([np.arange(-0.5 - hold, -0.5 + 1e-9, 0.025), hr_t])
  zeros = np.zeros_like(hr_t)
  turn = Rotation.from_rotvec(np.outer(SPIN * hr_t, [0.0, 0.0, 1.0])) * (
    Rotation.from_rotvec([TILT, 0.0, 0.0])
  )
  data = CameraEstimate(
    slot=slot,
    highrate_times=hr_t,
    highrate_positions=np.stack([offset + np.maximum(hr_t, 0.0), zeros, zeros], axis=1),
    highrate_quats=turn.as_quat(),
    highrate_vels=np.stack([hr_t >= 0.0, zeros, zeros], axis=1),
    body_poses=poses,
    keyframe_times=times,
    keyframe_frames=np.arange(n, dtype=np.int64) * 3,
    velocities=[np.array([0.1, 0.0, 0.0])] * n,
    gyro_biases=[np.array([0.001, 0.002, 0.003])] * n,
    accel_biases=[np.array([0.01, 0.02, 0.03])] * n,
    body_T_base_tag=BASE_TAG,
    body_T_rear_lens=gtsam.Pose3(
      gtsam.Rot3.RzRyRx(0.0, 0.0, np.pi), gtsam.Point3(0.0, -0.026, 0.0)
    ),
    base_tag_id=2000,
    joint_tags={"left_finger": 2001, "right_finger": 2002},
    anchor_obs_lines=[np.zeros((2, 2, 3)) for _ in range(n)],
    cross_camera_lines=[np.ones((2, 2, 3)) for _ in range(n)],
    joint_frame_angles={
      "left_finger": (times, np.full(n, 0.1)),
      "right_finger": (times, np.full(n, -0.1)),
    },
    tcp_in_base_tag=(0.0, 0.0, 0.02),
  )
  rmse = [0.4 + 0.01 * i for i in range(n)]
  # Two tags per keyframe, one of them the anchor, each with a corner quad and center.
  corners = np.array([[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]])
  detections = [
    [
      TagDetection(0, np.array([20.0, 20.0]), corners + i),
      TagDetection(2000, np.array([60.0, 20.0]), corners + 40 + i),
    ]
    for i in range(n)
  ]
  data.lenses = {
    "front": LensEstimate(
      points={0: np.zeros(3), 1: np.ones(3)},
      rmse_per_kf=rmse,
      keypoints=[np.full((3, 2), 5.0 + i) for i in range(n)],
      rejected_keypoints=[np.full((1, 2), 7.0 + i) for i in range(n)],
      landmark_reprojections=[np.full((3, 2), 5.2 + i) for i in range(n)],
      tag_detections=detections,
      tag_axes=[np.zeros((3, 2, 2)) + i for i in range(n)],
    ),
    "rear": LensEstimate(
      tag_detections=detections, tag_axes=[np.zeros((3, 2, 2)) + i for i in range(n)]
    ),
  }
  return data


def write_run(
  root: Path,
  session_id: str,
  cameras: list[CameraEstimate],
  *,
  front_rmse: list[float] | None = None,
  urdf_path: Path,
  frame_period: float,
) -> tuple[Path, Path]:
  """Write both layers under ``root`` and return the estimate and viz paths."""
  per_camera = [
    CameraSummary(
      c.slot,
      len(c.keyframe_times),
      sum(1 for lines in c.anchor_obs_lines if len(lines)),
      front_rmse=rmse,
      base_tag_id=c.base_tag_id,
      seat_rotation=(0.0, 0.0, 0.05),
      seat_anchor_obs=12,
      seat_cross_obs=7,
    )
    for c, rmse in zip(cameras, front_rmse or [0.4] * len(cameras), strict=True)
  ]
  estimate_path = root / "estimate" / f"{session_id}.rrd"
  viz_path = root / "viz" / f"{session_id}.rrd"
  write_estimate_layers(
    estimate_path=estimate_path,
    viz_path=viz_path,
    estimate=RunEstimate(
      cameras=cameras,
      per_camera=per_camera,
      world_T_anchor={0: gtsam.Pose3()},
      reference_tag_id=0,
    ),
    summary=Summary(
      cross_camera_obs=7,
      initial_graph_error=10.0,
      final_graph_error=1.0,
      solver_iterations=5,
      anchor_move_translation=float("nan"),
      anchor_move_rotation=float("nan"),
    ),
    anchor_tag_size=0.1,
    session_id=session_id,
    urdf_path=urdf_path,
    frame_period=frame_period,
    timeline_start=0.0,
  )
  return estimate_path, viz_path


def urdf_file(tmp_path: Path) -> Path:
  """A gripper URDF with the solve's finger joint names and mesh-free visuals."""
  path = tmp_path / "robot.urdf"
  link = (
    '<link name="{n}"><visual><geometry><box size="0.1 0.1 0.1"/></geometry>'
    "</visual></link>"
  )
  joint = (
    '<joint name="{n}" type="revolute"><parent link="base"/><child link="{n}"/>'
    '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>'
    "{mimic}</joint>"
  )
  # The left finger mirrors the right, as on the real gripper.
  mimic = {
    "left_finger": '<mimic joint="right_finger" multiplier="-1"/>',
    "right_finger": "",
  }
  # Every tag rides a fixed joint, so the frame a tag names is one the model places.
  tag = (
    '<link name="{n}_tag"/>'
    '<joint name="{n}_tag" type="fixed"><parent link="{n}"/><child link="{n}_tag"/>'
    '<origin xyz="0 0.01 0.02" rpy="0 0 0"/></joint>'
  )
  fingers = ("left_finger", "right_finger")
  path.write_text(
    '<?xml version="1.0"?><robot name="gripper">'
    + "".join(link.format(n=n) for n in ("base", *fingers))
    + "".join(joint.format(n=n, mimic=mimic[n]) for n in fingers)
    + "".join(tag.format(n=n) for n in ("base", *fingers))
    + "</robot>"
  )
  return path
