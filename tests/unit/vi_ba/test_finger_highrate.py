"""Pinning a per-frame finger-angle series to the keyframe solution."""

from __future__ import annotations

import numpy as np

from specialist.vi_ba.finger_kinematics import pin_angles_to_keyframes
from tests.mei_fixtures import synthetic_tag_detection, wide_mei_params


def test_pinned_series_passes_through_the_keyframe_angles() -> None:
  frame_t = np.arange(0.0, 1.01, 1 / 30)
  raw = np.sin(2 * np.pi * frame_t) + 0.2  # detection series with a +0.2 bias
  kf_t = frame_t[::3]
  solved = np.sin(2 * np.pi * kf_t)  # BA removed the bias at keyframes
  pinned = pin_angles_to_keyframes(frame_t, raw, kf_t, solved)
  for t, v in zip(kf_t, solved, strict=True):
    assert abs(pinned[np.argmin(np.abs(frame_t - t))] - v) < 1e-9


def test_between_keyframes_the_detection_shape_survives() -> None:
  """The pinned series keeps the detection's shape at close to its full height."""
  frame_t = np.arange(0.0, 0.11, 1 / 30)
  raw = np.zeros_like(frame_t)
  raw[1] = 0.5  # a fast flick between keyframes 0.0 and 0.1
  kf_t = np.array([0.0, 0.1])
  solved = np.array([0.0, 0.0])
  pinned = pin_angles_to_keyframes(frame_t, raw, kf_t, solved)
  assert abs(pinned[1] - 0.5) < 0.2


def test_nan_keyframes_leave_the_raw_series_untouched() -> None:
  frame_t = np.arange(0.0, 0.11, 1 / 30)
  raw = np.full_like(frame_t, 0.3)
  kf_t = np.array([0.0, 0.1])
  solved = np.array([np.nan, np.nan])
  pinned = pin_angles_to_keyframes(frame_t, raw, kf_t, solved)
  assert np.allclose(pinned, raw)


def test_frame_angles_recover_a_dense_synthetic_motion() -> None:
  """Angles come back within scan resolution, and undetected frames stay absent."""
  import gtsam

  from specialist.vi_ba.build import LensSetup
  from specialist.vi_ba.finger_kinematics import build_finger_reproj
  from specialist.vi_ba.urdf_joints import RevoluteJoint

  params = wide_mei_params()
  joint_pose = gtsam.Pose3(gtsam.Rot3.Rx(2.7), gtsam.Point3(0, 0, 0.25))
  joint_T_tag0 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.02, 0.03, 0.0))
  joint = RevoluteJoint(
    base_T_joint=joint_pose, lower=-0.7, upper=0.7, joint_T_tag0=joint_T_tag0
  )
  lens = LensSetup(body_T_lens=gtsam.Pose3(), params=params)

  true = np.concatenate([np.linspace(-0.1, 0.5, 9), [np.nan], np.linspace(0.5, 0.2, 5)])
  dets: list[list] = []
  for th in true:
    if np.isnan(th):
      dets.append([])  # an undetected frame
      continue
    body_T_tag = joint_pose.compose(
      gtsam.Pose3(gtsam.Rot3.Rz(float(th)), gtsam.Point3(0, 0, 0))
    ).compose(joint_T_tag0)
    dets.append([synthetic_tag_detection(body_T_tag, 2004, 0.05, params)])

  fr = build_finger_reproj(joint, dets, lens, 2004, gtsam.Pose3(), 0.05)
  assert fr is not None
  frames = list(fr.observations)
  angles = [o.theta_seed for o in fr.observations.values()]
  assert 9 not in frames
  assert len(frames) == len(true) - 1
  for f, a in zip(frames, angles, strict=True):
    assert abs(a - true[f]) < 0.05
