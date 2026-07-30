"""The map cull drops high-residual, behind-camera and degenerate-parallax landmarks."""

from __future__ import annotations

import gtsam
import numpy as np

from specialist.loader.mei_model import mei_project
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.landmark_cull import cull_outlier_landmarks
from tests.mei_fixtures import pinhole_mei_params


def _lens() -> LensSetup:
  return LensSetup(body_T_lens=gtsam.Pose3(), params=pinhole_mei_params())


def _project(lens: LensSetup, pose: gtsam.Pose3, point) -> np.ndarray:
  return mei_project(
    np.asarray(pose.compose(lens.body_T_lens).transformTo(gtsam.Point3(*point)))[
      None, :
    ],
    lens.params,
  )[0]


def _baseline_poses() -> list[gtsam.Pose3]:
  # Two cameras with a 0.3 m lateral baseline → ample parallax on a 2 m point.
  return [
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 0.0)),
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.3, 0.0, 0.0)),
  ]


def test_keeps_consistent_drops_high_residual_mismatch() -> None:
  lens = _lens()
  poses = _baseline_poses()
  pt = np.array([0.0, 0.0, 2.0])
  tracks = {
    1: {kf: _project(lens, poses[kf], pt) for kf in (0, 1)},
    2: {kf: _project(lens, poses[kf], pt) + np.array([60.0, 40.0]) for kf in (0, 1)},
  }
  kept = cull_outlier_landmarks({1: pt, 2: pt}, tracks, poses, lens)
  assert set(kept) == {1}
  assert set(kept[1]) == {0, 1}


def test_drops_landmark_with_too_few_surviving_obs() -> None:
  lens = _lens()
  poses = _baseline_poses()
  pt = np.array([0.0, 0.0, 2.0])
  # One clean obs + one garbage obs → only 1 survives the residual gate (<2).
  tracks = {
    1: {0: _project(lens, poses[0], pt), 1: _project(lens, poses[1], pt) + 80.0}
  }
  kept = cull_outlier_landmarks({1: pt}, tracks, poses, lens)
  assert not kept


def test_drops_degenerate_low_parallax() -> None:
  lens = _lens()
  # One center for both views gives zero parallax, though both reproject perfectly.
  poses = [gtsam.Pose3(), gtsam.Pose3()]
  pt = np.array([0.0, 0.0, 2.0])
  tracks = {1: {kf: _project(lens, poses[kf], pt) for kf in (0, 1)}}
  kept = cull_outlier_landmarks({1: pt}, tracks, poses, lens)
  assert not kept


def test_drops_behind_camera() -> None:
  lens = _lens()
  poses = _baseline_poses()
  pt = np.array([0.0, 0.0, -2.0])  # behind both cameras
  tracks = {1: {kf: np.array([320.0, 240.0]) for kf in (0, 1)}}
  kept = cull_outlier_landmarks({1: pt}, tracks, poses, lens)
  assert not kept
