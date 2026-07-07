"""The map cull drops high-residual, behind-camera and degenerate-parallax landmarks."""

from __future__ import annotations

import json
from pathlib import Path

import gtsam
import numpy as np
import pytest

from specialist.loader.mei_model import MeiParams, mei_project
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.landmark_cull import cull_outlier_landmarks


def _lens() -> LensSetup:
  return LensSetup(
    body_T_lens=gtsam.Pose3(),
    params=MeiParams(
      fx=500.0,
      fy=500.0,
      cx=320.0,
      cy=240.0,
      k1=0.0,
      k2=0.0,
      k3=0.0,
      p1=0.0,
      p2=0.0,
      xi=0.0,
    ),
  )


def _in_lens(pose: gtsam.Pose3, lens: LensSetup, point) -> np.ndarray:
  return np.asarray(pose.compose(lens.body_T_lens).transformTo(gtsam.Point3(*point)))


def _project(lens: LensSetup, pose: gtsam.Pose3, point) -> np.ndarray:
  return mei_project(_in_lens(pose, lens, point)[None, :], lens.params)[0]


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


def _load_front(path: str):
  d = json.loads(Path(path).read_text())
  name = d["slots"][0]
  lf = d["lens_by_tracker"][name]["front"]
  lens = LensSetup(
    body_T_lens=gtsam.Pose3(
      gtsam.Rot3(np.array(lf["body_T_lens"]["R"])),
      gtsam.Point3(*lf["body_T_lens"]["t"]),
    ),
    params=MeiParams(**lf["params"]),
  )
  poses = [
    gtsam.Pose3(gtsam.Rot3(np.array(p["R"])), gtsam.Point3(*p["t"]))
    for p in d["body_poses_by_tracker"][name]
  ]
  tracks = {
    int(tid): {int(kf): np.array(xy, float) for kf, xy in obs.items()}
    for tid, obs in d["tracks_by_tracker"][name]["front"].items()
  }
  points = {
    int(tid): np.array(p, float)
    for tid, p in d["landmarks_by_tracker"][name]["front"].items()
  }
  return lens, poses, tracks, points


def _rmse(ids, tracks, points, poses, lens) -> float:
  errs = []
  for tid in ids:
    for kf, xy in tracks[tid].items():
      if 0 <= kf < len(poses):
        pl = _in_lens(poses[kf], lens, points[tid])
        if pl[2] > 1e-3:
          errs.append(
            float(np.linalg.norm(mei_project(pl[None, :], lens.params)[0] - xy))
          )
  return float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0


def _max_view_depth(ids, tracks, points, poses, lens) -> float:
  d = 0.0
  for tid in ids:
    for kf in tracks[tid]:
      if 0 <= kf < len(poses):
        pl = _in_lens(poses[kf], lens, points[tid])
        d = max(d, abs(float(pl[2])))
  return d


@pytest.mark.skipif(not Path("qdata/03.json").exists(), reason="needs local run data")
def test_qdata_cull_cuts_rmse_and_runaways_keeps_bulk() -> None:
  lens, poses, tracks, points = _load_front("qdata/03.json")
  kept = cull_outlier_landmarks(points, tracks, poses, lens)
  # Bulk survives (cm-noisy landmarks are geometrically consistent).
  assert len(kept) > 0.9 * len(points)
  # RMSE collapses to ~1px once the mismatch tail (and bad obs) is gone.
  assert _rmse(points.keys(), tracks, points, poses, lens) > 5.0
  assert _rmse(kept.keys(), kept, points, poses, lens) < 1.5
  # Runaway-depth landmarks are removed.
  assert _max_view_depth(points.keys(), tracks, points, poses, lens) > 50.0
  assert _max_view_depth(kept.keys(), kept, points, poses, lens) < 10.0
