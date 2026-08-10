"""``vi_ba.single_camera``: anchor-PnP seeds and the loop-closure landmark merge."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import gtsam
import numpy as np

from specialist.loader.mei_model import mei_project
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.configs import FrontendCfg
from specialist.vi_ba.single_camera import (
  _add_revisit_landmarks,
  _anchor_metric_scale,
  _anchor_pose_seed,
)
from specialist.vi_ba.tracks import run_lens_frontend
from tests.frame_fixtures import translating_frames
from tests.mei_fixtures import narrow_mei_params


def _cpu_frontend_cfg() -> FrontendCfg:
  """A small feature budget and full-resolution cv2 flow, keeping the tests GPU-free."""
  return FrontendCfg(
    orb_n_features=300,
    max_corners=300,
    gpu_flow=False,
    flow_scale=1.0,
    ransac_threshold=3.0,
  )


def _tracks_equal(
  a: dict[int, dict[int, np.ndarray]], b: dict[int, dict[int, np.ndarray]]
) -> bool:
  if a.keys() != b.keys():
    return False
  for tid in a:
    if a[tid].keys() != b[tid].keys():
      return False
    for kf in a[tid]:
      if not np.array_equal(a[tid][kf], b[tid][kf]):
        return False
  return True


def test_frontend_parallel_matches_serial() -> None:
  params = narrow_mei_params()
  cfg = _cpu_frontend_cfg()
  front = translating_frames(6, 2, seed=1)
  rear = translating_frames(6, 3, seed=2)

  with ThreadPoolExecutor(max_workers=2) as pool:
    fut_front = pool.submit(run_lens_frontend, front, params, cfg)
    fut_rear = pool.submit(run_lens_frontend, rear, params, cfg)
    _, pf_tracks = fut_front.result()
    _, pr_tracks = fut_rear.result()
  _, sf_tracks = run_lens_frontend(front, params, cfg)
  _, sr_tracks = run_lens_frontend(rear, params, cfg)

  assert _tracks_equal(pf_tracks, sf_tracks)
  assert _tracks_equal(pr_tracks, sr_tracks)
  assert len(sf_tracks) > 0 and len(sr_tracks) > 0


def test_anchor_pose_seed_reconstructs_metric_body_poses() -> None:
  """The seed's world is the anchor-tag frame, through ``body_T_lens``."""
  ext = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.1, 0.0, 0.0), gtsam.Point3(0.05, 0.0, 0.0))
  n = 10
  body = [
    gtsam.Pose3(
      gtsam.Rot3.Rodrigues(0.0, 0.02 * k, 0.0), gtsam.Point3(0.1 * k, 0.0, 0.5)
    )
    for k in range(n)
  ]
  # lens_T_tag = (world_T_lens)^-1 with world = tag and world_T_lens = body o ext.
  anchor_obs = {k: {"front": body[k].compose(ext).inverse()} for k in range(n)}
  seed = _anchor_pose_seed(anchor_obs, ext, n)
  assert seed is not None and len(seed) == n
  for k in range(n):
    assert np.allclose(seed[k].matrix(), body[k].matrix(), atol=1e-9)


def test_anchor_pose_seed_none_when_too_few_anchor_keyframes() -> None:
  anchor_obs = {0: {"front": gtsam.Pose3()}, 1: {"front": gtsam.Pose3()}}
  assert _anchor_pose_seed(anchor_obs, gtsam.Pose3(), 100) is None


def test_anchor_metric_scale_recovers_scale() -> None:
  """With VO = metric positions times ``c``, the estimator returns ``1/c``."""
  ext = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.1, 0.0, 0.0), gtsam.Point3(0.05, 0.0, 0.0))
  n = 10
  metric_body = [
    gtsam.Pose3(
      gtsam.Rot3.Rodrigues(0.0, 0.02 * k, 0.0),
      gtsam.Point3(0.1 * k, 0.03 * k, 0.5),
    )
    for k in range(n)
  ]
  anchor_obs = {k: {"front": metric_body[k].compose(ext).inverse()} for k in range(n)}
  c = 0.5
  vo_body = [
    gtsam.Pose3(
      metric_body[k].rotation(),
      gtsam.Point3(*(c * np.asarray(metric_body[k].translation()))),
    )
    for k in range(n)
  ]
  lens = LensSetup(body_T_lens=ext, params=narrow_mei_params())
  scale = _anchor_metric_scale(vo_body, anchor_obs, lens, lens)
  assert scale is not None
  assert abs(scale - 1.0 / c) < 1e-9


def test_anchor_metric_scale_none_when_degenerate() -> None:
  """Fewer than 3 anchored keyframes or a zero-spread trajectory yields ``None``."""
  lens = LensSetup(body_T_lens=gtsam.Pose3(), params=narrow_mei_params())
  two = {k: {"front": gtsam.Pose3()} for k in range(2)}
  assert _anchor_metric_scale([gtsam.Pose3()] * 2, two, lens, lens) is None
  # Three anchored views but the VO is stationary → zero spread → None.
  stationary = [gtsam.Pose3()] * 3
  obs = {
    k: {"front": gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.1 * k, 0.0, 0.5)).inverse()}
    for k in range(3)
  }
  assert _anchor_metric_scale(stationary, obs, lens, lens) is None


def test_add_revisit_landmarks_reliable_views_resist_drift() -> None:
  """With ``anchor_kfs`` the landmark triangulates from well-localized views alone."""
  params = narrow_mei_params()
  ext = gtsam.Pose3()
  lens = LensSetup(body_T_lens=ext, params=params)
  n = 6
  body_poses = [gtsam.Pose3() for _ in range(n)]
  body_poses[4] = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.3, 0.0, 0.0))
  # tail kf 5: its true pose has a parallax baseline, but the ESTIMATE is drifted.
  true_pose5 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.6, 0.0, 0.0))
  body_poses[5] = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.6, 0.2, 0.0))
  p_world = np.array([0.2, 0.1, 1.5])

  def _proj(pose: gtsam.Pose3) -> np.ndarray:
    pc = np.asarray(
      pose.compose(ext).transformTo(gtsam.Point3(*p_world)), dtype=np.float64
    )
    return mei_project(pc[None, :], params)[0]

  # tail observation is taken from the TRUE pose (what the camera actually saw)
  obs = {1: _proj(body_poses[1]), 4: _proj(body_poses[4]), 5: _proj(true_pose5)}

  fp_rel: dict[int, np.ndarray] = {}
  fk_rel: dict[int, dict[int, np.ndarray]] = {7: {0: np.array([1.0, 1.0])}}
  _add_revisit_landmarks(
    fp_rel,
    fk_rel,
    {99: dict(obs)},
    body_poses,
    lens,
    min_track_length=2,
    max_landmark_depth=5.0,
    slot="t",
    anchor_kfs={1, 4},
  )
  fp_all: dict[int, np.ndarray] = {}
  fk_all: dict[int, dict[int, np.ndarray]] = {}
  _add_revisit_landmarks(
    fp_all,
    fk_all,
    {99: dict(obs)},
    body_poses,
    lens,
    min_track_length=2,
    max_landmark_depth=5.0,
    slot="t",
  )

  assert 99 in fp_rel and 99 in fp_all
  err_rel = float(np.linalg.norm(fp_rel[99] - p_world))
  err_all = float(np.linalg.norm(fp_all[99] - p_world))
  assert err_rel < 1e-2  # reliable views -> truth
  assert err_rel < err_all  # the drifted view pulls the naive landmark off truth
  # The drifted tail still feeds the BA factors, which pull its pose onto the structure.
  assert 5 in fk_rel[99]
  assert 7 in fk_rel
