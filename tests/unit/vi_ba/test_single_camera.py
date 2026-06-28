"""Tests for the anchored loop-closure merge (``_add_revisit_landmarks``)."""

from __future__ import annotations

import gtsam
import numpy as np

from specialist.loader.mei_model import MeiParams, mei_project
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.single_camera import _add_revisit_landmarks


def _params() -> MeiParams:
  return MeiParams(
    fx=160.0,
    fy=160.0,
    cx=120.0,
    cy=120.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
    xi=0.8,
  )


def test_add_revisit_landmarks_reliable_views_resist_drift() -> None:
  """With ``anchor_kfs`` the landmark triangulates from well-localized views alone."""
  params = _params()
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
    2,
    5.0,
    "t",
    "lc",
    anchor_kfs={1, 4},
  )
  fp_all: dict[int, np.ndarray] = {}
  fk_all: dict[int, dict[int, np.ndarray]] = {}
  _add_revisit_landmarks(
    fp_all, fk_all, {99: dict(obs)}, body_poses, lens, 2, 5.0, "t", "lc"
  )

  assert 99 in fp_rel and 99 in fp_all
  err_rel = float(np.linalg.norm(fp_rel[99] - p_world))
  err_all = float(np.linalg.norm(fp_all[99] - p_world))
  assert err_rel < 1e-2  # reliable views -> truth
  assert err_rel < err_all  # the drifted view pulls the naive landmark off truth
  # The drifted tail still feeds the BA factors, which pull its pose onto the structure.
  assert 5 in fk_rel[99]
  assert 7 in fk_rel
