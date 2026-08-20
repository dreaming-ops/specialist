"""``incremental_sfm`` recovers a synthetic camera sweep up to a similarity."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rscp

from specialist.vi_ba.init import incremental_sfm


def _scene(
  n_keyframes: int = 30, n_points: int = 250, noise: float = 0.0, seed: int = 0
):
  rng = np.random.default_rng(seed)
  # camera sweeps laterally 0.6 m and pans ~20° (T_world_camera = R_i, C_i)
  centers = np.stack(
    [np.linspace(-0.3, 0.3, n_keyframes), np.zeros(n_keyframes), np.zeros(n_keyframes)],
    axis=1,
  )
  yaw = np.linspace(-0.18, 0.18, n_keyframes)
  rots = [Rscp.from_euler("y", a).as_matrix() for a in yaw]  # cam→world
  # point cloud: near surface (parallax) + far wall, all in +z half-space
  pts = np.column_stack(
    [
      rng.uniform(-0.6, 0.6, n_points),
      rng.uniform(-0.35, 0.35, n_points),
      rng.uniform(0.3, 3.0, n_points),
    ]
  )
  bearings: dict[int, dict[int, np.ndarray]] = {}
  for tid, x in enumerate(pts):
    obs: dict[int, np.ndarray] = {}
    for k in range(n_keyframes):
      x_cam = rots[k].T @ (x - centers[k])  # world→cam
      if x_cam[2] <= 0.05:
        continue
      if abs(x_cam[0] / x_cam[2]) > 1.4 or abs(x_cam[1] / x_cam[2]) > 1.4:
        continue
      b = x_cam / np.linalg.norm(x_cam)
      if noise:
        b = b + rng.normal(0, noise, 3)
        b = b / np.linalg.norm(b)
      obs[k] = b
    if len(obs) >= 3:
      bearings[tid] = obs
  return bearings, centers, rots


def _umeyama_ate(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
  mu_s, mu_d = src.mean(0), dst.mean(0)
  sc, dc = src - mu_s, dst - mu_d
  u, dv, vt = np.linalg.svd(dc.T @ sc / len(src))
  w = np.eye(3)
  if np.linalg.det(u @ vt) < 0:
    w[-1, -1] = -1
  r = u @ w @ vt
  s = np.trace(np.diag(dv) @ w) / ((sc**2).sum() / len(src))
  t = mu_d - s * r @ mu_s
  return np.linalg.norm((s * (r @ src.T).T + t) - dst, axis=1)


@pytest.mark.parametrize(
  ("noise", "min_registered", "ate_fraction"),
  [(0.0, 0.8, 0.02), (1e-3, 0.7, 0.08)],
)
def test_recovers_the_trajectory_up_to_a_similarity(
  noise: float, min_registered: float, ate_fraction: float
) -> None:
  """Umeyama-aligned camera centers match truth from exact and from noisy bearings."""
  n = 30
  bearings, centers, _ = _scene(n_keyframes=n, noise=noise)
  lens_poses, landmark_map = incremental_sfm(bearings)
  assert len(lens_poses) >= int(min_registered * n), (
    f"only registered {len(lens_poses)}/{n}"
  )
  if not noise:
    assert len(landmark_map) >= 50
  ks = sorted(lens_poses)
  rec = np.array([np.asarray(lens_poses[k].translation()) for k in ks])
  ate = _umeyama_ate(rec, centers[ks])
  extent = float(np.linalg.norm(centers.max(0) - centers.min(0)))
  assert np.median(ate) < ate_fraction * extent, (
    f"median ATE {np.median(ate):.4f} of {extent:.3f}"
  )


def test_no_bootstrap_returns_empty():
  # too few shared tracks → cannot bootstrap
  bearings = {0: {0: np.array([0.0, 0.0, 1.0]), 1: np.array([0.0, 0.0, 1.0])}}
  lens_poses, landmark_map = incremental_sfm(bearings)
  assert lens_poses == {} and landmark_map == {}


def _observe(points: np.ndarray, centers: np.ndarray, frames: range | list[int]):
  """Unit bearings from identity-rotation cameras at ``centers``, in ``frames`` only."""
  return {
    tid: {k: (x - centers[k]) / float(np.linalg.norm(x - centers[k])) for k in frames}
    for tid, x in enumerate(points)
  }


def _weak_opening_scene(seed: int = 3):
  """A capture still over keyframes 0-2, whose surviving tracks start at keyframe 1."""
  rng = np.random.default_rng(seed)
  centers = np.zeros((8, 3))
  centers[:, 0] = [0.0, 1.0e-4, 2.0e-4, 0.1, 0.2, 0.3, 0.4, 0.5]

  def cloud(n: int) -> np.ndarray:
    return np.column_stack(
      [
        rng.uniform(-0.6, 0.6, n),
        rng.uniform(-0.4, 0.4, n),
        rng.uniform(1.5, 4.0, n),
      ]
    )

  opening = _observe(cloud(12), centers, [0, 1, 2])
  sweep = _observe(cloud(30), centers, range(1, 8))
  bearings = dict(opening)
  bearings.update({100 + tid: obs for tid, obs in sweep.items()})
  return bearings, centers


def test_bootstrap_bases_at_frame_zero_when_it_qualifies():
  """With a parallax-qualifying partner, keyframe 0 stays the base and world origin."""
  bearings, centers, _ = _scene(n_keyframes=30, noise=0.0)
  lens_poses, _ = incremental_sfm(bearings)
  assert 0 in lens_poses
  assert np.allclose(lens_poses[0].matrix(), np.eye(4), atol=1e-12)
  assert len(lens_poses) >= int(0.8 * len(centers))


def test_bootstrap_slides_base_past_a_weak_opening():
  """The base slides to the first keyframe spanning a parallax-qualifying pair."""
  bearings, centers = _weak_opening_scene()
  lens_poses, landmark_map = incremental_sfm(bearings, min_shared=8)
  # World = the slid base; the degenerate opening keyframe stays out.
  assert 0 not in lens_poses
  assert np.allclose(lens_poses[1].matrix(), np.eye(4), atol=1e-12)
  assert set(lens_poses) == set(range(1, len(centers)))
  assert len(landmark_map) >= 20
  ks = sorted(lens_poses)
  rec = np.array([np.asarray(lens_poses[k].translation()) for k in ks])
  ate = _umeyama_ate(rec, centers[ks])
  extent = float(np.linalg.norm(centers.max(0) - centers.min(0)))
  assert np.median(ate) < 0.02 * extent
