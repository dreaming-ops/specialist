"""The VI-BA pipeline recovers synthetic ground truth from a two-view init."""

from __future__ import annotations

import gtsam
import numpy as np
import numpy.typing as npt
from gtsam.symbol_shorthand import X
from specialist_camera import MeiParams, mei_project, mei_unproject

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.features import chain_tracks
from specialist.vi_ba.init import (
  body_poses_from_lens_poses,
  interpolate_camera_poses,
  triangulate_track_multiview,
  two_view_essential,
)
from specialist.vi_ba.optimize import optimize_graph
from tests.mei_fixtures import (
  WIDE_CALIBRATED_RADIUS,
  front_landmark_key,
  pinhole_mei_params,
  pinned_visual_graph,
)

Float64Array = npt.NDArray[np.float64]


def _project(
  body_pose: gtsam.Pose3, body_T_lens: gtsam.Pose3, xyz: Float64Array, params: MeiParams
) -> Float64Array:
  p_lens = body_pose.compose(body_T_lens).transformTo(
    gtsam.Point3(float(xyz[0]), float(xyz[1]), float(xyz[2]))
  )
  return mei_project(np.asarray(p_lens)[None, :], params)[0]


def test_pipeline_recovers_ground_truth_from_two_view_init() -> None:
  """Front lens, pinhole MEI: landmarks converge to truth up to the init-pair scale."""
  rng = np.random.default_rng(11)
  params = pinhole_mei_params()
  body_T_lens = gtsam.Pose3()

  # Ground-truth body poses: pure +x translation.
  true_poses = [
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 0.0)),
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.15, 0.0, 0.0)),
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.30, 0.0, 0.0)),
  ]
  n_keyframes = len(true_poses)

  n_lm = 20
  xyzs = rng.uniform(-0.4, 0.4, size=(n_lm, 3))
  xyzs[:, 2] = rng.uniform(1.5, 3.0, size=n_lm)

  # Keypoints stay in landmark-index order.
  keypoints = [
    np.array(
      [_project(body_pose, body_T_lens, xyzs[i], params) for i in range(n_lm)],
      dtype=np.float64,
    )
    for body_pose in true_poses
  ]

  matches_between = [
    np.stack([np.arange(n_lm), np.arange(n_lm)], axis=1).astype(np.int64)
    for _ in range(n_keyframes - 1)
  ]
  tracks = chain_tracks(keypoints, matches_between)
  assert len(tracks) == n_lm

  # Two-view init on the widest pair (0, n_keyframes-1).
  first, last = 0, n_keyframes - 1
  pix_a = np.array([tracks[tid][first] for tid in tracks])
  pix_b = np.array([tracks[tid][last] for tid in tracks])
  bears_a = mei_unproject(pix_a, params)
  bears_b = mei_unproject(pix_b, params)
  r_cw, t_cw, _ = two_view_essential(bears_a, bears_b, threshold=1e-3)

  lens_poses = interpolate_camera_poses(r_cw, t_cw, n_keyframes)
  body_poses_init = body_poses_from_lens_poses(lens_poses, body_T_lens)

  # Triangulate from earliest + latest observation per track.
  lens_poses = [pose.compose(body_T_lens) for pose in body_poses_init]
  lens_rotations = np.stack([pose.rotation().matrix() for pose in lens_poses])
  lens_centers = np.stack(
    [np.asarray(pose.translation(), dtype=np.float64) for pose in lens_poses]
  )
  points_init: dict[int, Float64Array] = {}
  for tid, obs in tracks.items():
    frames_sorted = sorted(obs.keys())
    f0, f1 = frames_sorted[0], frames_sorted[-1]
    b0 = mei_unproject(obs[f0][None, :], params)[0]
    b1 = mei_unproject(obs[f1][None, :], params)[0]
    # The two-view init's scale is arbitrary, so the depth gate is cheirality alone.
    xyz = triangulate_track_multiview(
      np.stack([b0, b1]),
      lens_rotations[[f0, f1]],
      lens_centers[[f0, f1]],
      min_landmark_depth=1e-6,
      max_landmark_depth=1e9,
    )
    if xyz is not None:
      points_init[tid] = xyz

  assert len(points_init) >= int(0.9 * n_lm)

  lens = LensSetup(
    body_T_lens=body_T_lens, params=params, calibrated_radius=WIDE_CALIBRATED_RADIUS
  )
  graph, values = pinned_visual_graph(
    body_poses_init, lens, {tid: tracks[tid] for tid in points_init}, points_init
  )
  initial_err = graph.error(values)
  result = optimize_graph(graph, values, max_iterations=50)

  assert result.final_error <= initial_err
  recovered = np.array(
    [np.asarray(result.values.atPoint3(front_landmark_key(tid))) for tid in points_init]
  )
  truth = np.array([xyzs[tid] for tid in points_init])

  # Centering and normalizing factor out the translation and the two-view init scale.
  def _normalized(p: np.ndarray) -> np.ndarray:
    c = p - p.mean(axis=0, keepdims=True)
    return c / np.linalg.norm(c)

  assert np.allclose(_normalized(recovered), _normalized(truth), atol=0.05)

  for i in range(n_keyframes):
    assert result.values.exists(X(i))
