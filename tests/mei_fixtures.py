"""MEI lens regimes, their projection, and a visual-only graph for the vi_ba tests."""

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, L, V, X
from specialist_camera import MeiParams, mei_project
from specialist_dataset.fiducials import tag_corners_in_frame

from specialist.vi_ba.build import LensSetup, add_mei_factors
from specialist.vi_ba.tag_detection import TagDetection


def front_landmark_key(track_id: int) -> int:
  return int(L(track_id))


def wide_mei_params() -> MeiParams:
  """The ξ=1.6 omni regime the gripper-tag tests project in."""
  return _undistorted(800.0, 960.0, 960.0, 1.6)


def synthetic_tag_detection(
  lens_T_tag: gtsam.Pose3, tag_id: int, tag_size: float, params: MeiParams
) -> TagDetection | None:
  """An ideal detection of the tag at this pose, or None if a corner is non-finite."""
  px = mei_project(tag_corners_in_frame(lens_T_tag.matrix(), tag_size), params)
  if not np.all(np.isfinite(px)):
    return None
  return TagDetection(tag_id=tag_id, center=px.mean(0), corners=px)


# Past every synthetic projection, so each sighting carries the corner-noise floor.
WIDE_CALIBRATED_RADIUS = 1000.0


def _undistorted(f: float, cx: float, cy: float, xi: float) -> MeiParams:
  return MeiParams(
    fx=f, fy=f, cx=cx, cy=cy, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0, xi=xi
  )


def pinhole_mei_params() -> MeiParams:
  return _undistorted(500.0, 320.0, 240.0, 0.0)


def narrow_mei_params() -> MeiParams:
  """240x240 undistorted MEI with a mild xi."""
  return _undistorted(160.0, 120.0, 120.0, 0.8)


def distorted_mei_params(xi: float) -> MeiParams:
  """1920x1920 lightly distorted MEI; ``xi`` sets its depth into super-fisheye."""
  return MeiParams(
    fx=800.0, fy=800.0, cx=960.0, cy=960.0,
    k1=0.02, k2=-0.01, k3=0.003, p1=0.001, p2=-0.002, xi=xi,
  )  # fmt: skip


def single_camera_keys() -> dict[str, object]:
  """The keying and containers a graph needs, for a test with one camera."""
  return {
    "pose_key_fn": lambda i: int(X(i)),
    "velocity_key_fn": lambda i: int(V(i)),
    "bias_key_fn": lambda i: int(B(i)),
    "front_landmark_key_fn": front_landmark_key,
    "graph": gtsam.NonlinearFactorGraph(),
    "values": gtsam.Values(),
  }


def project_point(
  world_T_body: gtsam.Pose3,
  body_T_lens: gtsam.Pose3,
  world_xyz: np.ndarray,
  params: MeiParams,
) -> np.ndarray:
  world_T_lens = world_T_body.compose(body_T_lens)
  p_lens = world_T_lens.transformTo(gtsam.Point3(*world_xyz))
  return mei_project(np.asarray(p_lens)[None, :], params)[0]


def pinned_visual_graph(
  body_poses: list[gtsam.Pose3],
  front: LensSetup,
  front_tracks: dict,
  front_points: dict,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """One lens's projection factors over a pose chain whose pinned ends fix the gauge."""
  graph = gtsam.NonlinearFactorGraph()
  values = gtsam.Values()
  pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
  tight = gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)
  weak = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 0.3))

  for i, pose in enumerate(body_poses):
    values.insert(int(X(i)), pose)
  n = len(body_poses)
  graph.push_back(gtsam.PriorFactorPose3(int(X(0)), body_poses[0], tight))
  if n > 1:
    graph.push_back(gtsam.PriorFactorPose3(int(X(n - 1)), body_poses[-1], tight))
  for i in range(1, n - 1):
    graph.push_back(gtsam.PriorFactorPose3(int(X(i)), body_poses[i], weak))

  pose_keys: list[int] = []
  point_keys: list[int] = []
  pixels: list[np.ndarray] = []
  for tid, xyz in front_points.items():
    lm_key = front_landmark_key(tid)
    values.insert(lm_key, gtsam.Point3(*np.asarray(xyz, dtype=np.float64)))
    for kf, pt in front_tracks[tid].items():
      pose_keys.append(int(X(kf)))
      point_keys.append(lm_key)
      pixels.append(pt)
  add_mei_factors(graph, pose_keys, point_keys, pixels, pixel_noise, front)
  return graph, values
