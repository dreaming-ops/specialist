"""Tag poses, the shared detector and window following in ``tag_detection``."""

from __future__ import annotations

import gc
import math
import resource
import threading
import time

import cv2
import gtsam
import numpy as np
import pytest
from specialist_camera import MeiParams, mei_project, mei_unproject
from specialist_dataset.fiducials import tag_corners_in_frame, tag_corners_local

from specialist.vi_ba import tag_detection
from specialist.vi_ba.tag_detection import (
  TagDetection,
  estimate_lens_T_tag,
  lens_T_tag_per_kf,
)


def _params() -> MeiParams:
  """A super-fisheye MEI close to a real rear lens (downsampled)."""
  return MeiParams(
    fx=400.0,
    fy=400.0,
    cx=480.0,
    cy=480.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
    xi=1.9,
  )


def _project_tag(
  lens_T_tag: gtsam.Pose3, tag_size: float, params: MeiParams
) -> np.ndarray:
  """Project the four tag corners in pupil_apriltags's order (BL, BR, TR, TL)."""
  return mei_project(tag_corners_in_frame(lens_T_tag.matrix(), tag_size), params)


def _ippe_solutions(
  corners_px: np.ndarray, tag_size: float, params: MeiParams
) -> list[gtsam.Pose3]:
  """Both IPPE solutions, enumerated by cv2 directly."""
  bearings = mei_unproject(corners_px.astype(np.float64), params)
  norm = (bearings[:, :2] / bearings[:, 2:3]).reshape(-1, 1, 2)
  obj = tag_corners_local(tag_size).reshape(-1, 1, 3)
  n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
    obj,
    norm,
    np.eye(3, dtype=np.float64),
    np.zeros(4, dtype=np.float64),
    flags=cv2.SOLVEPNP_IPPE,
  )
  out: list[gtsam.Pose3] = []
  for i in range(n):
    rmat, _ = cv2.Rodrigues(rvecs[i])
    t = np.asarray(tvecs[i], dtype=np.float64).reshape(3)
    out.append(gtsam.Pose3(gtsam.Rot3(rmat), gtsam.Point3(*t)))
  return out


def _rot_geodesic(a: gtsam.Pose3, b: gtsam.Pose3) -> float:
  return float(np.linalg.norm(gtsam.Rot3.Logmap(a.rotation().between(b.rotation()))))


def _camera_facing_tag_pose(
  tilt_axis: tuple[float, float, float], tilt_angle: float, distance: float
) -> gtsam.Pose3:
  """A tag ``distance`` m ahead facing the camera, tilted so PnP is non-degenerate."""
  axis = np.asarray(tilt_axis, dtype=np.float64)
  axis = axis / np.linalg.norm(axis)
  rot = gtsam.Rot3.Rx(math.pi).compose(
    gtsam.Rot3.Rodrigues(*(axis * tilt_angle).tolist())
  )
  return gtsam.Pose3(rot, gtsam.Point3(0.0, 0.0, distance))


def test_recovers_a_known_pose() -> None:
  params = _params()
  tag_size = 0.10
  lens_T_tag = _camera_facing_tag_pose((1.0, 0.5, 0.0), 0.25, 0.30)
  corners = _project_tag(lens_T_tag, tag_size, params)
  est = estimate_lens_T_tag(corners, tag_size, params)
  assert est is not None
  assert np.allclose(est.translation(), lens_T_tag.translation(), atol=1e-3)
  assert np.allclose(est.rotation().matrix(), lens_T_tag.rotation().matrix(), atol=1e-3)


def _flip_scenario() -> tuple[
  MeiParams, float, gtsam.Pose3, gtsam.Pose3, np.ndarray, np.ndarray, np.ndarray
]:
  """Two views of one tag, the second pitched 0.9 rad from the first."""
  params = _params()
  tag_size = 0.10
  cam1_T_tag = _camera_facing_tag_pose((1.0, 0.0, 0.0), 0.25, 0.60)
  delta = gtsam.Rot3.Rx(0.90).matrix()
  cam2_T_tag = gtsam.Pose3(gtsam.Rot3(delta), gtsam.Point3(0, 0, 0)).compose(cam1_T_tag)
  corners1 = _project_tag(cam1_T_tag, tag_size, params)
  corners2 = _project_tag(cam2_T_tag, tag_size, params)
  return params, tag_size, cam1_T_tag, cam2_T_tag, corners1, corners2, delta


def test_reprojection_ranking_picks_the_true_twin() -> None:
  params, tag_size, _cam1_T_tag, cam2_T_tag, _, corners2, _ = _flip_scenario()
  if len(_ippe_solutions(corners2, tag_size, params)) < 2:
    pytest.skip("IPPE returned a single solution for this geometry")
  est = estimate_lens_T_tag(corners2, tag_size, params)
  assert est is not None
  assert _rot_geodesic(est, cam2_T_tag) < 0.05


def test_anchor_chain_resolves_each_frame_on_its_own() -> None:
  """``lens_T_tag_per_kf`` survives a fast turn and an unsighted keyframe."""
  params, tag_size, cam1_T_tag, cam2_T_tag, corners1, corners2, _ = _flip_scenario()
  tags_per_kf = [
    [TagDetection(tag_id=1, center=corners1.mean(0), corners=corners1)],
    [],
    [TagDetection(tag_id=1, center=corners2.mean(0), corners=corners2)],
  ]
  out = lens_T_tag_per_kf(tags_per_kf, 1, tag_size, params)
  assert set(out) == {0, 2}
  assert _rot_geodesic(out[0], cam1_T_tag) < 0.05
  assert _rot_geodesic(out[2], cam2_T_tag) < 0.05


def test_tag_detection_never_runs_concurrently() -> None:
  """Two threads inside the native detect at once segfault."""
  peak = 0
  live = 0
  guard = threading.Lock()

  def hold() -> None:
    nonlocal peak, live
    with tag_detection._detect_slot():
      with guard:
        live += 1
        peak = max(peak, live)
      time.sleep(0.005)
      with guard:
        live -= 1

  threads = [threading.Thread(target=hold) for _ in range(16)]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()
  assert peak == 1


def test_the_detector_is_built_once_and_never_dropped() -> None:
  """Native teardown corrupts the heap beside another detector."""
  assert tag_detection._shared_detector() is tag_detection._shared_detector()


def test_detector_teardown_reclaims_its_memory() -> None:
  def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if peak > 10**7 else peak / 1024

  tag_detection._shared_detector()  # warm any one-off allocation
  gc.collect()
  before = rss_mb()
  for _ in range(3):
    detector = tag_detection._OrderedTeardownDetector(families="tagStandard41h12")
    del detector
    gc.collect()
  assert rss_mb() - before < 200.0


def test_pnp_never_returns_a_non_finite_pose() -> None:
  """The estimator declines the non-finite pose IPPE gives a near-degenerate quad."""
  params = _params()
  world_T_tag = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 1.2))
  for k in range(24):
    body = gtsam.Pose3(
      gtsam.Rot3.Rodrigues(0.02 * k, 0.01 * k, 0.0),
      gtsam.Point3(0.01 * k, 0.0, 0.0),
    )
    pixels = mei_project(
      tag_corners_in_frame(body.inverse().compose(world_T_tag).matrix(), 0.1),
      params,
    )
    if not np.all(np.isfinite(pixels)):
      continue
    pose = estimate_lens_T_tag(pixels, 0.1, params)
    assert pose is None or np.all(np.isfinite(pose.matrix())), k
