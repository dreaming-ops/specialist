"""AprilTag detection and lens_T_tag from planar PnP through the MEI lens."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import gtsam
import numpy as np
import numpy.typing as npt
from pupil_apriltags import Detector
from specialist_camera import MeiParams, mei_project, mei_unproject
from specialist_dataset.fiducials import TAG_FAMILY, tag_corners_local

from specialist.vi_ba.features import cpu_budget

Float64Array = npt.NDArray[np.float64]

# Cap on the detector's own thread pool.
_DETECT_THREADS = 48
_detect_lock = threading.Lock()
_detector: Detector | None = None
_parallel = False
_local = threading.local()

__all__ = [
  "TagDetection",
  "lens_T_tag_per_kf",
  "configure_detector",
  "detect_all_tags",
  "estimate_lens_T_tag",
  "tag_center_bearing",
]


def configure_detector(parallel: bool) -> None:
  """Use per-thread detectors (``parallel``, ~160 MB each) or one behind a lock."""
  global _parallel
  _parallel = parallel


@contextmanager
def _detect_slot() -> Iterator[None]:
  """Hold the shared detector's lock unless detectors are per-thread."""
  if _parallel:
    yield
    return
  with _detect_lock:
    yield


class _OrderedTeardownDetector(Detector):  # type: ignore[misc]
  """Free the detector before its families, fixing the binding's use-after-free."""

  def __del__(self) -> None:
    detector = getattr(self, "tag_detector_ptr", None)
    if detector is None:
      return
    self.libc.apriltag_detector_destroy.restype = None
    self.libc.apriltag_detector_destroy(detector)
    for family, tf in self.tag_families.items():
      destroy = getattr(self.libc, f"{family}_destroy")
      destroy.restype = None
      destroy(tf)
    self.tag_detector_ptr = None


def _shared_detector() -> Detector:
  """This thread's detector when parallel, else the process-wide one."""
  global _detector
  if _parallel:
    got = getattr(_local, "detector", None)
    if got is None:
      got = _OrderedTeardownDetector(
        families=TAG_FAMILY, nthreads=max(1, min(_DETECT_THREADS, cpu_budget()) // 2)
      )
      _local.detector = got
    return got
  with _detect_lock:
    if _detector is None:
      _detector = _OrderedTeardownDetector(
        families=TAG_FAMILY, nthreads=min(_DETECT_THREADS, cpu_budget())
      )
    return _detector


@dataclass(frozen=True)
class TagDetection:
  """One AprilTag hit in one keyframe: id, center, corners (CCW image order)."""

  tag_id: int
  center: Float64Array  # (2,)
  corners: Float64Array  # (4, 2)


def tag_center_bearing(det: TagDetection, params: MeiParams) -> Float64Array:
  """Lens-frame bearing of a tag's center, the cheirality ray for MEI factors."""
  bearing = mei_unproject(
    np.asarray(det.center, dtype=np.float64).reshape(1, 2), params
  )[0]
  return np.asarray(bearing, dtype=np.float64)


def estimate_lens_T_tag(
  corners_px: Float64Array,
  tag_size: float,
  params: MeiParams,
) -> gtsam.Pose3 | None:
  """``lens_T_tag`` by planar IPPE PnP on the unprojected corners, else ``None``."""
  bearings = mei_unproject(corners_px.astype(np.float64), params)
  if not np.all(np.isfinite(bearings)) or np.any(bearings[:, 2] <= 1e-3):
    return None
  normalized = (bearings[:, :2] / bearings[:, 2:3]).astype(np.float64).reshape(-1, 1, 2)
  obj = tag_corners_local(tag_size).reshape(-1, 1, 3)
  k = np.eye(3)
  dist = np.zeros(4)
  _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
    obj, normalized, k, dist, flags=cv2.SOLVEPNP_IPPE
  )
  poses: list[gtsam.Pose3] = []
  for rvec, tvec in zip(rvecs, tvecs, strict=True):
    rmat, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    if np.all(np.isfinite(rmat)) and np.all(np.isfinite(t)):
      poses.append(gtsam.Pose3(gtsam.Rot3(rmat), gtsam.Point3(*t)))
  if not poses:
    return None
  # Bootstrap cheirality: tag +z must point toward the camera (R[:, 2] · t < 0).
  cheir = [
    p
    for p in poses
    if float(p.rotation().matrix()[:, 2] @ np.asarray(p.translation())) < 0.0
  ]
  if cheir:
    poses = cheir
  if len(poses) == 1:
    return poses[0]
  local = tag_corners_local(tag_size)
  points = np.concatenate(
    [(p.rotation().matrix() @ local.T).T + np.asarray(p.translation()) for p in poses]
  )
  pixels = mei_project(points, params).reshape(len(poses), 4, 2)
  rms = np.sqrt(np.mean(np.sum((pixels - corners_px) ** 2, axis=2), axis=1))
  return poses[int(np.argmin(rms))]


def lens_T_tag_per_kf(
  tags_per_kf: list[list[TagDetection]],
  tag_id: int,
  tag_size: float,
  params: MeiParams,
) -> dict[int, gtsam.Pose3]:
  """``{kf: lens_T_tag}`` from pre-detected tag lists."""
  out: dict[int, gtsam.Pose3] = {}
  for kf, tags in enumerate(tags_per_kf):
    for det in tags:
      if det.tag_id != tag_id:
        continue
      pose = estimate_lens_T_tag(det.corners, tag_size, params)
      if pose is not None:
        out[kf] = pose
        break
  return out


def detect_all_tags(frames: Sequence[np.ndarray]) -> list[list[TagDetection]]:
  """Per-keyframe AprilTag detections, one detector call at a time."""
  detector = _shared_detector()
  with _detect_slot():
    return [
      [
        TagDetection(
          tag_id=int(t.tag_id),
          center=np.asarray(t.center, dtype=np.float64),
          corners=np.asarray(t.corners, dtype=np.float64),
        )
        for t in detector.detect(
          frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )
      ]
      for frame in frames
    ]
