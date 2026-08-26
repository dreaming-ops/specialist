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
from specialist_camera import MeiParams, mei_unproject
from specialist_dataset.fiducials import TAG_FAMILY, tag_corners_local

from specialist.vi_ba.features import Rect, cpu_budget

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
  "apply_static_mask_rects",
  "configure_detector",
  "detect_all_tags",
  "estimate_lens_T_tag",
  "non_anchor_tag_masks",
  "tag_center_bearing",
  "tag_half_angle",
]


def non_anchor_tag_masks(
  shape: tuple[int, int],
  tags_per_kf: list[list[TagDetection]],
  exclude_tag_ids: set[int],
  pad_px: int,
) -> list[list[Rect]]:
  """Per keyframe, the padded, clipped box of each tag not in ``exclude_tag_ids``."""
  h, w = shape
  blocked: list[list[Rect]] = []
  for tags in tags_per_kf:
    rects: list[Rect] = []
    for tag in tags:
      if int(tag.tag_id) in exclude_tag_ids:
        continue
      corners = np.asarray(tag.corners, dtype=np.float64)
      x0 = max(int(np.floor(corners[:, 0].min())) - pad_px, 0)
      y0 = max(int(np.floor(corners[:, 1].min())) - pad_px, 0)
      x1 = min(int(np.ceil(corners[:, 0].max())) + pad_px, w)
      y1 = min(int(np.ceil(corners[:, 1].max())) + pad_px, h)
      if x1 > x0 and y1 > y0:
        rects.append((x0, y0, x1, y1))
    blocked.append(rects)
  return blocked


def apply_static_mask_rects(
  blocked: list[list[Rect]], rects: list[Rect], shape: tuple[int, int]
) -> None:
  if not rects or not blocked:
    return
  h, w = shape
  for x1, y1, x2, y2 in rects:
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
      raise ValueError(
        f"mask_rear_rects entry {(x1, y1, x2, y2)} is empty after clamping to the "
        f"{w}x{h} frame — check the coordinate order (x1 y1 x2 y2) and bounds"
      )
    for frame_rects in blocked:
      frame_rects.append((x1c, y1c, x2c, y2c))


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


def tag_half_angle(det: TagDetection, params: MeiParams) -> float:
  """Off-axis angle (rad) of the mean corner bearing; ``inf`` if unprojection fails."""
  corners = np.asarray(det.corners, dtype=np.float64)
  bearing = mei_unproject(corners.mean(axis=0).reshape(1, 2), params)[0]
  norm = float(np.linalg.norm(bearing))
  if not np.isfinite(norm) or norm <= 1e-9:
    return float("inf")
  return float(np.arccos(np.clip(bearing[2] / norm, -1.0, 1.0)))


def estimate_lens_T_tag(
  corners_px: Float64Array,
  tag_size: float,
  params: MeiParams,
  prev_cam_T_tag: gtsam.Pose3 | None = None,
) -> gtsam.Pose3 | None:
  """Camera-to-tag pose by planar IPPE PnP; ``prev_cam_T_tag`` picks the candidate."""
  bearings = mei_unproject(corners_px.astype(np.float64), params)
  if not np.all(np.isfinite(bearings)) or np.any(bearings[:, 2] <= 1e-3):
    return None
  normalized = (bearings[:, :2] / bearings[:, 2:3]).astype(np.float64).reshape(-1, 1, 2)
  obj = tag_corners_local(tag_size).reshape(-1, 1, 3)
  k = np.eye(3)
  dist = np.zeros(4)
  n_sols, rvecs, tvecs, _ = cv2.solvePnPGeneric(
    obj, normalized, k, dist, flags=cv2.SOLVEPNP_IPPE
  )
  if n_sols == 0:
    return None
  poses: list[gtsam.Pose3] = []
  for rvec, tvec in zip(rvecs, tvecs, strict=True):
    rmat, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    poses.append(gtsam.Pose3(gtsam.Rot3(rmat), gtsam.Point3(*t)))
  # Bootstrap cheirality: tag +z must point toward the camera (R[:, 2] · t < 0).
  cheir = [
    p
    for p in poses
    if float(p.rotation().matrix()[:, 2] @ np.asarray(p.translation())) < 0.0
  ]
  if cheir:
    poses = cheir
  if prev_cam_T_tag is None or len(poses) == 1:
    return poses[0]
  return min(
    poses,
    key=lambda p: float(
      np.linalg.norm(gtsam.Rot3.Logmap(prev_cam_T_tag.rotation().between(p.rotation())))
    ),
  )


def lens_T_tag_per_kf(
  tags_per_kf: list[list[TagDetection]],
  tag_id: int,
  tag_size: float,
  params: MeiParams,
  kf_rotations: list[Float64Array],
) -> dict[int, gtsam.Pose3]:
  """``{kf: lens_T_tag}``, IPPE disambiguated by a gyro-propagated prior."""
  out: dict[int, gtsam.Pose3] = {}
  prev: gtsam.Pose3 | None = None
  prev_kf = 0
  for kf, tags in enumerate(tags_per_kf):
    prior = _rotate_prior(prev, kf_rotations, prev_kf, kf) if prev is not None else None
    for det in tags:
      if det.tag_id != tag_id:
        continue
      pose = estimate_lens_T_tag(det.corners, tag_size, params, prev_cam_T_tag=prior)
      if pose is not None:
        out[kf] = pose
        prev = pose
        prev_kf = kf
        break
  return out


def _rotate_prior(
  prev: gtsam.Pose3,
  kf_rotations: list[Float64Array],
  prev_kf: int,
  kf: int,
) -> gtsam.Pose3:
  """Predict a ``lens_T_tag`` prior at ``kf`` from the pose sighted at ``prev_kf``."""
  acc = np.eye(3)
  for j in range(prev_kf + 1, kf + 1):
    acc = kf_rotations[j] @ acc
  return gtsam.Pose3(gtsam.Rot3(acc), gtsam.Point3(0, 0, 0)).compose(prev)


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
