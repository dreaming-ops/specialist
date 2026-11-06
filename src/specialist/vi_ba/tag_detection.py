"""AprilTag detection and lens_T_tag from planar PnP through the MEI lens."""

import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
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
_DETECT_THREADS = 16
# Pixels of margin around a followed tag's last quad in a windowed search.
_FOLLOW_MARGIN = 160
# Below this decision margin a decode is texture, not a tag; real tags score 10 and up.
_MIN_DECISION_MARGIN = 5.0
_local = threading.local()

__all__ = [
  "TagDetection",
  "lens_T_tag_per_kf",
  "estimate_lens_T_tags",
  "detect_all_tags",
  "detect_tags_following",
  "detect_tags_following_split",
  "following_segments",
  "estimate_lens_T_tag",
  "tag_center_bearing",
  "tag_center_bearings",
]


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


def _thread_detector() -> Detector:
  """This thread's detector (~160 MB), built on first use."""
  got = getattr(_local, "detector", None)
  if got is None:
    got = _OrderedTeardownDetector(
      families=TAG_FAMILY, nthreads=max(1, min(_DETECT_THREADS, cpu_budget()) // 4)
    )
    _local.detector = got
  return got


@dataclass(frozen=True)
class TagDetection:
  """One AprilTag hit in one keyframe: id, center, corners (CCW image order)."""

  tag_id: int
  center: Float64Array  # (2,)
  corners: Float64Array  # (4, 2)


def tag_center_bearing(det: TagDetection, params: MeiParams) -> Float64Array:
  """Lens-frame bearing of a tag's center, zero where the lens cannot unproject it."""
  bearing: Float64Array = tag_center_bearings([det], params)[0]
  return bearing


def tag_center_bearings(
  dets: Sequence[TagDetection], params: MeiParams
) -> Float64Array:
  """``tag_center_bearing`` of every detection, as ``(N, 3)`` rows."""
  if not dets:
    return np.zeros((0, 3), dtype=np.float64)
  centers = np.stack([det.center for det in dets])
  bearings = mei_unproject(centers, params)
  bearings[~np.all(np.isfinite(bearings), axis=1)] = 0.0
  return bearings


def estimate_lens_T_tag(
  corners_px: Float64Array,
  tag_size: float,
  params: MeiParams,
) -> gtsam.Pose3 | None:
  """``lens_T_tag`` by planar IPPE PnP on the unprojected corners, else ``None``."""
  return estimate_lens_T_tags(corners_px[None], tag_size, params)[0]


def estimate_lens_T_tags(
  corners_px: Float64Array,
  tag_size: float,
  params: MeiParams,
) -> list[gtsam.Pose3 | None]:
  """``estimate_lens_T_tag`` over ``(N, 4, 2)`` sightings of one tag size."""
  corners = np.asarray(corners_px, dtype=np.float64)
  n = corners.shape[0]
  out: list[gtsam.Pose3 | None] = [None] * n
  if n == 0:
    return out
  bearings = mei_unproject(corners.reshape(-1, 2), params).reshape(n, 4, 3)
  usable = np.all(np.isfinite(bearings), axis=(1, 2)) & np.all(
    bearings[:, :, 2] > 1e-3, axis=1
  )
  obj = tag_corners_local(tag_size).reshape(-1, 1, 3)
  k = np.eye(3)
  dist = np.zeros(4)
  # Sightings still holding several candidates, ranked by reprojection RMS.
  contested: list[tuple[int, list[gtsam.Pose3]]] = []
  for i in np.flatnonzero(usable):
    normalized = (bearings[i, :, :2] / bearings[i, :, 2:3]).reshape(-1, 1, 2)
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
      continue
    # Cheirality: tag +z toward the lens.
    cheir = [
      p
      for p in poses
      if float(p.rotation().matrix()[:, 2] @ np.asarray(p.translation())) < 0.0
    ]
    if cheir:
      poses = cheir
    if len(poses) == 1:
      out[i] = poses[0]
    else:
      contested.append((int(i), poses))
  if contested:
    local = tag_corners_local(tag_size)
    points = np.concatenate(
      [
        (p.rotation().matrix() @ local.T).T + np.asarray(p.translation())
        for _, poses in contested
        for p in poses
      ]
    )
    pixels = mei_project(points, params).reshape(-1, 4, 2)
    begin = 0
    for index, poses in contested:
      rms = np.sqrt(
        np.mean(
          np.sum((pixels[begin : begin + len(poses)] - corners[index]) ** 2, axis=2),
          axis=1,
        )
      )
      out[index] = poses[int(np.argmin(rms))]
      begin += len(poses)
  return out


def lens_T_tag_per_kf(
  tags_per_kf: list[list[TagDetection]],
  tag_id: int,
  tag_size: float,
  params: MeiParams,
) -> dict[int, gtsam.Pose3]:
  """``{kf: lens_T_tag}``, each keyframe's first resolved sighting."""
  sightings = [
    (kf, det)
    for kf, tags in enumerate(tags_per_kf)
    for det in tags
    if det.tag_id == tag_id
  ]
  if not sightings:
    return {}
  poses = estimate_lens_T_tags(
    np.stack([det.corners for _, det in sightings]), tag_size, params
  )
  out: dict[int, gtsam.Pose3] = {}
  for (kf, _), pose in zip(sightings, poses, strict=True):
    if pose is not None and kf not in out:
      out[kf] = pose
  return out


def _detect(
  detector: Detector, frame: np.ndarray, offset: tuple[int, int] = (0, 0)
) -> list[TagDetection]:
  """Every tag in ``frame``, shifted by ``offset`` into its parent frame's pixels."""
  shift = np.asarray(offset, dtype=np.float64)
  return [
    TagDetection(
      tag_id=int(t.tag_id),
      center=np.asarray(t.center, dtype=np.float64) + shift,
      corners=np.asarray(t.corners, dtype=np.float64) + shift,
    )
    for t in detector.detect(np.ascontiguousarray(frame))
    if t.decision_margin >= _MIN_DECISION_MARGIN
  ]


def detect_all_tags(frames: Sequence[np.ndarray]) -> list[list[TagDetection]]:
  """Per-keyframe AprilTag detections, one detector call at a time."""
  detector = _thread_detector()
  return [_detect(detector, frame) for frame in frames]


def _follow_window(
  quads: Iterable[Float64Array], shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
  """The ``(x0, y0, x1, y1)`` frame window holding every quad plus the margin."""
  corners = np.concatenate(list(quads))
  x0, y0 = np.floor(corners.min(axis=0)).astype(int) - _FOLLOW_MARGIN
  x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + _FOLLOW_MARGIN
  return max(0, x0), max(0, y0), min(shape[1], x1), min(shape[0], y1)


def following_segments(n: int, full_every: int, segments: int) -> list[tuple[int, int]]:
  """``n`` frames as at most ``segments`` runs starting on ``full_every`` multiples."""
  if n <= 0:
    return []
  full_every = max(1, full_every)
  per = -(-n // max(1, segments))
  per = -(-per // full_every) * full_every
  return [(begin, min(begin + per, n)) for begin in range(0, n, per)]


def detect_tags_following_split(
  frames_of: Callable[[int, int], Sequence[np.ndarray]],
  n: int,
  full_every: int,
  follow: set[int],
  segments: int,
) -> list[list[TagDetection]]:
  """``detect_tags_following`` over ``n`` frames as ``segments`` parallel runs."""
  runs = following_segments(n, full_every, segments)
  if not runs:
    return []

  def walk(run: tuple[int, int]) -> list[list[TagDetection]]:
    begin, end = run
    return detect_tags_following(frames_of(begin, end), full_every, follow)

  with ThreadPoolExecutor(max_workers=len(runs)) as pool:
    parts = list(pool.map(walk, runs))
  return [dets for part in parts for dets in part]


def detect_tags_following(
  frames: Sequence[np.ndarray],
  full_every: int,
  follow: set[int],
) -> list[list[TagDetection]]:
  """Per-frame tags: whole frame every ``full_every``-th, else around ``follow``."""
  detector = _thread_detector()
  out: list[list[TagDetection]] = []
  last: dict[int, Float64Array] = {}
  for k, frame in enumerate(frames):
    if k % full_every == 0 or not last:
      dets = _detect(detector, frame)
    else:
      x0, y0, x1, y1 = _follow_window(last.values(), frame.shape)
      dets = _detect(detector, frame[y0:y1, x0:x1], (x0, y0))
    last = {d.tag_id: d.corners for d in dets if d.tag_id in follow}
    out.append(dets)
  return out
