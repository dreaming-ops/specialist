"""AprilTag detection and lens_T_tag from planar PnP through the MEI lens."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import gtsam
import numpy as np
import numpy.typing as npt
from pupil_apriltags import Detector

from specialist.loader.mei_model import MeiParams, mei_unproject

Float64Array = npt.NDArray[np.float64]

_TAG_FAMILY = "tagStandard41h12"

__all__ = [
  "TagDetection",
  "lens_T_tag_per_kf",
  "detect_all_tags",
  "estimate_lens_T_tag",
  "rotate_prior",
]


@dataclass(frozen=True)
class TagDetection:
  """One AprilTag hit in one keyframe: id, center, corners (CCW image order)."""

  tag_id: int
  center: Float64Array  # (2,)
  corners: Float64Array  # (4, 2)


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
  norm = (bearings[:, :2] / bearings[:, 2:3]).astype(np.float64).reshape(-1, 1, 2)
  half = float(tag_size / 2.0)
  obj = np.array(
    [
      [-half, -half, 0.0],
      [+half, -half, 0.0],
      [+half, +half, 0.0],
      [-half, +half, 0.0],
    ],
    dtype=np.float64,
  ).reshape(-1, 1, 3)
  k = np.eye(3, dtype=np.float64)
  dist = np.zeros(4, dtype=np.float64)
  n_sols, rvecs, tvecs, _ = cv2.solvePnPGeneric(
    obj, norm, k, dist, flags=cv2.SOLVEPNP_IPPE
  )
  if n_sols == 0:
    return None
  poses: list[gtsam.Pose3] = []
  for rvec, tvec in zip(rvecs, tvecs, strict=True):
    rmat, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    poses.append(
      gtsam.Pose3(
        gtsam.Rot3(rmat),
        gtsam.Point3(float(t[0]), float(t[1]), float(t[2])),
      )
    )
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
    for det in tags:
      if det.tag_id != tag_id:
        continue
      prior = (
        rotate_prior(prev, kf_rotations, prev_kf, kf) if prev is not None else None
      )
      pose = estimate_lens_T_tag(det.corners, tag_size, params, prev_cam_T_tag=prior)
      if pose is not None:
        out[kf] = pose
        prev = pose
        prev_kf = kf
        break
  return out


def rotate_prior(
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
  """Per-keyframe AprilTag list: every detected tag id + center + corners."""
  detector = Detector(families=_TAG_FAMILY)
  out: list[list[TagDetection]] = []
  for frame in frames:
    img = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.append(
      [
        TagDetection(
          tag_id=int(t.tag_id),
          center=np.asarray(t.center, dtype=np.float64),
          corners=np.asarray(t.corners, dtype=np.float64),
        )
        for t in detector.detect(img)
      ]
    )
  return out
