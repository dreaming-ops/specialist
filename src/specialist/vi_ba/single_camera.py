"""One camera's lenses and IMU prepared for the joint graph."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypedDict, TypeVar

import cv2
import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import MeiParams, mei_unproject
from specialist.loader.parse_insv import InsvData, load_insv
from specialist.record.offset_v3 import parse_offset_v3
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.configs import (
  AnchorCfg,
  FrontendCfg,
  GripperCfg,
  ImuCfg,
)
from specialist.vi_ba.features import (
  Frame,
  Rect,
  detect_orb,
  mask_plane,
  match_pair,
  parallel_map,
)
from specialist.vi_ba.finger_kinematics import FingerReproj, build_finger_reproj
from specialist.vi_ba.imu import (
  GRAVITY,
  GRAVITY_WORLD,
  ImuSamples,
  factory_imu_bias,
  find_stationary_windows,
  keyframe_delta_rotations,
  make_preintegration_params,
  preintegrate_between_keyframes,
)
from specialist.vi_ba.init import (
  Alignment,
  apply_alignment_to_poses,
  body_poses_from_lens_poses,
  incremental_sfm,
  interpolate_camera_poses,
  pick_init_pair,
  two_view_essential,
  visual_inertial_align,
)
from specialist.vi_ba.pipeline import (
  BandedKeyframes,
  LensFrontendConfig,
  filter_tracks_by_parallax,
  keyframe_timestamps,
  lens_setups,
  probe_capture,
  release_freed_memory,
  resolve_keyframe_every,
  run_lens_frontend,
  shared_keyframe_bound,
  triangulate_lens,
  unproject_tracks,
)
from specialist.vi_ba.tag_detection import (
  TagDetection,
  configure_detector,
  detect_all_tags,
  lens_T_tag_per_kf,
)
from specialist.vi_ba.urdf_joints import read_revolute_joints

Float64Array = npt.NDArray[np.float64]

_A = TypeVar("_A")
_B = TypeVar("_B")

# ORB loop-closure track recovery (``_revisit_tracks``).
_REVISIT_N_NEIGHBORS = 8  # spatially-closest non-adjacent revisit kfs to re-match
_REVISIT_MATCH_RATIO = 0.8  # Lowe ratio-test threshold
_REVISIT_RANSAC_THRESHOLD = 4.5  # homography RANSAC reprojection tolerance, px
_REVISIT_MIN_VIEWS = 3  # keyframes a union-found track must span to be kept
_REVISIT_MIN_INLIERS = 10  # geometrically-verified matches required per pair

# Visual-odometry pose-spike rejection (``_reject_vo_pose_outliers``).
_VO_REJECT_FACTOR = 8.0  # residual/median-step ratio marking a PnP teleport

# Keyframes read at once from a banded lens.
_PREP_BAND = 256


__all__ = [
  "PreparedCamera",
  "CameraArgs",
  "prepare_camera",
  "run_two_concurrently",
]


def _enough_registered(
  n_registered: int, n_keyframes: int, ratio: float = 0.25
) -> bool:
  """A seed needs ``ratio`` of the keyframes, and at least eight."""
  return n_registered >= max(8, int(n_keyframes * ratio))


def run_two_concurrently(
  task_a: Callable[[], _A], task_b: Callable[[], _B]
) -> tuple[_A, _B]:
  """Run two CPU-bound thunks on separate threads and return ``(a, b)``."""
  with ThreadPoolExecutor(max_workers=2) as ex:
    fut_a = ex.submit(task_a)
    fut_b = ex.submit(task_b)
    return fut_a.result(), fut_b.result()


@dataclass
class CameraArgs:
  """Per-tracker prep knobs composed from the nested configs in ``configs.py``."""

  insv_path: Path
  base_tag_id: int
  anchor: AnchorCfg
  start: float
  duration: float | None
  keyframe_every: int  # 0 = auto (round(fps/10)); see WindowCfg.keyframe_every
  urdf_path: Path | None  # gripper URDF for finger joint kinematics
  # 0 disables the corresponding URDF revolute joint angle recovery.
  left_finger_tag_id: int
  right_finger_tag_id: int
  frontend: FrontendCfg
  imu: ImuCfg
  gripper: GripperCfg
  # Set when sibling captures prep at once; it gates per-thread AprilTag detectors.
  parallel_siblings: bool = False


class ViGraphInputs(TypedDict):
  """One tracker's state and lens data, keyed to ``build_vi_graph``'s parameters."""

  body_poses: list[gtsam.Pose3]
  velocities: list[Float64Array]
  biases: list[gtsam.imuBias.ConstantBias]
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements]
  front: LensSetup
  front_tracks: dict[int, dict[int, Float64Array]]
  front_points: dict[int, Float64Array]
  rear: LensSetup
  rear_tracks: dict[int, dict[int, Float64Array]]
  rear_points: dict[int, Float64Array]


@dataclass
class PreparedCamera:
  """Everything one camera contributes to the joint graph + viz."""

  name: str
  body_poses: list[gtsam.Pose3]
  velocities: list[Float64Array]
  biases: list[gtsam.imuBias.ConstantBias]
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements]
  front_lens: LensSetup
  rear_lens: LensSetup
  front_tracks_tri: dict[int, dict[int, Float64Array]]
  rear_tracks_tri: dict[int, dict[int, Float64Array]]
  front_points: dict[int, Float64Array]
  rear_points: dict[int, Float64Array]
  decoded_width: int
  # ``{tag_id: {kf: {side: lens_T_tag}}}``.
  anchor_obs_by_tag: dict[int, dict[int, dict[str, gtsam.Pose3]]]
  anchor_locals_by_tag: dict[int, gtsam.Pose3]
  body_T_base_tag: gtsam.Pose3 | None
  keyframe_times: Float64Array
  front_keypoints: list[Float64Array]
  rear_keypoints: list[Float64Array]
  front_tags_per_kf: list[list[TagDetection]]
  rear_tags_per_kf: list[list[TagDetection]]
  front_rejected: dict[int, dict[int, Float64Array]]
  rear_rejected: dict[int, dict[int, Float64Array]]
  rear_static_mask_rects: list[tuple[int, int, int, int]]
  # URDF joint name -> kinematic inputs for the joint-angle reprojection factors.
  finger_reproj: dict[str, FingerReproj]
  left_finger_tag_id: int
  right_finger_tag_id: int
  # Body-frame, rescaled IMU samples, for IMU-rate propagation between keyframes.
  imu: ImuSamples | None = None
  # Per-joint raw ``(times, angles)`` at the finger rate; export pins them to the solve.
  finger_frame_angles: dict[str, tuple[Float64Array, Float64Array]] | None = None
  # Joint name -> solved angle about the joint z-axis per keyframe, NaN where unseen.
  joint_angles: dict[str, list[float]] = field(default_factory=dict)

  @property
  def n_keyframes(self) -> int:
    return len(self.body_poses)

  def vi_graph_inputs(self) -> ViGraphInputs:
    """Bundle this tracker's state + lens data for ``build_vi_graph``."""
    return ViGraphInputs(
      body_poses=self.body_poses,
      velocities=self.velocities,
      biases=self.biases,
      preintegrated=self.preintegrated,
      front=self.front_lens,
      front_tracks=self.front_tracks_tri,
      front_points=self.front_points,
      rear=self.rear_lens,
      rear_tracks=self.rear_tracks_tri,
      rear_points=self.rear_points,
    )


def _build_non_anchor_tag_masks(
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


def _apply_static_mask_rects(
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


# Anchor-detection lens side: name, lens, per-kf tag detections, gyro increments.
_LensSide = tuple[str, LensSetup, list[list[TagDetection]], list[Float64Array]]


def _detect_anchor_obs(
  tag_id: int,
  tag_size: float,
  lens_sides: tuple[_LensSide, ...],
) -> dict[int, dict[str, gtsam.Pose3]]:
  """``{kf: {side: lens_T_tag}}`` for one anchor id across both lens sides."""
  obs: dict[int, dict[str, gtsam.Pose3]] = {}
  for side, lens, tags, rotations in lens_sides:
    for kf, pose in lens_T_tag_per_kf(
      tags,
      tag_id,
      tag_size,
      lens.params,
      kf_rotations=rotations,
    ).items():
      obs.setdefault(kf, {})[side] = pose
  return obs


def _choose_prep_anchor(
  tag_ids: tuple[int, ...],
  obs_by_tag: dict[int, dict[int, dict[str, gtsam.Pose3]]],
) -> int:
  """The id sighted at most keyframes; a tie goes to the earlier ``tag_ids`` entry."""
  return max(tag_ids, key=lambda tid: len(obs_by_tag.get(tid, {})))


def _print_anchor_geometry(
  slot: str,
  tag_id: int,
  lens_sides: tuple[_LensSide, ...],
  obs: dict[int, dict[str, gtsam.Pose3]],
) -> None:
  """Print per-anchor edge length (px), bearing half-angle and PnP incidence angle."""
  edges: list[float] = []
  half_angles: list[float] = []
  for _side, lens, tags, _rotations in lens_sides:
    for dets in tags:
      for det in dets:
        if int(det.tag_id) != tag_id:
          continue
        corners = np.asarray(det.corners, dtype=np.float64)
        edge = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        edges.append(float(edge.min()))
        bearing = mei_unproject(corners.mean(axis=0).reshape(1, 2), lens.params)[0]
        norm = float(np.linalg.norm(bearing))
        if np.isfinite(norm) and norm > 1e-9:
          half_angles.append(
            float(np.degrees(np.arccos(np.clip(bearing[2] / norm, -1.0, 1.0))))
          )
  if not edges:
    return
  incidences: list[float] = []
  for sides in obs.values():
    for pose in sides.values():
      t = np.asarray(pose.translation(), dtype=np.float64)
      dist = float(np.linalg.norm(t))
      if dist < 1e-9:
        continue
      cos_incidence = -float(pose.rotation().matrix()[:, 2] @ t) / dist
      incidences.append(float(np.degrees(np.arccos(np.clip(cos_incidence, -1.0, 1.0)))))

  def med_p90(values: list[float]) -> str:
    if not values:
      return "-/-"
    return f"{float(np.median(values)):.0f}/{float(np.percentile(values, 90)):.0f}"

  print(
    f"  [{slot}] anchor {tag_id} geometry: {len(edges)} sightings, "
    f"edge px min/median {min(edges):.0f}/{float(np.median(edges)):.0f}, "
    f"half-angle median/p90 {med_p90(half_angles)} deg, "
    f"incidence median/p90 {med_p90(incidences)} deg"
  )


def _first_sighting_local(
  obs: dict[int, dict[str, gtsam.Pose3]],
  front_lens: LensSetup,
  rear_lens: LensSetup,
  body_poses: list[gtsam.Pose3],
) -> gtsam.Pose3:
  """An anchor's local-world pose from its first sighting, front lens preferred."""
  first_kf = min(obs)
  lens, lens_T_tag = _side_lens_and_pose(obs[first_kf], front_lens, rear_lens)
  return body_poses[first_kf].compose(lens.body_T_lens).compose(lens_T_tag)


def _translation(pose: gtsam.Pose3) -> Float64Array:
  return np.asarray(pose.translation(), dtype=np.float64)


def _mean_pose(poses: list[gtsam.Pose3]) -> gtsam.Pose3:
  """Mean translation and log-mean rotation about the first pose."""
  ref = poses[0]
  translations = np.stack([_translation(p) for p in poses])
  log_sum = np.zeros(3)
  for p in poses:
    log_sum += gtsam.Rot3.Logmap(ref.rotation().between(p.rotation()))
  delta = gtsam.Rot3.Expmap(log_sum / len(poses))
  return gtsam.Pose3(
    ref.rotation().compose(delta), gtsam.Point3(*translations.mean(axis=0))
  )


def _bearings_by_track(
  tracks: dict[int, dict[int, Float64Array]], params: MeiParams
) -> dict[int, dict[int, Float64Array]]:
  """``{tid: {kf: unit bearing}}`` for the visual-odometry front-end."""
  return {
    tid: dict(zip(kfs, bearings, strict=True))
    for tid, kfs, bearings in unproject_tracks(tracks, params, min_obs=1)
  }


def _subsample_obs(
  track: dict[int, Float64Array], max_obs: int
) -> dict[int, Float64Array]:
  """Keep at most ``max_obs`` observations, the endpoints plus an even middle spread."""
  kfs = sorted(track)
  if max_obs <= 0 or len(kfs) <= max_obs:
    return track
  idx = np.unique(np.linspace(0, len(kfs) - 1, max_obs).round().astype(int))
  return {kfs[i]: track[kfs[i]] for i in idx}


def _cap_landmarks(
  points: dict[int, Float64Array],
  tracks: dict[int, dict[int, Float64Array]],
  cap: int,
) -> dict[int, Float64Array]:
  """Keep at most ``cap`` landmarks, longest tracks first; ``cap <= 0`` keeps all."""
  if cap <= 0 or len(points) <= cap:
    return points
  keep = set(sorted(points, key=lambda t: -len(tracks.get(t, {})))[:cap])
  return {t: p for t, p in points.items() if t in keep}


def _revisit_tracks(
  orb: list[Frame],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  base_track_id: int,
  max_center_dist: float,
  min_kf_gap: int,
) -> dict[int, dict[int, Float64Array]]:
  """Recover tracks broken by fast lifts via ORB loop closure."""
  centers = np.asarray([_translation(p.compose(lens.body_T_lens)) for p in body_poses])
  n = len(orb)
  parent: dict[tuple[int, int], tuple[int, int]] = {}

  def find(x: tuple[int, int]) -> tuple[int, int]:
    parent.setdefault(x, x)
    root = x
    while parent[root] != root:
      root = parent[root]
    while parent[x] != root:
      parent[x], x = root, parent[x]
    return root

  pairs: list[tuple[int, int]] = []
  for i in range(n):
    if len(orb[i].keypoints) == 0:
      continue
    dist = np.linalg.norm(centers - centers[i], axis=1)
    cand = [
      int(j)
      for j in np.argsort(dist)
      if j > i and (j - i) > min_kf_gap and dist[j] <= max_center_dist
    ][:_REVISIT_N_NEIGHBORS]
    pairs.extend((i, j) for j in cand)

  def verify_pair(i: int, j: int) -> npt.NDArray[np.int64] | None:
    m = match_pair(orb[i], orb[j], _REVISIT_MATCH_RATIO)
    if len(m) < _REVISIT_MIN_INLIERS:
      return None
    pa = orb[i].keypoints[m[:, 0]]
    pb = orb[j].keypoints[m[:, 1]]
    # The cv2 RNG is thread-local, so a per-pair seed keeps RANSAC deterministic.
    cv2.setRNGSeed(i * n + j)
    _, mask = cv2.findHomography(pa, pb, cv2.RANSAC, _REVISIT_RANSAC_THRESHOLD)
    if mask is None:
      return None
    inl = mask.ravel().astype(bool)
    if int(inl.sum()) < _REVISIT_MIN_INLIERS:
      return None
    return np.asarray(m[inl], dtype=np.int64)

  verified = parallel_map(verify_pair, [i for i, _ in pairs], [j for _, j in pairs])
  for (i, j), matches in zip(pairs, verified, strict=True):
    if matches is None:
      continue
    for a_idx, b_idx in matches:
      ra = find((i, int(a_idx)))
      rb = find((j, int(b_idx)))
      if ra != rb:
        parent[ra] = rb

  comps: dict[tuple[int, int], dict[int, int]] = {}
  for node in list(parent):
    comps.setdefault(find(node), {})[node[0]] = node[1]
  tracks: dict[int, dict[int, Float64Array]] = {}
  tid = base_track_id + 1
  for by_kf in comps.values():
    if len(by_kf) < _REVISIT_MIN_VIEWS:
      continue
    # Store pixels; downstream code unprojects through the MEI model itself.
    tracks[tid] = {
      kf: np.asarray(orb[kf].keypoints[kp], dtype=np.float64)
      for kf, kp in by_kf.items()
    }
    tid += 1
  return tracks


def _add_revisit_landmarks(
  front_points: dict[int, Float64Array],
  front_kept: dict[int, dict[int, Float64Array]],
  revisits: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  *,
  min_track_length: int,
  max_landmark_depth: float,
  slot: str,
  anchor_kfs: set[int] | None = None,
) -> None:
  """Add ``revisits`` to ``front_kept`` and their landmarks to ``front_points``."""
  if anchor_kfs:
    tri_tracks = {
      t: {k: v for k, v in obs.items() if k in anchor_kfs}
      for t, obs in revisits.items()
    }
  else:
    tri_tracks = revisits
  revisit_pts = triangulate_lens(
    tri_tracks,
    body_poses,
    lens.body_T_lens,
    lens.params,
    min_track_length,
    max_landmark_depth=max_landmark_depth,
  )
  front_points.update(revisit_pts)
  front_kept.update(revisits)
  print(
    f"  [{slot}] loop closure: +{len(revisit_pts)} landmarks "
    f"from {len(revisits)} revisit tracks"
  )


def _reject_vo_pose_outliers(
  lens_poses: dict[int, gtsam.Pose3],
) -> tuple[dict[int, gtsam.Pose3], list[int]]:
  """Drop incremental-SfM keyframes whose camera center is a local spike."""
  kept = dict(lens_poses)
  removed: list[int] = []
  for _ in range(len(lens_poses)):
    kfs = sorted(kept)
    if len(kfs) < 5:
      break
    centers = {kf: _translation(kept[kf]) for kf in kfs}
    steps = [
      float(np.linalg.norm(centers[kfs[i + 1]] - centers[kfs[i]]))
      / (kfs[i + 1] - kfs[i])
      for i in range(len(kfs) - 1)
    ]
    median_step = float(np.median(steps))
    if median_step < 1e-12:
      break
    worst_kf: int | None = None
    worst_residual = 0.0
    for i in range(1, len(kfs) - 1):
      prev_kf, kf, next_kf = kfs[i - 1], kfs[i], kfs[i + 1]
      frac = (kf - prev_kf) / (next_kf - prev_kf)
      interpolated = (1.0 - frac) * centers[prev_kf] + frac * centers[next_kf]
      residual = float(np.linalg.norm(centers[kf] - interpolated))
      if residual > worst_residual:
        worst_residual, worst_kf = residual, kf
    if worst_kf is not None and worst_residual > _VO_REJECT_FACTOR * median_step:
      del kept[worst_kf]
      removed.append(worst_kf)
    else:
      break
  return kept, sorted(removed)


def _fill_lens_poses(
  lens_poses: dict[int, gtsam.Pose3], n_keyframes: int
) -> list[gtsam.Pose3]:
  """Every keyframe's camera pose; gaps interpolate on SE(3) and hold at the ends."""
  reg = sorted(lens_poses)
  out: list[gtsam.Pose3] = []
  for k in range(n_keyframes):
    if k in lens_poses:
      out.append(lens_poses[k])
      continue
    lo = max((r for r in reg if r < k), default=None)
    hi = min((r for r in reg if r > k), default=None)
    if lo is None:
      out.append(lens_poses[hi])  # type: ignore[index]
    elif hi is None:
      out.append(lens_poses[lo])
    else:
      pa, pb = lens_poses[lo], lens_poses[hi]
      f = (k - lo) / (hi - lo)
      out.append(pa.compose(gtsam.Pose3.Expmap(f * gtsam.Pose3.Logmap(pa.between(pb)))))
  return out


def _anchor_pose_seed(
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]],
  body_T_front_lens: gtsam.Pose3,
  n_keyframes: int,
) -> list[gtsam.Pose3] | None:
  """Anchor-PnP body poses, gaps filled; ``None`` when too few keyframes see it."""
  lens_poses = {
    kf: sides["front"].inverse()
    for kf, sides in anchor_obs_per_kf.items()
    if "front" in sides
  }
  if not _enough_registered(len(lens_poses), n_keyframes):
    return None
  return body_poses_from_lens_poses(
    _fill_lens_poses(lens_poses, n_keyframes), body_T_front_lens
  )


def _anchor_metric_scale(
  body_poses_unscaled: list[gtsam.Pose3],
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]],
  body_T_front_lens: gtsam.Pose3,
  rear_extrinsic: gtsam.Pose3,
) -> float | None:
  """Metric scale: anchor-implied over visual body-position spread, or ``None``."""
  vo: list[Float64Array] = []
  anc: list[Float64Array] = []
  for kf, sides in anchor_obs_per_kf.items():
    if "front" in sides:
      ext, lens_T_tag = body_T_front_lens, sides["front"]
    elif "rear" in sides:
      ext, lens_T_tag = rear_extrinsic, sides["rear"]
    else:
      continue
    body_in_tag = ext.compose(lens_T_tag).inverse()
    anc.append(_translation(body_in_tag))
    vo.append(_translation(body_poses_unscaled[kf]))
  if len(vo) < 3:
    return None
  vo_arr = np.asarray(vo, dtype=np.float64)
  anc_arr = np.asarray(anc, dtype=np.float64)
  vo_spread = float(np.sqrt(((vo_arr - vo_arr.mean(axis=0)) ** 2).sum()))
  anc_spread = float(np.sqrt(((anc_arr - anc_arr.mean(axis=0)) ** 2).sum()))
  if vo_spread < 1e-9:
    return None
  return anc_spread / vo_spread


def _side_lens_and_pose(
  sides: dict[str, gtsam.Pose3], front_lens: LensSetup, rear_lens: LensSetup
) -> tuple[LensSetup, gtsam.Pose3]:
  """The anchor observation's lens and ``lens_T_tag``, front lens preferred."""
  if "front" in sides:
    return front_lens, sides["front"]
  return rear_lens, sides["rear"]


def _body_imu_samples(data: InsvData, body_R_imu: Float64Array) -> ImuSamples:
  """Raw telemetry IMU rotated into the body frame."""
  return ImuSamples(
    t=np.asarray(data.imu_timestamps, dtype=np.float64),
    gyro=data.gyro @ body_R_imu.T,
    accel=np.asarray(data.accel, dtype=np.float64) @ body_R_imu.T,
  )


def _imu_and_initial_gravity(
  args: CameraArgs,
  data: InsvData,
  imu: ImuSamples,
  keyframe_times: Float64Array,
  body_poses_prealign: list[gtsam.Pose3],
) -> tuple[ImuSamples, gtsam.imuBias.ConstantBias, Float64Array | None]:
  """IMU rescaled to |g| at rest, body-frame factory bias, world gravity or ``None``."""
  body_R_imu = np.asarray(args.imu.body_R_imu, dtype=np.float64)
  raw_factory_bias = factory_imu_bias(data.gyro_calib_numbers)
  bias0 = gtsam.imuBias.ConstantBias(
    body_R_imu @ np.asarray(raw_factory_bias.accelerometer(), dtype=np.float64),
    body_R_imu @ np.asarray(raw_factory_bias.gyroscope(), dtype=np.float64),
  )
  windows = find_stationary_windows(imu, bias0)
  clip_end = imu.t[-1] if args.duration is None else args.start + args.duration
  windows = [(s, e) for s, e in windows if args.start <= imu.t[s] <= clip_end]
  gravity_world_initial: Float64Array | None = None
  if windows:
    first_start, first_end = windows[0]
    stationary_mean = (imu.accel[first_start:first_end] - bias0.accelerometer()).mean(
      axis=0
    )
    stationary_mag = float(np.linalg.norm(stationary_mean))
    if 8.0 < stationary_mag < 11.0 and abs(stationary_mag - GRAVITY) > 0.05:
      accel_scale = GRAVITY / stationary_mag
      imu = ImuSamples(
        t=imu.t, gyro=imu.gyro, accel=(imu.accel * accel_scale).astype(np.float64)
      )
      stationary_mean = stationary_mean * accel_scale
    gravity_body = -stationary_mean / float(np.linalg.norm(stationary_mean)) * GRAVITY
    t_stationary_mid = 0.5 * (imu.t[first_start] + imu.t[first_end - 1])
    kf_for_g = int(np.argmin(np.abs(keyframe_times - t_stationary_mid)))
    gravity_world_initial = (
      body_poses_prealign[kf_for_g].rotation().matrix() @ gravity_body
    )
  return imu, bias0, gravity_world_initial


def _recover_gripper_tags(
  args: CameraArgs,
  rear_tags_per_kf: list[list[TagDetection]],
  rear_lens: LensSetup,
  n_keyframes: int,
  kf_rotations: list[Float64Array],
  rear_tags_per_frame: list[list[TagDetection]] | None = None,
  frame_times: Float64Array | None = None,
) -> tuple[
  gtsam.Pose3 | None,
  dict[str, FingerReproj],
  dict[str, tuple[Float64Array, Float64Array]],
]:
  """Base-tag body pose and, per finger, reprojection inputs and frame angles."""
  base_tag_obs = lens_T_tag_per_kf(
    rear_tags_per_kf,
    tag_id=args.base_tag_id,
    tag_size=args.gripper.tag_size,
    params=rear_lens.params,
    kf_rotations=kf_rotations,
  )
  body_T_base_tag: gtsam.Pose3 | None = None
  if base_tag_obs:
    body_T_base_tag = _mean_pose(
      [rear_lens.body_T_lens.compose(p) for p in base_tag_obs.values()]
    )
  if args.base_tag_id > 0 and body_T_base_tag is None:
    raise ValueError(
      f"base tag {args.base_tag_id} was configured but never detected in the rear "
      f"lens across {n_keyframes} keyframes — its base-tag, finger, and cross-tracker "
      f"factors would all be silently disabled. Wrong id, or the tag out of the "
      f"rear-lens field of view?"
    )

  finger_joint_tags = (
    ("left_finger", args.left_finger_tag_id),
    ("right_finger", args.right_finger_tag_id),
  )
  finger_reproj: dict[str, FingerReproj] = {}
  if args.urdf_path is not None:
    # URDF joints sit in the base frame; body_T_base_tag carries them to the body.
    joints = read_revolute_joints(args.urdf_path)
    for joint_name, finger_id in finger_joint_tags:
      if finger_id <= 0:
        continue
      joint = joints.get(joint_name)
      if joint is None or joint.joint_T_tag0 is None:
        raise ValueError(
          f"finger tag {finger_id} was configured for '{joint_name}', but "
          f"{args.urdf_path} has no revolute joint '{joint_name}' carrying an "
          f"AprilTag frame (revolute joints present: {sorted(joints)}). Wrong URDF?"
        )
      if body_T_base_tag is None:
        continue
      reproj = build_finger_reproj(
        joint,
        rear_tags_per_kf[:n_keyframes],
        rear_lens,
        finger_id,
        body_T_base_tag,
        args.gripper.tag_size,
      )
      if reproj is None:
        print(
          f"  [warn] finger tag {finger_id} ({joint_name}) never detected in the "
          f"rear keyframes; its angle stays unsolved"
        )
      else:
        finger_reproj[joint_name] = reproj
  frame_angles: dict[str, tuple[Float64Array, Float64Array]] = {}
  if (
    rear_tags_per_frame is not None
    and frame_times is not None
    and args.urdf_path is not None
    and body_T_base_tag is not None
  ):
    joints = read_revolute_joints(args.urdf_path)
    for joint_name, finger_id in finger_joint_tags:
      joint = joints.get(joint_name)
      if finger_id <= 0 or joint is None or joint.joint_T_tag0 is None:
        continue
      framed = build_finger_reproj(
        joint,
        rear_tags_per_frame,
        rear_lens,
        finger_id,
        body_T_base_tag,
        args.gripper.tag_size,
      )
      if framed is not None:
        frame_angles[joint_name] = (
          frame_times[list(framed.observations)],
          np.array([o.theta_seed for o in framed.observations.values()]),
        )
  return body_T_base_tag, finger_reproj, frame_angles


@dataclass
class _LensChannel:
  """One lens's prep state: grays and tag detections in, tracks and landmarks out."""

  setup: LensSetup
  gray: Sequence[np.ndarray]
  tags: list[list[TagDetection]]
  masks: list[list[Rect]] = field(default_factory=list)
  orb: list[Frame] = field(default_factory=list)
  feature_frames: list[Frame] = field(default_factory=list)
  tracks: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  kept: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  rejected: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  landmarks_prealign: dict[int, Float64Array] = field(default_factory=dict)
  points: dict[int, Float64Array] = field(default_factory=dict)
  tracks_tri: dict[int, dict[int, Float64Array]] = field(default_factory=dict)


def prepare_camera(slot: str, args: CameraArgs) -> PreparedCamera:
  """Every per-camera stage; body poses land in the camera's gravity-aligned world."""
  configure_detector(args.frontend.parallel_tag_detect and args.parallel_siblings)
  data = load_insv(args.insv_path)
  offset_v3 = parse_offset_v3(data.offset_v3)
  rear_path, fps, decoded_width = probe_capture(args.insv_path)
  front_lens, rear_lens = lens_setups(offset_v3, decoded_width)

  keyframe_every = resolve_keyframe_every(args.keyframe_every, fps)
  # One lens at a time: a lens's keyframes are the largest thing prep holds.
  bound = shared_keyframe_bound(args.insv_path, rear_path)

  def lens_frames(path: Path) -> Sequence[np.ndarray]:
    """This lens's keyframes, decoded a band at a time."""
    return BandedKeyframes(path, bound, fps, args.start, args.duration, keyframe_every)

  def per_band(
    fn: Callable[[list[np.ndarray]], list[_A]], frames: Sequence[np.ndarray]
  ) -> list[_A]:
    """Map over the frames a band at a time so a banded sequence reads in order."""
    out: list[_A] = []
    for begin in range(0, len(frames), _PREP_BAND):
      end = min(begin + _PREP_BAND, len(frames))
      out.extend(fn([frames[i] for i in range(begin, end)]))
    return out

  # Front tags and loop-closure ORB share one banded walk, so the lens decodes once.
  orb_exclude = set(args.anchor.tag_ids)

  def front_band(
    band: list[np.ndarray],
  ) -> list[tuple[list[TagDetection], Frame | None]]:
    tags = detect_all_tags(band)
    if not args.frontend.revisit:
      return [(t, None) for t in tags]
    rects = (
      _build_non_anchor_tag_masks(
        band[0].shape[:2], tags, orb_exclude, args.frontend.mask_tag_pad
      )
      if args.frontend.mask_tags
      else [[] for _ in band]
    )
    orb = parallel_map(
      lambda f, m: detect_orb(
        f, args.frontend.orb_n_features, 20, mask_plane(f.shape[:2], m)
      ),
      band,
      rects,
    )
    return list(zip(tags, orb, strict=True))

  front_fused = per_band(front_band, lens_frames(args.insv_path))
  front_tags_per_kf = [tags for tags, _ in front_fused]
  front_orb = [orb for _, orb in front_fused]
  release_freed_memory()
  # With fingers configured, the rear walk detects tags at the finger stride.
  wants_fingers = args.urdf_path is not None and (
    args.left_finger_tag_id > 0 or args.right_finger_tag_id > 0
  )
  rear_tags_per_frame: list[list[TagDetection]] | None = None
  rear_frame_times: Float64Array | None = None
  if wants_fingers:
    # Finger angles target ~30 Hz; the stride must divide the keyframe interval.
    finger_stride = max(
      (
        d
        for d in range(1, keyframe_every + 1)
        if keyframe_every % d == 0 and d <= fps / 30.0
      ),
      default=1,
    )
    rear_all = BandedKeyframes(
      rear_path, bound, fps, args.start, args.duration, finger_stride
    )
    rear_tags_per_frame = per_band(detect_all_tags, rear_all)
    rear_frame_times = (
      args.start + np.arange(len(rear_tags_per_frame)) * finger_stride / fps
    )
    rear_tags_per_kf = rear_tags_per_frame[:: keyframe_every // finger_stride]
  else:
    rear_tags_per_kf = per_band(detect_all_tags, lens_frames(rear_path))
  n_keyframes = min(len(front_tags_per_kf), len(rear_tags_per_kf))
  release_freed_memory()
  if n_keyframes < 3:
    raise RuntimeError(f"only {n_keyframes} keyframes decoded — window too short")
  front_tags_per_kf = front_tags_per_kf[:n_keyframes]
  rear_tags_per_kf = rear_tags_per_kf[:n_keyframes]
  keyframe_times = keyframe_timestamps(args.start, fps, keyframe_every, n_keyframes)

  body_R_imu = np.asarray(args.imu.body_R_imu, dtype=np.float64)
  imu = _body_imu_samples(data, body_R_imu)
  body_deltas = keyframe_delta_rotations(imu, keyframe_times)

  def lens_kf_rotations(lens: LensSetup) -> list[Float64Array]:
    r_bl = lens.body_T_lens.rotation().matrix()
    return [r_bl.T @ d.T @ r_bl for d in body_deltas]

  front_kf_rotations = lens_kf_rotations(front_lens)
  rear_kf_rotations = lens_kf_rotations(rear_lens)

  lens_sides = (
    ("front", front_lens, front_tags_per_kf, front_kf_rotations),
    ("rear", rear_lens, rear_tags_per_kf, rear_kf_rotations),
  )
  obs_by_tag = {
    tag_id: _detect_anchor_obs(tag_id, args.anchor.tag_size, lens_sides)
    for tag_id in args.anchor.tag_ids
  }
  prep_anchor_tag_id = _choose_prep_anchor(args.anchor.tag_ids, obs_by_tag)
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]] = obs_by_tag.get(
    prep_anchor_tag_id, {}
  )
  if prep_anchor_tag_id != args.anchor.tag_ids[0]:
    n_ref = len(obs_by_tag.get(args.anchor.tag_ids[0], {}))
    print(
      f"  [{slot}] prep reference anchor: tag {prep_anchor_tag_id} "
      f"({len(anchor_obs_per_kf)} sighted keyframes vs {n_ref} for "
      f"tag {args.anchor.tag_ids[0]})"
    )
  for tag_id, obs in obs_by_tag.items():
    _print_anchor_geometry(slot, tag_id, lens_sides, obs)

  front = _LensChannel(setup=front_lens, gray=[], tags=front_tags_per_kf)
  rear = _LensChannel(setup=rear_lens, gray=[], tags=rear_tags_per_kf)
  anchor_tag_ids = set(obs_by_tag)
  static_rects = list(args.frontend.mask_rear_rects)

  cfg = LensFrontendConfig(
    n_features=args.frontend.orb_n_features,
    ransac_threshold=args.frontend.ransac_threshold,
    max_half_angle=args.frontend.max_half_angle,
    gpu_flow=args.frontend.gpu_flow,
    flow_scale=args.frontend.flow_scale,
  )

  def run_lens(ch: _LensChannel, path: Path) -> None:
    """Decode this lens, take what downstream needs from its pixels, then drop them."""
    ch.gray = lens_frames(path)
    shape = ch.gray[0].shape[:2]
    ch.masks = (
      _build_non_anchor_tag_masks(
        shape, ch.tags, anchor_tag_ids, args.frontend.mask_tag_pad
      )
      if args.frontend.mask_tags
      else [[] for _ in range(len(ch.tags))]
    )
    if ch is rear:
      _apply_static_mask_rects(ch.masks, static_rects, shape)
    ch.feature_frames, ch.tracks = run_lens_frontend(
      ch.gray, ch.setup.params, cfg, masks=ch.masks
    )
    if ch is front and args.frontend.revisit:
      ch.orb = [f for f in front_orb[:n_keyframes] if f is not None]
    ch.gray = []
    ch.masks = []
    release_freed_memory()

  run_lens(front, args.insv_path)
  if args.frontend.rear_landmarks:
    run_lens(rear, rear_path)
  for ch in (front, rear):
    ch.kept = {
      t: o for t, o in ch.tracks.items() if len(o) >= args.frontend.min_track_length
    }
    if args.frontend.min_parallax > 0.0:
      ch.kept, ch.rejected = filter_tracks_by_parallax(
        ch.kept, ch.setup.params, args.frontend.min_parallax
      )

  f_last, f_shared = pick_init_pair(
    front.kept, n_keyframes, args.frontend.init_min_shared_tracks
  )
  r_last, r_shared = pick_init_pair(
    rear.kept, n_keyframes, args.frontend.init_min_shared_tracks
  )
  init_tracks, init_lens, init_last, init_shared = (
    (front.kept, front_lens, f_last, f_shared)
    if f_shared >= r_shared
    else (rear.kept, rear_lens, r_last, r_shared)
  )
  if init_shared < 8:
    raise RuntimeError(f"{slot}: no lens has ≥8 shared tracks (best={init_shared})")

  # The init pair is anchored at keyframe 0.
  shared = [obs for obs in init_tracks.values() if 0 in obs and init_last in obs]
  pix_a = np.asarray([obs[0] for obs in shared], dtype=np.float64)
  pix_b = np.asarray([obs[init_last] for obs in shared], dtype=np.float64)
  bears_a = mei_unproject(pix_a, init_lens.params)
  bears_b = mei_unproject(pix_b, init_lens.params)
  r_cw_last, t_cw_last, _ = two_view_essential(bears_a, bears_b)
  lens_poses = interpolate_camera_poses(r_cw_last, t_cw_last, n_keyframes)
  body_poses_prealign = body_poses_from_lens_poses(lens_poses, init_lens.body_T_lens)

  # Visual odometry shapes the trajectory, else the anchor-PnP seed or straight line.
  anchor_seed_used = False
  sfm_cams, _ = incremental_sfm(
    _bearings_by_track(front.kept, front_lens.params),
    min_shared=args.frontend.init_min_shared_tracks,
  )
  sfm_cams, vo_outliers = _reject_vo_pose_outliers(sfm_cams)
  if vo_outliers:
    print(f"  [{slot}] visual-odometry rejected {len(vo_outliers)} rotation-spike kfs")
  if _enough_registered(
    len(sfm_cams), n_keyframes, args.frontend.vo_min_registered_ratio
  ):
    body_poses_prealign = body_poses_from_lens_poses(
      _fill_lens_poses(sfm_cams, n_keyframes), front_lens.body_T_lens
    )
    print(
      f"  [{slot}] visual-odometry init: registered "
      f"{len(sfm_cams)}/{n_keyframes} keyframes"
    )
  else:
    anchor_seed = _anchor_pose_seed(
      anchor_obs_per_kf, front_lens.body_T_lens, n_keyframes
    )
    if anchor_seed is not None:
      body_poses_prealign = anchor_seed
      anchor_seed_used = True
      n_anchor_kf = sum("front" in s for s in anchor_obs_per_kf.values())
      print(
        f"  [{slot}] visual-odometry starved ({len(sfm_cams)}/{n_keyframes}); "
        f"anchor-pose init from {n_anchor_kf} anchor keyframes"
      )
    else:
      print(
        f"  [{slot}] visual-odometry init: only {len(sfm_cams)}/{n_keyframes} "
        f"keyframes registered — using straight-line init"
      )

  imu, bias0, gravity_world_initial = _imu_and_initial_gravity(
    args, data, imu, keyframe_times, body_poses_prealign
  )

  for ch in (front, rear):
    ch.landmarks_prealign = triangulate_lens(
      ch.kept,
      body_poses_prealign,
      ch.setup.body_T_lens,
      ch.setup.params,
      args.frontend.min_track_length,
      min_landmark_depth=0.01,
      max_landmark_depth=1000.0,
    )

  # Scale: 1.0 for an anchor seed, else from a visible anchor, else floated on the IMU.
  anchor_scale = _anchor_metric_scale(
    body_poses_prealign,
    anchor_obs_per_kf,
    front_lens.body_T_lens,
    rear_lens.body_T_lens,
  )
  fixed_scale = 1.0 if anchor_seed_used else anchor_scale
  # Gravity never enters the deltas, so one PIM serves alignment and export.
  pim = preintegrate_between_keyframes(
    make_preintegration_params(GRAVITY_WORLD), imu, keyframe_times, bias0
  )

  def _align(poses: list[gtsam.Pose3]) -> Alignment:
    return visual_inertial_align(
      poses,
      pim,
      fixed_gravity=gravity_world_initial,
      fixed_scale=fixed_scale,
    )

  alignment = _align(body_poses_prealign)
  if alignment.scale < 0.0:
    origin = _translation(body_poses_prealign[0])
    body_poses_prealign = [
      gtsam.Pose3(
        p.rotation(),
        gtsam.Point3(*(2.0 * origin - _translation(p))),
      )
      for p in body_poses_prealign
    ]
    alignment = _align(body_poses_prealign)
  # Prefer the IMU stationary-window gravity to the LSQ fit, which can tilt the world.
  if gravity_world_initial is not None:
    alignment = replace(alignment, gravity_world=gravity_world_initial)
  print(f"  [{slot}] VI scale={alignment.scale:.4f}")
  body_poses, world_R_prealign = apply_alignment_to_poses(
    body_poses_prealign, alignment
  )
  world_rot_mat = world_R_prealign.matrix()
  velocities = [world_rot_mat @ v for v in alignment.velocities]

  use_anchor = bool(anchor_obs_per_kf)
  # Each sighted anchor's pose in this camera's local world, for the shared-world BFS.
  anchor_locals_by_tag: dict[int, gtsam.Pose3] = {
    tag_id: _first_sighting_local(obs, front_lens, rear_lens, body_poses)
    for tag_id, obs in obs_by_tag.items()
    if obs
  }

  if use_anchor:
    revisits: dict[int, dict[int, Float64Array]] = {}
    if args.frontend.revisit:
      revisits = _revisit_tracks(
        front.orb,
        body_poses,
        front_lens,
        max(front.kept, default=0),
        args.frontend.loop_closure_max_dist,
        args.frontend.loop_closure_min_kf_gap,
      )
    anchor_kfs = set(anchor_obs_per_kf.keys())
    cap = args.frontend.max_landmarks_per_lens
    for ch in (front, rear):
      ch.points = triangulate_lens(
        {t: ch.kept[t] for t in ch.kept if t in ch.landmarks_prealign},
        body_poses,
        ch.setup.body_T_lens,
        ch.setup.params,
        args.frontend.min_track_length,
        max_landmark_depth=args.frontend.max_landmark_depth,
      )
      ch.points = _cap_landmarks(ch.points, ch.kept, cap)
    if revisits:
      # Merge after the cap so the per-lens budget never culls the revisit landmarks.
      _add_revisit_landmarks(
        front.points,
        front.kept,
        revisits,
        body_poses,
        front_lens,
        min_track_length=args.frontend.min_track_length,
        max_landmark_depth=args.frontend.max_landmark_depth,
        slot=slot,
        anchor_kfs=anchor_kfs if args.frontend.anchored_revisit else None,
      )
  else:
    align_mat = alignment.scale * world_R_prealign.matrix()
    for ch in (front, rear):
      ch.points = {
        tid: align_mat @ np.asarray(ch.landmarks_prealign[tid], dtype=np.float64)
        for tid in ch.landmarks_prealign
      }

  n_obs = args.frontend.max_obs_per_track
  for ch in (front, rear):
    ch.tracks_tri = {
      t: _subsample_obs(ch.kept[t], n_obs) for t in ch.points if t in ch.kept
    }

  v_max = float(max(np.linalg.norm(v) for v in velocities))
  print(f"  [{slot}] VIO LSQ |v|max={v_max:.2f} m/s")
  biases = [bias0 for _ in range(n_keyframes)]

  body_T_base_tag, finger_reproj, finger_frame_angles = _recover_gripper_tags(
    args,
    rear_tags_per_kf,
    rear_lens,
    n_keyframes,
    rear_kf_rotations,
    rear_tags_per_frame,
    rear_frame_times,
  )

  return PreparedCamera(
    name=slot,
    body_poses=body_poses,
    velocities=velocities,
    biases=biases,
    preintegrated=pim,
    front_lens=front_lens,
    rear_lens=rear_lens,
    front_tracks_tri=front.tracks_tri,
    rear_tracks_tri=rear.tracks_tri,
    front_points=front.points,
    rear_points=rear.points,
    decoded_width=decoded_width,
    anchor_obs_by_tag=obs_by_tag,
    anchor_locals_by_tag=anchor_locals_by_tag,
    body_T_base_tag=body_T_base_tag,
    keyframe_times=keyframe_times,
    front_keypoints=[f.keypoints for f in front.feature_frames][:n_keyframes],
    rear_keypoints=[f.keypoints for f in rear.feature_frames][:n_keyframes],
    front_tags_per_kf=front_tags_per_kf,
    rear_tags_per_kf=rear_tags_per_kf,
    front_rejected=front.rejected,
    rear_rejected=rear.rejected,
    rear_static_mask_rects=static_rects,
    finger_reproj=finger_reproj,
    left_finger_tag_id=args.left_finger_tag_id,
    right_finger_tag_id=args.right_finger_tag_id,
    imu=imu,
    finger_frame_angles=finger_frame_angles or None,
  )
