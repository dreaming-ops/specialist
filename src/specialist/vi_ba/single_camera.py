"""One camera's lenses and IMU prepared for the joint graph."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeVar

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
from specialist.vi_ba.features import detect_orb, match_pair, parallel_map
from specialist.vi_ba.finger_kinematics import FingerReproj, build_finger_reproj
from specialist.vi_ba.imu import (
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
  FrontendConfig,
  decode_dual_keyframes,
  filter_tracks_by_parallax,
  keyframe_timestamps,
  lens_setups,
  probe_capture,
  resolve_keyframe_every,
  run_lens_frontend,
  triangulate_lens,
  unproject_tracks,
)
from specialist.vi_ba.tag_detection import (
  TagDetection,
  detect_all_tags,
  lens_T_tag_per_kf,
)
from specialist.vi_ba.urdf_joints import read_revolute_joints

Float64Array = npt.NDArray[np.float64]

_A = TypeVar("_A")
_B = TypeVar("_B")

# ORB loop-closure track recovery (``_revisit_tracks``).
_REVISIT_N_NEIGHBORS = 4  # spatially-closest non-adjacent revisit kfs to re-match
_REVISIT_MATCH_RATIO = 0.7  # Lowe ratio-test threshold
_REVISIT_RANSAC_THRESHOLD = 4.5  # homography RANSAC reprojection tolerance, px
_REVISIT_MIN_VIEWS = 3  # keyframes a union-found track must span to be kept
_REVISIT_MIN_INLIERS = 20  # geometrically-verified matches required per pair

# Visual-odometry pose-spike rejection (``_reject_vo_pose_outliers``).
_VO_REJECT_FACTOR = 8.0  # residual/median-step ratio marking a PnP teleport

__all__ = [
  "PreparedCamera",
  "CameraArgs",
  "prepare_camera",
  "run_two_concurrently",
]


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
  start: float = 0.0
  duration: float | None = None
  keyframe_every: int = 0  # 0 = auto (round(fps/10)); see WindowCfg.keyframe_every
  # Retain decoded keyframes (as JPEG) for the Rerun frame overlay.
  keep_frames: bool = False
  urdf_path: Path | None = None  # gripper URDF for finger joint kinematics
  # 0 disables the corresponding URDF revolute joint angle recovery.
  left_finger_tag_id: int = 0
  right_finger_tag_id: int = 0
  frontend: FrontendCfg = field(default_factory=FrontendCfg)
  imu: ImuCfg = field(default_factory=ImuCfg)
  gripper: GripperCfg = field(default_factory=GripperCfg)


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
  # Prep-reference anchor: the most-sighted configured id.
  prep_anchor_tag_id: int
  # ``{tag_id: {kf: {side: lens_T_tag}}}``.
  anchor_obs_by_tag: dict[int, dict[int, dict[str, gtsam.Pose3]]]
  anchor_locals_by_tag: dict[int, gtsam.Pose3]
  body_T_base_tag: gtsam.Pose3 | None
  keyframe_times: Float64Array
  # JPEG-encoded keyframes for the Rerun overlay (empty unless retained).
  front_frames: list[bytes]
  rear_frames: list[bytes]
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
  n_keyframes: int
  # Joint name -> solved angle about the joint z-axis per keyframe, NaN where unseen.
  joint_angles: dict[str, list[float]] = field(default_factory=dict)

  def vi_graph_inputs(self) -> dict[str, object]:
    """Bundle this tracker's state + lens data for ``build_vi_graph``."""
    return {
      "body_poses": self.body_poses,
      "velocities": self.velocities,
      "biases": self.biases,
      "preintegrated": self.preintegrated,
      "front": self.front_lens,
      "front_tracks": self.front_tracks_tri,
      "front_points": self.front_points,
      "rear": self.rear_lens,
      "rear_tracks": self.rear_tracks_tri,
      "rear_points": self.rear_points,
    }


def _build_non_anchor_tag_masks(
  frames: list[np.ndarray],
  tags_per_kf: list[list[TagDetection]],
  exclude_tag_ids: set[int],
  pad_px: int,
) -> list[np.ndarray]:
  masks: list[np.ndarray] = []
  for frame, tags in zip(frames, tags_per_kf, strict=True):
    h, w = frame.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for tag in tags:
      if int(tag.tag_id) in exclude_tag_ids:
        continue
      corners = np.asarray(tag.corners, dtype=np.float64)
      x0 = max(int(np.floor(corners[:, 0].min())) - pad_px, 0)
      y0 = max(int(np.floor(corners[:, 1].min())) - pad_px, 0)
      x1 = min(int(np.ceil(corners[:, 0].max())) + pad_px, w)
      y1 = min(int(np.ceil(corners[:, 1].max())) + pad_px, h)
      if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 0
    masks.append(mask)
  return masks


def _apply_static_mask_rects(
  masks: list[np.ndarray], rects: list[tuple[int, int, int, int]]
) -> list[np.ndarray]:
  if not rects or not masks:
    return masks
  h, w = masks[0].shape[:2]
  for x1, y1, x2, y2 in rects:
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
      raise ValueError(
        f"mask_rear_rects entry {(x1, y1, x2, y2)} is empty after clamping to the "
        f"{w}x{h} frame — check the coordinate order (x1 y1 x2 y2) and bounds"
      )
    for m in masks:
      m[y1c:y2c, x1c:x2c] = 0
  return masks


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
  incidences: list[float] = []
  for sides in obs.values():
    for pose in sides.values():
      t = np.asarray(pose.translation(), dtype=np.float64)
      dist = float(np.linalg.norm(t))
      if dist < 1e-9:
        continue
      cos_incidence = -float(pose.rotation().matrix()[:, 2] @ t) / dist
      incidences.append(float(np.degrees(np.arccos(np.clip(cos_incidence, -1.0, 1.0)))))
  if not edges:
    return

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
  assert lens_T_tag is not None
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
  frames: list[np.ndarray],
  masks: list[np.ndarray],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  n_features: int,
  base_track_id: int,
  max_center_dist: float,
  min_kf_gap: int,
) -> dict[int, dict[int, Float64Array]]:
  """Recover tracks broken by fast lifts via ORB loop closure."""
  orb = parallel_map(lambda f, m: detect_orb(f, n_features, 20, m), frames, masks)
  centers = np.asarray([_translation(p.compose(lens.body_T_lens)) for p in body_poses])
  n = len(frames)
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
  min_track_length: int,
  max_landmark_depth: float,
  slot: str,
  label: str,
  anchor_kfs: set[int] | None = None,
) -> None:
  """Add ``revisits`` to the front lens; ``anchor_kfs`` gates their triangulation."""
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
    f"  [{slot}] {label}: +{len(revisit_pts)} landmarks "
    f"from {len(revisits)} revisit tracks"
  )


def _reject_vo_pose_outliers(
  lens_poses: dict[int, gtsam.Pose3],
) -> tuple[dict[int, gtsam.Pose3], list[int]]:
  """Drop incremental-SfM keyframes whose camera center is a local spike."""
  kept = dict(lens_poses)
  removed: list[int] = []
  for _ in range(len(lens_poses)):
    ks = sorted(kept)
    if len(ks) < 5:
      break
    c = {k: _translation(kept[k]) for k in ks}
    steps = [
      float(np.linalg.norm(c[ks[i + 1]] - c[ks[i]])) / (ks[i + 1] - ks[i])
      for i in range(len(ks) - 1)
    ]
    median_step = float(np.median(steps))
    if median_step < 1e-12:
      break
    worst_k: int | None = None
    worst_r = 0.0
    for i in range(1, len(ks) - 1):
      kp, k, kn = ks[i - 1], ks[i], ks[i + 1]
      f = (k - kp) / (kn - kp)
      pred = (1.0 - f) * c[kp] + f * c[kn]
      r = float(np.linalg.norm(c[k] - pred))
      if r > worst_r:
        worst_r, worst_k = r, k
    if worst_k is not None and worst_r > _VO_REJECT_FACTOR * median_step:
      del kept[worst_k]
      removed.append(worst_k)
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
  if len(lens_poses) < max(8, n_keyframes // 4):
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
) -> tuple[LensSetup, gtsam.Pose3 | None]:
  """The anchor observation's lens and ``lens_T_tag``, front lens preferred."""
  lens = front_lens if "front" in sides else rear_lens
  lens_T_tag = sides.get("front", sides.get("rear"))
  return lens, lens_T_tag


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
  keyframe_times: Float64Array,
  body_poses_unscaled: list[gtsam.Pose3],
) -> tuple[ImuSamples, gtsam.imuBias.ConstantBias, Float64Array | None]:
  """Body-frame IMU, factory bias, and stationary-window world gravity or ``None``."""
  body_R_imu = np.asarray(args.imu.body_R_imu, dtype=np.float64)
  imu = _body_imu_samples(data, body_R_imu)
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
    if 8.0 < stationary_mag < 11.0 and abs(stationary_mag - 9.81) > 0.05:
      accel_scale = 9.81 / stationary_mag
      imu = ImuSamples(
        t=imu.t, gyro=imu.gyro, accel=(imu.accel * accel_scale).astype(np.float64)
      )
      stationary_mean = stationary_mean * accel_scale
    gravity_body = -stationary_mean / float(np.linalg.norm(stationary_mean)) * 9.81
    t_stationary_mid = 0.5 * (imu.t[first_start] + imu.t[first_end - 1])
    kf_for_g = int(np.argmin(np.abs(keyframe_times - t_stationary_mid)))
    gravity_world_initial = (
      body_poses_unscaled[kf_for_g].rotation().matrix() @ gravity_body
    )
  return imu, bias0, gravity_world_initial


def _recover_gripper_tags(
  args: CameraArgs,
  rear_tags_per_kf: list[list[TagDetection]],
  rear_lens: LensSetup,
  n_keyframes: int,
  kf_rotations: list[Float64Array],
) -> tuple[gtsam.Pose3 | None, dict[str, FingerReproj]]:
  """Base-tag body pose and per-finger reprojection inputs."""
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
  return body_T_base_tag, finger_reproj


def _snap_to_anchor_and_fill_gaps(
  args: CameraArgs,
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]],
  front_lens: LensSetup,
  rear_lens: LensSetup,
  body_poses: list[gtsam.Pose3],
  velocities: list[Float64Array],
  pim: list[gtsam.PreintegratedCombinedMeasurements],
  bias0: gtsam.imuBias.ConstantBias,
  n_keyframes: int,
  slot: str,
) -> gtsam.Pose3:
  """Local-world anchor pose; without VO, snap anchored keyframes, IMU-fill gaps."""
  anchor_local = _first_sighting_local(
    anchor_obs_per_kf, front_lens, rear_lens, body_poses
  )
  if args.frontend.use_visual_odometry:
    return anchor_local

  for kf, kf_sides in anchor_obs_per_kf.items():
    lens, lens_T_tag = _side_lens_and_pose(kf_sides, front_lens, rear_lens)
    assert lens_T_tag is not None
    body_poses[kf] = anchor_local.compose(
      lens.body_T_lens.compose(lens_T_tag).inverse()
    )
  sorted_kfs = sorted(anchor_obs_per_kf.keys())
  if len(sorted_kfs) >= 2:
    jumps = [
      float(
        np.linalg.norm(
          _translation(body_poses[sorted_kfs[i]])
          - _translation(body_poses[sorted_kfs[i - 1]])
        )
      )
      for i in range(1, len(sorted_kfs))
    ]
    i = int(np.argmax(jumps))
    print(f"  [{slot}] max anchor-init Δp: {jumps[i]:.3f} m at kf {sorted_kfs[i + 1]}")
  first_anchor = sorted_kfs[0]
  for k in range(first_anchor):
    body_poses[k] = body_poses[first_anchor]
    velocities[k] = velocities[first_anchor]
  for i in range(len(sorted_kfs)):
    a = sorted_kfs[i]
    b = sorted_kfs[i + 1] if i + 1 < len(sorted_kfs) else n_keyframes
    if b - a <= 1:
      continue
    # A gap ending on an anchor kf spreads the drift across it; the trailing gap cannot.
    bidir = b < n_keyframes
    nav = gtsam.NavState(body_poses[a], velocities[a])
    fwd_pose: dict[int, gtsam.Pose3] = {}
    fwd_vel: dict[int, Float64Array] = {}
    for k in range(a + 1, b + 1 if bidir else b):
      nav = pim[k - 1].predict(nav, bias0)
      fwd_pose[k] = nav.pose()
      fwd_vel[k] = np.asarray(nav.velocity(), dtype=np.float64)
    dp = (
      _translation(body_poses[b]) - _translation(fwd_pose[b]) if bidir else np.zeros(3)
    )
    for k in range(a + 1, b):
      if k in anchor_obs_per_kf:
        continue
      p = fwd_pose[k]
      if bidir:
        tk = _translation(p) + (k - a) / (b - a) * dp
        body_poses[k] = gtsam.Pose3(p.rotation(), gtsam.Point3(*tk))
      else:
        body_poses[k] = p
      velocities[k] = fwd_vel[k]
  return anchor_local


def prepare_camera(slot: str, args: CameraArgs) -> PreparedCamera:
  """Every per-camera stage; body poses land in the camera's gravity-aligned world."""
  data = load_insv(args.insv_path)
  offset_v3 = parse_offset_v3(data.offset_v3)
  rear_path, fps, decoded_width = probe_capture(args.insv_path)
  front_lens, rear_lens = lens_setups(offset_v3, decoded_width)

  keyframe_every = resolve_keyframe_every(args.keyframe_every, fps)
  front_gray, rear_gray, front_color, rear_color = decode_dual_keyframes(
    args.insv_path,
    rear_path,
    fps,
    args.start,
    args.duration,
    keyframe_every,
    keep_color=args.keep_frames,
  )
  n_keyframes = len(front_gray)
  if n_keyframes < 3:
    raise RuntimeError(f"only {n_keyframes} keyframes decoded — window too short")
  keyframe_times = keyframe_timestamps(args.start, fps, keyframe_every, n_keyframes)

  front_tags_per_kf, rear_tags_per_kf = run_two_concurrently(
    lambda: detect_all_tags(front_gray),
    lambda: detect_all_tags(rear_gray),
  )

  body_R_imu = np.asarray(args.imu.body_R_imu, dtype=np.float64)
  body_deltas = keyframe_delta_rotations(
    _body_imu_samples(data, body_R_imu), keyframe_times
  )

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

  anchor_tag_ids = set(obs_by_tag)
  front_masks = _build_non_anchor_tag_masks(
    front_gray, front_tags_per_kf, anchor_tag_ids, args.frontend.mask_tag_pad
  )
  rear_masks = _build_non_anchor_tag_masks(
    rear_gray, rear_tags_per_kf, anchor_tag_ids, args.frontend.mask_tag_pad
  )
  static_rects = list(args.frontend.mask_rear_rects)
  _apply_static_mask_rects(rear_masks, static_rects)

  cfg = FrontendConfig(
    n_features=args.frontend.orb_n_features,
    ransac_threshold=args.frontend.ransac_threshold,
    max_half_angle=args.frontend.max_half_angle,
  )
  (front_feature_frames, front_tracks), (rear_feature_frames, rear_tracks) = (
    run_two_concurrently(
      lambda: run_lens_frontend(
        front_gray,
        front_lens.params,
        cfg,
        masks=front_masks,
      ),
      lambda: run_lens_frontend(
        rear_gray,
        rear_lens.params,
        cfg,
        masks=rear_masks,
      ),
    )
  )
  front_kept = {
    t: o for t, o in front_tracks.items() if len(o) >= args.frontend.min_track_length
  }
  rear_kept = {
    t: o for t, o in rear_tracks.items() if len(o) >= args.frontend.min_track_length
  }
  front_rejected: dict[int, dict[int, Float64Array]] = {}
  rear_rejected: dict[int, dict[int, Float64Array]] = {}
  if args.frontend.min_parallax > 0.0:
    front_kept, front_rejected = filter_tracks_by_parallax(
      front_kept, front_lens.params, args.frontend.min_parallax
    )
    rear_kept, rear_rejected = filter_tracks_by_parallax(
      rear_kept, rear_lens.params, args.frontend.min_parallax
    )

  f_first, f_last, f_shared = pick_init_pair(
    front_kept, n_keyframes, args.frontend.init_min_shared_tracks
  )
  r_first, r_last, r_shared = pick_init_pair(
    rear_kept, n_keyframes, args.frontend.init_min_shared_tracks
  )
  if f_shared >= r_shared:
    init_tracks, init_params, init_extrinsic = (
      front_kept,
      front_lens.params,
      front_lens.body_T_lens,
    )
    init_first, init_last, init_shared = f_first, f_last, f_shared
  else:
    init_tracks, init_params, init_extrinsic = (
      rear_kept,
      rear_lens.params,
      rear_lens.body_T_lens,
    )
    init_first, init_last, init_shared = r_first, r_last, r_shared
  if init_shared < 8:
    raise RuntimeError(f"{slot}: no lens has ≥8 shared tracks (best={init_shared})")

  shared = [
    obs for obs in init_tracks.values() if init_first in obs and init_last in obs
  ]
  pix_a = np.asarray([obs[init_first] for obs in shared], dtype=np.float64)
  pix_b = np.asarray([obs[init_last] for obs in shared], dtype=np.float64)
  bears_a = mei_unproject(pix_a, init_params)
  bears_b = mei_unproject(pix_b, init_params)
  r_cw_last, t_cw_last, _ = two_view_essential(bears_a, bears_b)
  lens_poses = interpolate_camera_poses(r_cw_last, t_cw_last, n_keyframes)
  body_poses_unscaled = body_poses_from_lens_poses(lens_poses, init_extrinsic)

  # Visual odometry recovers the trajectory shape from front-lens correspondences.
  anchor_seed_used = False
  if args.frontend.use_visual_odometry:
    sfm_cams, _ = incremental_sfm(
      _bearings_by_track(front_kept, front_lens.params),
      min_shared=args.frontend.init_min_shared_tracks,
    )
    sfm_cams, vo_outliers = _reject_vo_pose_outliers(sfm_cams)
    if vo_outliers:
      print(
        f"  [{slot}] visual-odometry rejected {len(vo_outliers)} rotation-spike kfs"
      )
    if len(sfm_cams) >= max(8, n_keyframes // 4):
      body_poses_unscaled = body_poses_from_lens_poses(
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
        body_poses_unscaled = anchor_seed
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
    args, data, keyframe_times, body_poses_unscaled
  )

  front_unscaled = triangulate_lens(
    front_kept,
    body_poses_unscaled,
    front_lens.body_T_lens,
    front_lens.params,
    args.frontend.min_track_length,
    min_landmark_depth=0.01,
    max_landmark_depth=1000.0,
  )
  rear_unscaled = triangulate_lens(
    rear_kept,
    body_poses_unscaled,
    rear_lens.body_T_lens,
    rear_lens.params,
    args.frontend.min_track_length,
    min_landmark_depth=0.01,
    max_landmark_depth=1000.0,
  )

  # Placeholder vertical gravity; without a stationary window the LSQ-fit value sticks.
  pim_gravity = (
    (
      float(gravity_world_initial[0]),
      float(gravity_world_initial[1]),
      float(gravity_world_initial[2]),
    )
    if gravity_world_initial is not None
    else (0.0, 0.0, -9.81)
  )

  def _pim_params(
    gravity: tuple[float, float, float],
  ) -> gtsam.PreintegrationCombinedParams:
    return make_preintegration_params(
      gyro_noise_sigma=args.imu.gyro_noise_sigma,
      accel_noise_sigma=args.imu.accel_noise_sigma,
      gyro_bias_rw_sigma=args.imu.gyro_random_walk,
      accel_bias_rw_sigma=args.imu.accel_random_walk,
      gravity_world=gravity,
    )

  params_initial = _pim_params(pim_gravity)
  # Scale: 1.0 for an anchor seed, else from a visible anchor, else floated on the IMU.
  anchor_scale = _anchor_metric_scale(
    body_poses_unscaled,
    anchor_obs_per_kf,
    front_lens.body_T_lens,
    rear_lens.body_T_lens,
  )
  fixed_scale = 1.0 if anchor_seed_used else anchor_scale
  pim_init = preintegrate_between_keyframes(params_initial, imu, keyframe_times, bias0)

  def _align(poses: list[gtsam.Pose3]) -> Alignment:
    return visual_inertial_align(
      poses,
      pim_init,
      fixed_gravity=gravity_world_initial,
      fixed_scale=fixed_scale,
    )

  alignment = _align(body_poses_unscaled)
  if alignment.scale < 0.0:
    origin = _translation(body_poses_unscaled[0])
    body_poses_unscaled = [
      gtsam.Pose3(
        p.rotation(),
        gtsam.Point3(*(2.0 * origin - _translation(p))),
      )
      for p in body_poses_unscaled
    ]
    alignment = _align(body_poses_unscaled)
  # Prefer the IMU stationary-window gravity to the LSQ fit, which can tilt the world.
  if gravity_world_initial is not None:
    alignment = replace(alignment, gravity_world=gravity_world_initial)
  print(f"  [{slot}] VI scale={alignment.scale:.4f}")
  body_poses, world_R_prealign = apply_alignment_to_poses(
    body_poses_unscaled, alignment
  )
  world_rot_mat = world_R_prealign.matrix()
  velocities = [world_rot_mat @ v for v in alignment.velocities]
  params_aligned = _pim_params((0.0, 0.0, -9.81))
  pim = preintegrate_between_keyframes(params_aligned, imu, keyframe_times, bias0)

  use_anchor = bool(anchor_obs_per_kf)
  anchor_local: gtsam.Pose3 | None = (
    _snap_to_anchor_and_fill_gaps(
      args,
      anchor_obs_per_kf,
      front_lens,
      rear_lens,
      body_poses,
      velocities,
      pim,
      bias0,
      n_keyframes,
      slot,
    )
    if use_anchor
    else None
  )

  # Each sighted anchor's pose in this camera's local world, for the shared-world BFS.
  anchor_locals_by_tag: dict[int, gtsam.Pose3] = {}
  if anchor_local is not None:
    anchor_locals_by_tag[prep_anchor_tag_id] = anchor_local
  for tag_id, obs in obs_by_tag.items():
    if tag_id == prep_anchor_tag_id or not obs:
      continue
    anchor_locals_by_tag[tag_id] = _first_sighting_local(
      obs, front_lens, rear_lens, body_poses
    )

  revisits: dict[int, dict[int, Float64Array]] = {}
  if args.frontend.revisit and (
    args.frontend.use_visual_odometry or (args.frontend.anchored_revisit and use_anchor)
  ):
    revisits = _revisit_tracks(
      front_gray,
      front_masks,
      body_poses,
      front_lens,
      args.frontend.orb_n_features,
      max(front_kept, default=0),
      args.frontend.loop_closure_max_dist,
      args.frontend.loop_closure_min_kf_gap,
    )
  if use_anchor:
    anchor_kfs = set(anchor_obs_per_kf.keys())
    if not args.frontend.use_visual_odometry:
      # Without VO, keep only the tracks covisible with an anchor keyframe.
      front_kept = {
        t: front_kept[t] for t in front_kept if anchor_kfs & front_kept[t].keys()
      }
      rear_kept = {
        t: rear_kept[t] for t in rear_kept if anchor_kfs & rear_kept[t].keys()
      }
    front_points = triangulate_lens(
      {t: front_kept[t] for t in front_kept if t in front_unscaled},
      body_poses,
      front_lens.body_T_lens,
      front_lens.params,
      args.frontend.min_track_length,
      max_landmark_depth=args.frontend.max_landmark_depth,
    )
    rear_points = triangulate_lens(
      {t: rear_kept[t] for t in rear_kept if t in rear_unscaled},
      body_poses,
      rear_lens.body_T_lens,
      rear_lens.params,
      args.frontend.min_track_length,
      max_landmark_depth=args.frontend.max_landmark_depth,
    )
    if args.frontend.use_visual_odometry:
      cap = args.frontend.vo_max_landmarks_per_lens
      front_points = _cap_landmarks(front_points, front_kept, cap)
      rear_points = _cap_landmarks(rear_points, rear_kept, cap)
      if revisits:
        # Merge after the cap so the per-lens budget never culls the revisit landmarks.
        _add_revisit_landmarks(
          front_points,
          front_kept,
          revisits,
          body_poses,
          front_lens,
          args.frontend.min_track_length,
          args.frontend.max_landmark_depth,
          slot,
          "loop closure",
          anchor_kfs=anchor_kfs if args.frontend.anchored_revisit else None,
        )
    elif revisits:
      # Merge after gating: loop landmarks tie the drifting tail to the early keyframes.
      _add_revisit_landmarks(
        front_points,
        front_kept,
        revisits,
        body_poses,
        front_lens,
        args.frontend.min_track_length,
        args.frontend.max_landmark_depth,
        slot,
        "anchored loop closure",
        anchor_kfs=anchor_kfs,
      )
  else:
    align_mat = alignment.scale * world_R_prealign.matrix()
    front_points = {
      tid: align_mat @ np.asarray(front_unscaled[tid], dtype=np.float64)
      for tid in front_unscaled
    }
    rear_points = {
      tid: align_mat @ np.asarray(rear_unscaled[tid], dtype=np.float64)
      for tid in rear_unscaled
    }

  front_tracks_tri = {t: front_kept[t] for t in front_points if t in front_kept}
  rear_tracks_tri = {t: rear_kept[t] for t in rear_points if t in rear_kept}
  if args.frontend.use_visual_odometry:
    m = args.frontend.vo_max_obs_per_track
    front_tracks_tri = {t: _subsample_obs(o, m) for t, o in front_tracks_tri.items()}
    rear_tracks_tri = {t: _subsample_obs(o, m) for t, o in rear_tracks_tri.items()}

  v_max = float(max(np.linalg.norm(v) for v in velocities))
  print(f"  [{slot}] VIO LSQ |v|max={v_max:.2f} m/s")
  biases = [bias0 for _ in range(n_keyframes)]

  body_T_base_tag, finger_reproj = _recover_gripper_tags(
    args, rear_tags_per_kf, rear_lens, n_keyframes, rear_kf_rotations
  )

  return PreparedCamera(
    name=slot,
    body_poses=body_poses,
    velocities=velocities,
    biases=biases,
    preintegrated=pim,
    front_lens=front_lens,
    rear_lens=rear_lens,
    front_tracks_tri=front_tracks_tri,
    rear_tracks_tri=rear_tracks_tri,
    front_points=front_points,
    rear_points=rear_points,
    decoded_width=decoded_width,
    prep_anchor_tag_id=prep_anchor_tag_id,
    anchor_obs_by_tag=obs_by_tag,
    anchor_locals_by_tag=anchor_locals_by_tag,
    body_T_base_tag=body_T_base_tag,
    keyframe_times=keyframe_times,
    front_frames=front_color,
    rear_frames=rear_color,
    front_keypoints=[f.keypoints for f in front_feature_frames],
    rear_keypoints=[f.keypoints for f in rear_feature_frames],
    front_tags_per_kf=front_tags_per_kf,
    rear_tags_per_kf=rear_tags_per_kf,
    front_rejected=front_rejected,
    rear_rejected=rear_rejected,
    rear_static_mask_rects=static_rects,
    finger_reproj=finger_reproj,
    left_finger_tag_id=args.left_finger_tag_id,
    right_finger_tag_id=args.right_finger_tag_id,
    n_keyframes=n_keyframes,
  )
