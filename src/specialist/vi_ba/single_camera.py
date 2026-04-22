"""One camera's lenses and IMU prepared for the joint graph."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import mei_unproject
from specialist.loader.parse_insv import InsvData, load_insv
from specialist.record.offset_v3 import parse_offset_v3
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.configs import (
  AnchorCfg,
  FrontendCfg,
  GripperCfg,
  ImuCfg,
)
from specialist.vi_ba.imu import (
  ImuSamples,
  factory_imu_bias,
  find_stationary_windows,
  keyframe_delta_rotations,
  make_preintegration_params,
  preintegrate_between_keyframes,
)
from specialist.vi_ba.init import (
  apply_alignment_to_poses,
  body_poses_from_lens_poses,
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
)
from specialist.vi_ba.tag_detection import (
  TagDetection,
  detect_all_tags,
  estimate_lens_T_tag,
  lens_T_tag_per_kf,
)

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "PreparedCamera",
  "CameraArgs",
  "prepare_camera",
]


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
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]]
  anchor_local: gtsam.Pose3 | None
  body_T_base_tag: gtsam.Pose3 | None
  # URDF joint name → per-keyframe angle about the URDF z-axis, NaN with no tag pose.
  joint_angles: dict[str, list[float]]
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
  body_T_left_finger_per_kf: dict[int, gtsam.Pose3]
  body_T_right_finger_per_kf: dict[int, gtsam.Pose3]
  left_finger_tag_id: int
  right_finger_tag_id: int
  n_keyframes: int

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
  exclude_tag_id: int,
  pad_px: int,
) -> list[np.ndarray]:
  masks: list[np.ndarray] = []
  for frame, tags in zip(frames, tags_per_kf, strict=True):
    h, w = frame.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for tag in tags:
      if exclude_tag_id >= 0 and int(tag.tag_id) == exclude_tag_id:
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


def _per_kf_cam_T_tag(
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  tag_id: int,
  tag_size: float,
  n_keyframes: int,
  kf_rotations: list[Float64Array],
) -> list[gtsam.Pose3 | None]:
  """Per-keyframe ``lens_T_tag`` of one tag, IPPE resolved by the gyro-rotated prior."""
  out: list[gtsam.Pose3 | None] = [None] * n_keyframes
  for kf, p in lens_T_tag_per_kf(
    tags_per_kf[:n_keyframes], tag_id, tag_size, lens.params, kf_rotations
  ).items():
    out[kf] = p
  return out


def _mean_pose(poses: list[gtsam.Pose3]) -> gtsam.Pose3:
  """Mean translation and log-mean rotation about the first pose."""
  ref = poses[0]
  translations = np.stack(
    [np.asarray(p.translation(), dtype=np.float64) for p in poses]
  )
  log_sum = np.zeros(3)
  for p in poses:
    log_sum += gtsam.Rot3.Logmap(ref.rotation().between(p.rotation()))
  delta = gtsam.Rot3.Expmap(log_sum / len(poses))
  return gtsam.Pose3(
    ref.rotation().compose(delta), gtsam.Point3(*translations.mean(axis=0))
  )


def _finger_joint_angles_per_kf(
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  base_tag_id: int,
  finger_tag_id: int,
  tag_size: float,
  n_keyframes: int,
  kf_rotations: list[Float64Array],
) -> list[float]:
  """Per keyframe, the finger's signed +z angle to the base tag; NaN where unseen."""
  cam_T_base_per_kf = _per_kf_cam_T_tag(
    tags_per_kf, lens, base_tag_id, tag_size, n_keyframes, kf_rotations
  )
  lens_T_finger_tag_per_kf = _per_kf_cam_T_tag(
    tags_per_kf, lens, finger_tag_id, tag_size, n_keyframes, kf_rotations
  )
  observed_base = [p for p in cam_T_base_per_kf if p is not None]
  if not observed_base:
    return [float("nan")] * n_keyframes
  lens_T_base_tag = _mean_pose(observed_base)
  out: list[float] = [float("nan")] * n_keyframes
  for kf in range(n_keyframes):
    lens_T_finger_tag = lens_T_finger_tag_per_kf[kf]
    if lens_T_finger_tag is None:
      continue
    base_tag_T_finger_tag = lens_T_base_tag.between(lens_T_finger_tag)
    out[kf] = float(gtsam.Rot3.Logmap(base_tag_T_finger_tag.rotation())[2])
  return out


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

  front_tags_per_kf = detect_all_tags(front_gray)
  rear_tags_per_kf = detect_all_tags(rear_gray)

  # Gyro rotation increments carry the tag disambiguation prior through fast rotation.
  body_R_imu = np.asarray(args.imu.body_R_imu, dtype=np.float64)
  body_deltas = keyframe_delta_rotations(
    _body_imu_samples(data, body_R_imu), keyframe_times
  )

  def lens_kf_rotations(lens: LensSetup) -> list[Float64Array]:
    r_bl = lens.body_T_lens.rotation().matrix()
    return [r_bl.T @ d.T @ r_bl for d in body_deltas]

  front_kf_rotations = lens_kf_rotations(front_lens)
  rear_kf_rotations = lens_kf_rotations(rear_lens)

  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]] = {}
  if args.anchor.tag_id >= 0:
    for side, lens, tags, rotations in (
      ("front", front_lens, front_tags_per_kf, front_kf_rotations),
      ("rear", rear_lens, rear_tags_per_kf, rear_kf_rotations),
    ):
      for kf, pose in lens_T_tag_per_kf(
        tags,
        args.anchor.tag_id,
        args.anchor.tag_size,
        lens.params,
        kf_rotations=rotations,
      ).items():
        anchor_obs_per_kf.setdefault(kf, {})[side] = pose

  front_masks = _build_non_anchor_tag_masks(
    front_gray, front_tags_per_kf, args.anchor.tag_id, args.frontend.mask_tag_pad
  )
  rear_masks = _build_non_anchor_tag_masks(
    rear_gray, rear_tags_per_kf, args.anchor.tag_id, args.frontend.mask_tag_pad
  )
  static_rects = list(args.frontend.mask_rear_rects)
  _apply_static_mask_rects(rear_masks, static_rects)

  cfg = FrontendConfig(
    n_features=args.frontend.orb_n_features,
    ransac_threshold=args.frontend.ransac_threshold,
    max_half_angle=args.frontend.max_half_angle,
  )
  front_feature_frames, front_tracks = run_lens_frontend(
    front_gray,
    front_lens.params,
    cfg,
    masks=front_masks,
  )
  rear_feature_frames, rear_tracks = run_lens_frontend(
    rear_gray,
    rear_lens.params,
    cfg,
    masks=rear_masks,
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
  body_poses_visual = body_poses_unscaled

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
  params_pim = make_preintegration_params(
    gyro_noise_sigma=args.imu.gyro_noise_sigma,
    accel_noise_sigma=args.imu.accel_noise_sigma,
    gyro_bias_rw_sigma=args.imu.gyro_random_walk,
    accel_bias_rw_sigma=args.imu.accel_random_walk,
    gravity_world=pim_gravity,
  )
  pim_init = preintegrate_between_keyframes(params_pim, imu, keyframe_times, bias0)
  alignment = visual_inertial_align(body_poses_visual, pim_init)
  if alignment.scale < 0.0:
    origin = np.asarray(body_poses_visual[0].translation(), dtype=np.float64)
    body_poses_visual = [
      gtsam.Pose3(
        p.rotation(),
        gtsam.Point3(*(2.0 * origin - np.asarray(p.translation(), dtype=np.float64))),
      )
      for p in body_poses_visual
    ]
    alignment = visual_inertial_align(body_poses_visual, pim_init)
  # Prefer the IMU stationary-window gravity to the LSQ fit, which can tilt the world.
  if gravity_world_initial is not None:
    alignment = replace(alignment, gravity_world=gravity_world_initial)
  body_poses, world_R_prealign = apply_alignment_to_poses(body_poses_visual, alignment)
  world_rot_mat = world_R_prealign.matrix()
  velocities = [world_rot_mat @ v for v in alignment.velocities]
  params_aligned = make_preintegration_params(
    gyro_noise_sigma=args.imu.gyro_noise_sigma,
    accel_noise_sigma=args.imu.accel_noise_sigma,
    gyro_bias_rw_sigma=args.imu.gyro_random_walk,
    accel_bias_rw_sigma=args.imu.accel_random_walk,
    gravity_world=(0.0, 0.0, -9.81),
  )
  pim = preintegrate_between_keyframes(params_aligned, imu, keyframe_times, bias0)

  use_anchor = args.anchor.tag_id >= 0 and bool(anchor_obs_per_kf)
  anchor_local: gtsam.Pose3 | None = None
  if use_anchor:
    first_kf = min(anchor_obs_per_kf.keys())
    sides = anchor_obs_per_kf[first_kf]
    first_lens, first_cam_T_tag = _side_lens_and_pose(sides, front_lens, rear_lens)
    assert first_cam_T_tag is not None
    anchor_local = (
      body_poses[first_kf].compose(first_lens.body_T_lens).compose(first_cam_T_tag)
    )
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
            np.asarray(body_poses[sorted_kfs[i]].translation())
            - np.asarray(body_poses[sorted_kfs[i - 1]].translation())
          )
        )
        for i in range(1, len(sorted_kfs))
      ]
      i = int(np.argmax(jumps))
      print(
        f"  [{slot}] max anchor-init Δp: {jumps[i]:.3f} m at kf {sorted_kfs[i + 1]}"
      )
    # IMU forward-propagation from each anchor kf fills the gap kfs.
    first_anchor = sorted_kfs[0]
    for k in range(first_anchor):
      body_poses[k] = body_poses[first_anchor]
      velocities[k] = velocities[first_anchor]
    for i in range(len(sorted_kfs)):
      a = sorted_kfs[i]
      b = sorted_kfs[i + 1] if i + 1 < len(sorted_kfs) else n_keyframes
      if b - a <= 1:
        continue
      nav = gtsam.NavState(body_poses[a], velocities[a])
      for k in range(a + 1, b):
        nav = pim[k - 1].predict(nav, bias0)
        if k in anchor_obs_per_kf:
          continue
        body_poses[k] = nav.pose()
        velocities[k] = np.asarray(nav.velocity(), dtype=np.float64)

  align_mat = alignment.scale * world_R_prealign.matrix()
  front_unscaled_metric = {
    tid: align_mat @ np.asarray(front_unscaled[tid], dtype=np.float64)
    for tid in front_unscaled
  }
  rear_unscaled_metric = {
    tid: align_mat @ np.asarray(rear_unscaled[tid], dtype=np.float64)
    for tid in rear_unscaled
  }
  if use_anchor:
    anchor_kfs = set(anchor_obs_per_kf.keys())
    front_kept = {
      t: front_kept[t] for t in front_kept if anchor_kfs & front_kept[t].keys()
    }
    rear_kept = {t: rear_kept[t] for t in rear_kept if anchor_kfs & rear_kept[t].keys()}
    front_points = triangulate_lens(
      {t: front_kept[t] for t in front_kept if t in front_unscaled},
      body_poses,
      front_lens.body_T_lens,
      front_lens.params,
      args.frontend.min_track_length,
    )
    rear_points = triangulate_lens(
      {t: rear_kept[t] for t in rear_kept if t in rear_unscaled},
      body_poses,
      rear_lens.body_T_lens,
      rear_lens.params,
      args.frontend.min_track_length,
    )
  else:
    front_points = front_unscaled_metric
    rear_points = rear_unscaled_metric

  front_tracks_tri = {t: front_kept[t] for t in front_points if t in front_kept}
  rear_tracks_tri = {t: rear_kept[t] for t in rear_points if t in rear_kept}

  v_max = float(max(np.linalg.norm(v) for v in velocities))
  print(f"  [{slot}] VIO LSQ |v|max={v_max:.2f} m/s")
  biases = [bias0 for _ in range(n_keyframes)]

  base_tag_obs = lens_T_tag_per_kf(
    rear_tags_per_kf,
    tag_id=args.base_tag_id,
    tag_size=args.gripper.tag_size,
    params=rear_lens.params,
    kf_rotations=rear_kf_rotations,
  )
  body_T_base_tag: gtsam.Pose3 | None = None
  if base_tag_obs:
    body_T_base_tag = _mean_pose(
      [rear_lens.body_T_lens.compose(p) for p in base_tag_obs.values()]
    )

  def _per_kf_body_T_finger(finger_id: int) -> dict[int, gtsam.Pose3]:
    if finger_id <= 0:
      return {}
    out: dict[int, gtsam.Pose3] = {}
    for kf in range(min(n_keyframes, len(rear_tags_per_kf))):
      corners = next(
        (d.corners for d in rear_tags_per_kf[kf] if d.tag_id == finger_id),
        None,
      )
      if corners is None:
        continue
      lens_T_finger_tag = estimate_lens_T_tag(
        corners, args.gripper.tag_size, rear_lens.params
      )
      if lens_T_finger_tag is None:
        continue
      out[kf] = rear_lens.body_T_lens.compose(lens_T_finger_tag)
    return out

  body_T_left_finger_per_kf = _per_kf_body_T_finger(args.left_finger_tag_id)
  body_T_right_finger_per_kf = _per_kf_body_T_finger(args.right_finger_tag_id)

  joint_angles: dict[str, list[float]] = {}
  for joint_name, finger_id in (
    ("left_finger", args.left_finger_tag_id),
    ("right_finger", args.right_finger_tag_id),
  ):
    if finger_id <= 0:
      continue
    joint_angles[joint_name] = _finger_joint_angles_per_kf(
      rear_tags_per_kf,
      rear_lens,
      base_tag_id=args.base_tag_id,
      finger_tag_id=finger_id,
      tag_size=args.gripper.tag_size,
      n_keyframes=n_keyframes,
      kf_rotations=rear_kf_rotations,
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
    anchor_obs_per_kf=anchor_obs_per_kf,
    anchor_local=anchor_local,
    body_T_base_tag=body_T_base_tag,
    joint_angles=joint_angles,
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
    body_T_left_finger_per_kf=body_T_left_finger_per_kf,
    body_T_right_finger_per_kf=body_T_right_finger_per_kf,
    left_finger_tag_id=args.left_finger_tag_id,
    right_finger_tag_id=args.right_finger_tag_id,
    n_keyframes=n_keyframes,
  )
