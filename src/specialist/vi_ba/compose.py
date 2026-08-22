"""Composition of per-camera subgraphs into one joint VI-BA factor graph."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial

import gtsam
import numpy as np
import numpy.typing as npt
from specialist_camera import MeiParams
from specialist_dataset.fiducials import tag_corners_local

from specialist.vi_ba._mei_factor_cpp import (
  make_mei_anchor_tag_factor_cpp,
  make_mei_seated_base_tag_factor_cpp,
  make_mei_seated_cross_finger_factor_cpp,
  make_mei_seated_cross_tag_factor_cpp,
  make_mei_seated_finger_factor_cpp,
)
from specialist.vi_ba.build import (
  LensChannel,
  LensSetup,
  add_mei_tag_factor,
  build_vi_graph,
  huber_pixel_noise,
)
from specialist.vi_ba.configs import Args
from specialist.vi_ba.single_camera import PreparedCamera
from specialist.vi_ba.symbols import LANDMARK_CHAR
from specialist.vi_ba.tag_detection import (
  TagDetection,
  estimate_lens_T_tag,
  tag_center_bearing,
)
from specialist.vi_ba.urdf_joints import FINGER_JOINTS

Float64Array = npt.NDArray[np.float64]

# One lens's surviving observations, ``{track_id: {keyframe: pixel}}``.
LandmarkTracks = dict[int, dict[int, Float64Array]]
# Per camera slot, the ``(front, rear)`` pair each lens map comes in.
LandmarkKeep = dict[str, tuple[LandmarkTracks, LandmarkTracks]]
__all__ = [
  "CameraKeys",
  "CrossSighting",
  "LandmarkKeep",
  "LandmarkTracks",
  "align_to_shared_world",
  "anchor_sightings",
  "anchored_keyframes",
  "assemble_joint_graph",
  "cross_sightings",
  "points_from_values",
  "seat_evidence",
]

BASE_TAG_CHAR = "G"
_POSE_CHAR = "p"
_VEL_CHAR = "v"
_BIAS_CHAR = "b"
_JOINT_ANGLE_CHAR = "a"
SEAT_CHAR = "E"
ANCHOR_CHAR = "A"

# Finger corner σ (px) and the symmetry prior σ (rad) on θ_left + θ_right.
_FINGER_REPROJ_SIGMA = 1.5
_FINGER_SYMMETRY_SIGMA = 0.05
# Base-tag corner σ (px) and its PnP-seeded pose prior σ (rad, m).
_BASE_TAG_REPROJ_SIGMA = 1.5
_BASE_TAG_PRIOR_SIGMA_ROT = 1.0
_BASE_TAG_PRIOR_SIGMA_TRANS = 0.3
# Cross-camera corner σ (px).
_CROSS_REPROJ_SIGMA = 3.0
# Corner σ growth (px) per unit fractional overshoot of the calibrated radius.
_SIGMA_PER_RADIUS = 40.0
# Gauge prior σ on the reference anchor's position and yaw.
_GAUGE_PRIOR_SIGMA = 1e-4
# Loose σ on the reference anchor's tilt, which gravity makes observable.
_TILT_FREE_SIGMA = 1e3
# Loose rear-seat rotation prior σ (rad): factory seats err by several degrees.
_SEAT_PRIOR_SIGMA_ROT = 0.2

_MAX_KEYFRAMES_PER_CAMERA = 100_000
_MAX_LANDMARKS_PER_CAMERA = 1_000_000


class CameraKeys:
  """Collision-free GTSAM keys per camera; anchor keys use the tag id alone."""

  def __init__(self, slots: Sequence[str]) -> None:
    if len(set(slots)) != len(slots):
      raise ValueError(f"camera slots must be unique: {list(slots)}")
    self._index = {slot: i for i, slot in enumerate(slots)}

  def _kf_key(self, char: str, slot: str, kf: int) -> int:
    """Collision-free key for a per-keyframe variable in one camera's slice."""
    return int(gtsam.symbol(char, self._index[slot] * _MAX_KEYFRAMES_PER_CAMERA + kf))

  def pose_key(self, slot: str, kf: int) -> int:
    return self._kf_key(_POSE_CHAR, slot, kf)

  def velocity_key(self, slot: str, kf: int) -> int:
    return self._kf_key(_VEL_CHAR, slot, kf)

  def bias_key(self, slot: str, kf: int) -> int:
    return self._kf_key(_BIAS_CHAR, slot, kf)

  def _lm_key(self, slot: str, half: int, track_id: int) -> int:
    """Landmark key in a front or rear slot; raises ValueError on slot overflow."""
    if not 0 <= track_id < _MAX_LANDMARKS_PER_CAMERA:
      raise ValueError(
        f"track id {track_id} does not fit the {_MAX_LANDMARKS_PER_CAMERA}-wide "
        f"landmark slot for {slot!r}; its key would collide with another slot"
      )
    return self._slice_key(
      LANDMARK_CHAR, slot, half, _MAX_LANDMARKS_PER_CAMERA, track_id
    )

  def front_landmark_key(self, slot: str, track_id: int) -> int:
    return self._lm_key(slot, 0, track_id)

  def rear_landmark_key(self, slot: str, track_id: int) -> int:
    return self._lm_key(slot, 1, track_id)

  def theta_key(self, slot: str, finger_idx: int, kf: int) -> int:
    """Key for a finger joint angle at ``kf``; ``finger_idx`` 0 is left, 1 is right."""
    return self._slice_key(
      _JOINT_ANGLE_CHAR, slot, finger_idx, _MAX_KEYFRAMES_PER_CAMERA, kf
    )

  def _slice_key(self, char: str, slot: str, half: int, width: int, index: int) -> int:
    """One integer key in the camera's ``half`` of two ``width``-wide slices."""
    return int(gtsam.symbol(char, (self._index[slot] * 2 + half) * width + index))

  def anchor_key(self, tag_id: int) -> int:
    """Key for a wall tag's world pose, shared by every camera that sights it."""
    return int(gtsam.symbol(ANCHOR_CHAR, tag_id))

  def base_tag_key(self, slot: str) -> int:
    """Key for the camera's static base-tag pose in its body frame."""
    return int(gtsam.symbol(BASE_TAG_CHAR, self._index[slot]))

  def seat_key(self, slot: str) -> int:
    """Key for the camera's static rear-lens seat rotation."""
    return int(gtsam.symbol(SEAT_CHAR, self._index[slot]))


def _bfs_world_alignments(
  anchor_locals_by_camera: dict[str, dict[int, gtsam.Pose3]],
  sightings_by_camera: dict[str, dict[int, int]],
  anchor_tag_ids: tuple[int, ...],
) -> tuple[dict[str, gtsam.Pose3], dict[int, gtsam.Pose3]]:
  """Each camera's world alignment and anchor world poses by BFS over co-visibility
  from the first anchor; raise when a camera cannot reach it."""
  ref_tag_id = anchor_tag_ids[0]
  ref = next(
    (
      locals_[ref_tag_id]
      for locals_ in anchor_locals_by_camera.values()
      if ref_tag_id in locals_
    ),
    None,
  )
  if ref is None:
    raise RuntimeError("no camera observes the reference anchor; cannot share a world")
  world_T_anchor: dict[int, gtsam.Pose3] = {ref_tag_id: ref}
  aligns: dict[str, gtsam.Pose3] = {}
  pending = list(anchor_locals_by_camera)
  while pending:
    aligned_now: list[str] = []
    for slot in pending:
      locals_ = anchor_locals_by_camera[slot]
      via = next(
        (a for a in anchor_tag_ids if a in world_T_anchor and a in locals_), None
      )
      if via is None:
        continue
      aligns[slot] = world_T_anchor[via].compose(locals_[via].inverse())
      aligned_now.append(slot)
    if not aligned_now:
      raise RuntimeError(
        f"cameras {pending} sight no anchor with an established world pose "
        f"(established anchors: {sorted(world_T_anchor)}); the camera-anchor "
        f"co-visibility graph is disconnected"
      )
    for tag_id in {t for n in aligned_now for t in anchor_locals_by_camera[n]}:
      if tag_id in world_T_anchor:
        continue
      best = max(
        (n for n in aligned_now if tag_id in anchor_locals_by_camera[n]),
        key=lambda n: sightings_by_camera[n].get(tag_id, 0),
      )
      world_T_anchor[tag_id] = aligns[best].compose(
        anchor_locals_by_camera[best][tag_id]
      )
    pending = [n for n in pending if n not in aligned_now]
  return aligns, world_T_anchor


def align_to_shared_world(
  cameras: list[PreparedCamera], anchor_tag_ids: tuple[int, ...]
) -> dict[int, gtsam.Pose3]:
  """Move every camera into the world whose origin and x heading are the reference
  anchor's, returning each sighted anchor's world pose."""
  locals_by_camera = {t.name: t.anchor_locals_by_tag for t in cameras}
  sightings_by_camera = {
    t.name: {tag_id: len(obs) for tag_id, obs in t.anchor_obs_by_tag.items()}
    for t in cameras
  }
  aligns, world_T_anchor = _bfs_world_alignments(
    locals_by_camera, sightings_by_camera, anchor_tag_ids
  )
  ref = world_T_anchor[anchor_tag_ids[0]]
  x_axis = ref.rotation().matrix()[:, 0]
  yaw = gtsam.Rot3.Rz(-float(np.arctan2(x_axis[1], x_axis[0])))
  canon = gtsam.Pose3(yaw, yaw.rotate(gtsam.Point3(*-np.asarray(ref.translation()))))
  aligns = {slot: canon.compose(align) for slot, align in aligns.items()}
  world_T_anchor = {
    tag_id: canon.compose(pose) for tag_id, pose in world_T_anchor.items()
  }
  for t in cameras:
    align = aligns[t.name]
    align_R = align.rotation().matrix()
    align_t = np.asarray(align.translation(), dtype=np.float64)
    t.body_poses = [align.compose(p) for p in t.body_poses]
    t.velocities = [align_R @ v for v in t.velocities]
    for ch in t.lenses:
      ch.points = {tid: align_R @ p + align_t for tid, p in ch.points.items()}
  return world_T_anchor


@dataclass(frozen=True)
class CrossSighting:
  """One observer-lens sighting of another camera's tag, for a cross factor."""

  observer: str
  owner: str
  keyframe: int
  setup: LensSetup
  corners: Float64Array  # (4, 2) detected corner pixels
  bearing: Float64Array  # (3,) lens-frame tag direction
  finger: int | None  # None = base tag; 0 = left finger, 1 = right finger


def cross_sightings(
  cameras: list[PreparedCamera],
  base_tag_size: float,
) -> list[CrossSighting]:
  """Every sighting of a camera's base or finger tag in another camera's rear lens;
  raise when two keyframe grids disagree."""
  sightings: list[CrossSighting] = []
  for owner in cameras:
    # The base tag, plus each finger tag with URDF joint kinematics.
    targets: dict[int, int | None] = {owner.tags.base: None}
    fingers = zip(FINGER_JOINTS, owner.tags.fingers, strict=True)
    for idx, (name, finger_id) in enumerate(fingers):
      if finger_id is not None and name in owner.finger_reproj:
        targets[finger_id] = idx
    for observer in cameras:
      if observer is owner:
        continue
      n = min(observer.n_keyframes, owner.n_keyframes)
      ch = observer.rear
      for kf in range(n):
        for det in ch.tags[kf]:
          if det.tag_id not in targets:
            continue
          finger = targets[det.tag_id]
          if finger is None and (
            estimate_lens_T_tag(det.corners, base_tag_size, ch.setup.params) is None
          ):
            continue  # base-tag PnP gate: drop degenerate / behind-lens
          sightings.append(
            CrossSighting(
              observer=observer.name,
              owner=owner.name,
              keyframe=kf,
              setup=ch.setup,
              corners=det.corners,
              bearing=tag_center_bearing(det, ch.setup.params),
              finger=finger,
            )
          )
  return sightings


def _finger_symmetry_factor(
  key_l: int, key_r: int, noise: gtsam.noiseModel.Base
) -> gtsam.LinearContainerFactor:
  """Soft URDF-mimic prior on θ_left + θ_right, exact as a linear container factor."""
  jacobian = gtsam.JacobianFactor(
    key_l, np.array([[1.0]]), key_r, np.array([[1.0]]), np.zeros(1), noise
  )
  lin_point = gtsam.Values()
  lin_point.insert(key_l, 0.0)
  lin_point.insert(key_r, 0.0)
  return gtsam.LinearContainerFactor(jacobian, lin_point)


def corner_sigma(
  max_radius: float, calibrated_radius: float, base_sigma: float
) -> float:
  """Corner σ (px): ``base_sigma`` plus overshoot growth, added in quadrature."""
  over = max(0.0, max_radius / calibrated_radius - 1.0)
  return float(np.hypot(base_sigma, _SIGMA_PER_RADIUS * over))


def sighting_max_radius(corners: Float64Array, params: MeiParams) -> float:
  """Pixel distance of the furthest corner from the principal point."""
  pixels = np.asarray(corners, dtype=np.float64)
  return float(np.hypot(pixels[:, 0] - params.cx, pixels[:, 1] - params.cy).max())


def corner_noise(
  corners: Float64Array, setup: LensSetup, base_sigma: float
) -> gtsam.noiseModel.Base:
  """Robust noise for one sighting's corner factors, priced by its corner radius."""
  return huber_pixel_noise(
    corner_sigma(
      sighting_max_radius(corners, setup.params), setup.calibrated_radius, base_sigma
    )
  )


def anchor_gauge_prior(
  anchor_key: int, world_T_anchor: gtsam.Pose3
) -> gtsam.PriorFactorPose3:
  """Gauge prior on the reference anchor's position and yaw, with tilt free."""
  rotation = world_T_anchor.rotation().matrix()
  up = np.array([0.0, 0.0, 1.0])
  free = np.eye(3) - np.outer(up, up)
  world_cov = _TILT_FREE_SIGMA**2 * free + _GAUGE_PRIOR_SIGMA**2 * np.outer(up, up)
  cov = np.zeros((6, 6))
  cov[:3, :3] = rotation.T @ world_cov @ rotation
  cov[3:, 3:] = _GAUGE_PRIOR_SIGMA**2 * np.eye(3)
  return gtsam.PriorFactorPose3(
    anchor_key, world_T_anchor, gtsam.noiseModel.Gaussian.Covariance(cov)
  )


def anchor_sightings(
  camera: PreparedCamera,
  anchor_tag_id: int,
) -> list[tuple[LensChannel, int, TagDetection]]:
  """Per-keyframe anchor-tag detections in both lenses."""
  return [
    (ch, kf, det)
    for ch in camera.lenses
    for kf, dets in enumerate(ch.tags)
    for det in dets
    if det.tag_id == anchor_tag_id
  ]


def anchored_keyframes(camera: PreparedCamera) -> set[int]:
  """Keyframes with a PnP sighting of any configured anchor."""
  return set().union(*camera.anchor_obs_by_tag.values())


def _add_corner_factors_for_anchor(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  base_sigma: float,
  tag_id: int,
  tag_size: float,
  graph: gtsam.NonlinearFactorGraph,
) -> int:
  """Add one anchor's corner factors from every sighting; return the sighting count."""
  corners = tag_corners_local(tag_size)
  anchor_key = keys.anchor_key(tag_id)
  n_sightings = 0
  for t in cameras:
    for ch, kf, det in anchor_sightings(t, tag_id):
      n_sightings += 1
      pose_key = keys.pose_key(t.name, kf)
      bearing = tag_center_bearing(det, ch.setup.params)
      sighting_noise = corner_noise(det.corners, ch.setup, base_sigma)
      seat_key = keys.seat_key(t.name) if ch.lens == "rear" else None
      for corner, pixel in zip(corners, det.corners, strict=True):
        add_mei_tag_factor(
          graph,
          make_mei_anchor_tag_factor_cpp,
          (pose_key, anchor_key),
          pixel,
          sighting_noise,
          ch.setup,
          corner,
          bearing,
          seat_key=seat_key,
        )
  return n_sightings


def _add_anchor_factors(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  args: Args,
  world_T_anchor: dict[int, gtsam.Pose3],
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
) -> None:
  """Add each anchor's pose and corner factors and the gauge prior on the reference."""
  base_sigma = args.anchor.corner_reproj_sigma
  reference = args.anchor.tag_ids[0]
  for tag_id, world_T_tag in world_T_anchor.items():
    key = keys.anchor_key(tag_id)
    values.insert(key, world_T_tag)
    n = _add_corner_factors_for_anchor(
      cameras, keys, base_sigma, tag_id, args.anchor.tag_size, graph
    )
    print(f"[graph] anchor {tag_id} corner reprojection: {n} sightings")
  for tag_id in args.anchor.tag_ids:
    if tag_id in world_T_anchor:
      continue
    rear = sum(
      1
      for camera in cameras
      for ch, _kf, _det in anchor_sightings(camera, tag_id)
      if ch.lens == "rear"
    )
    print(
      f"  [warn] [graph] anchor {tag_id} has no world pose and carries no "
      f"factors: prep resolves an anchor in the front lens alone, and this one "
      f"was sighted there by no camera ({rear} rear sightings unused)"
    )
  graph.add(anchor_gauge_prior(keys.anchor_key(reference), world_T_anchor[reference]))


def _add_seat_variables(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
) -> None:
  """One rear-seat rotation variable per camera, with a prior at its factory value."""
  noise = gtsam.noiseModel.Isotropic.Sigma(3, _SEAT_PRIOR_SIGMA_ROT)
  for t in cameras:
    key = keys.seat_key(t.name)
    rot = t.rear.setup.body_T_lens.rotation()
    values.insert(key, rot)
    graph.add(gtsam.PriorFactorRot3(key, rot, noise))


def seat_evidence(
  camera: PreparedCamera,
  solved_anchor_ids: Iterable[int],
  cross_obs: list[CrossSighting],
) -> tuple[int, int]:
  """``(anchor, cross)`` counts of the rear-lens sightings that measure its seat."""
  anchors = sum(
    1
    for tag_id in solved_anchor_ids
    for ch, _kf, _det in anchor_sightings(camera, tag_id)
    if ch.lens == "rear"
  )
  cross = sum(1 for s in cross_obs if s.observer == camera.name)
  return anchors, cross


def _add_gripper_factors(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  args: Args,
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
) -> tuple[int, int]:
  """Add each camera's base-tag pose, its seated base and finger corner factors and
  the finger symmetry priors; return the base and finger factor counts."""
  base_tag_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([_BASE_TAG_PRIOR_SIGMA_ROT] * 3 + [_BASE_TAG_PRIOR_SIGMA_TRANS] * 3)
  )
  sym_noise = gtsam.noiseModel.Isotropic.Sigma(1, _FINGER_SYMMETRY_SIGMA)
  base_tag_corners = tag_corners_local(args.gripper.tag_size)
  finger_factor_count = 0
  base_tag_factor_count = 0
  for t in cameras:
    reprojs = {
      idx: reproj
      for idx, joint in enumerate(FINGER_JOINTS)
      if (reproj := t.finger_reproj.get(joint)) is not None
    }
    bt_key = keys.base_tag_key(t.name)
    values.insert(bt_key, t.body_T_base_tag)
    graph.add(gtsam.PriorFactorPose3(bt_key, t.body_T_base_tag, base_tag_prior_noise))
    for dets in t.rear.tags:
      det = next((d for d in dets if d.tag_id == t.tags.base), None)
      if det is None:
        continue
      bearing = tag_center_bearing(det, t.rear.setup.params)
      base_tag_noise = corner_noise(det.corners, t.rear.setup, _BASE_TAG_REPROJ_SIGMA)
      for corner, pixel in zip(base_tag_corners, det.corners, strict=True):
        add_mei_tag_factor(
          graph,
          make_mei_seated_base_tag_factor_cpp,
          (bt_key, keys.seat_key(t.name)),
          pixel,
          base_tag_noise,
          t.rear.setup,
          corner,
          bearing,
        )
      base_tag_factor_count += 4
    both_fingers = len(reprojs) == 2
    observed_kf = sorted({kf for r in reprojs.values() for kf in r.observations})
    for kf in observed_kf:
      seen_seed: dict[int, float] = {}
      for finger_idx, reproj in reprojs.items():
        obs = reproj.observations.get(kf)
        if obs is None:
          continue
        tkey = keys.theta_key(t.name, finger_idx, kf)
        values.insert(tkey, float(obs.theta_seed))
        finger_noise = corner_noise(obs.pixels, t.rear.setup, _FINGER_REPROJ_SIGMA)
        for corner, pixel in zip(reproj.joint_t_corners, obs.pixels, strict=True):
          add_mei_tag_factor(
            graph,
            make_mei_seated_finger_factor_cpp,
            (tkey, bt_key, keys.seat_key(t.name)),
            pixel,
            finger_noise,
            t.rear.setup,
            corner,
            obs.obs_bearing,
            reproj.base_T_joint,
          )
        finger_factor_count += 4
        seen_seed[finger_idx] = float(obs.theta_seed)
      if both_fingers:
        for finger_idx in (0, 1):
          if finger_idx not in seen_seed:  # unseen finger: inferred from the prior only
            values.insert(
              keys.theta_key(t.name, finger_idx, kf), -next(iter(seen_seed.values()))
            )
        graph.add(
          _finger_symmetry_factor(
            keys.theta_key(t.name, 0, kf), keys.theta_key(t.name, 1, kf), sym_noise
          )
        )
  return base_tag_factor_count, finger_factor_count


def _add_cross_camera_factors(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  args: Args,
  cross_obs: list[CrossSighting],
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
) -> tuple[int, int]:
  """Add each cross sighting's corner factors on the owner variables that
  ``_add_gripper_factors`` inserted; return the base and finger factor counts."""
  base_tag_corners = tag_corners_local(args.gripper.tag_size)
  camera_by_slot = {t.name: t for t in cameras}
  base_count = finger_count = 0
  for sighting in cross_obs:
    obs_key = keys.pose_key(sighting.observer, sighting.keyframe)
    owner_key = keys.pose_key(sighting.owner, sighting.keyframe)
    owner_bt_key = keys.base_tag_key(sighting.owner)
    seat_key = keys.seat_key(sighting.observer)
    noise = corner_noise(sighting.corners, sighting.setup, _CROSS_REPROJ_SIGMA)
    if sighting.finger is None:
      for corner, pixel in zip(base_tag_corners, sighting.corners, strict=True):
        add_mei_tag_factor(
          graph,
          make_mei_seated_cross_tag_factor_cpp,
          (obs_key, owner_key, owner_bt_key, seat_key),
          pixel,
          noise,
          sighting.setup,
          corner,
          sighting.bearing,
        )
      base_count += 4
    elif values.exists(
      theta_key := keys.theta_key(sighting.owner, sighting.finger, sighting.keyframe)
    ):
      finger_name = FINGER_JOINTS[sighting.finger]
      reproj = camera_by_slot[sighting.owner].finger_reproj[finger_name]
      for corner, pixel in zip(reproj.joint_t_corners, sighting.corners, strict=True):
        add_mei_tag_factor(
          graph,
          make_mei_seated_cross_finger_factor_cpp,
          (obs_key, owner_key, owner_bt_key, theta_key, seat_key),
          pixel,
          noise,
          sighting.setup,
          corner,
          sighting.bearing,
          reproj.base_T_joint,
        )
      finger_count += 4
  return base_count, finger_count


def _add_smoothness_priors(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  graph: gtsam.NonlinearFactorGraph,
) -> tuple[int, int]:
  """Add pose priors (prep rotation, zero translation) and velocity priors between
  unanchored keyframe pairs; return both counts."""
  smooth_sigmas = np.array([np.deg2rad(2.0)] * 3 + [0.10] * 3, dtype=np.float64)
  smooth_noise = gtsam.noiseModel.Diagonal.Sigmas(smooth_sigmas)
  vel_smooth_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.3, 0.3, 0.3], dtype=np.float64)
  )
  smooth_count = 0
  vel_smooth_count = 0
  zero_t = gtsam.Point3(0.0, 0.0, 0.0)
  for t in cameras:
    anchored_kfs = anchored_keyframes(t)
    for k in range(1, t.n_keyframes):
      # A pair with an anchored end would pull the free pose toward the pinned one.
      if (k - 1) in anchored_kfs or k in anchored_kfs:
        continue
      # The prep poses carry the IMU-propagated relative rotation.
      prep_rel = t.body_poses[k - 1].inverse().compose(t.body_poses[k])
      smooth_mean = gtsam.Pose3(prep_rel.rotation(), zero_t)
      graph.add(
        gtsam.BetweenFactorPose3(
          keys.pose_key(t.name, k - 1),
          keys.pose_key(t.name, k),
          smooth_mean,
          smooth_noise,
        )
      )
      smooth_count += 1
      vel_delta = np.asarray(t.velocities[k] - t.velocities[k - 1], dtype=np.float64)
      graph.add(
        gtsam.BetweenFactorVector(
          keys.velocity_key(t.name, k - 1),
          keys.velocity_key(t.name, k),
          vel_delta,
          vel_smooth_noise,
        )
      )
      vel_smooth_count += 1
  return smooth_count, vel_smooth_count


def assemble_joint_graph(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  args: Args,
  cross_obs: list[CrossSighting],
  world_T_anchor: dict[int, gtsam.Pose3],
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """Build the joint factor graph and initial values over every camera and anchor."""
  graph = gtsam.NonlinearFactorGraph()
  values = gtsam.Values()
  for t in cameras:
    build_vi_graph(
      body_poses=t.body_poses,
      velocities=t.velocities,
      biases=t.biases,
      preintegrated=t.preintegrated,
      front=t.front,
      rear=t.rear,
      sigma=args.frontend.reproj_sigma,
      velocity_prior_sigma=args.solver.velocity_prior_sigma,
      graph=graph,
      values=values,
      pose_key_fn=partial(keys.pose_key, t.name),
      velocity_key_fn=partial(keys.velocity_key, t.name),
      bias_key_fn=partial(keys.bias_key, t.name),
      front_landmark_key_fn=partial(keys.front_landmark_key, t.name),
      rear_landmark_key_fn=partial(keys.rear_landmark_key, t.name),
    )
  _add_seat_variables(cameras, keys, graph, values)
  _add_anchor_factors(cameras, keys, args, world_T_anchor, graph, values)
  base_tags, fingers = _add_gripper_factors(cameras, keys, args, graph, values)
  cross_base, cross_finger = _add_cross_camera_factors(
    cameras, keys, args, cross_obs, graph, values
  )
  poses_smoothed, velocities_smoothed = _add_smoothness_priors(cameras, keys, graph)
  print(
    f"[graph] reprojection factors: base-tag {base_tags}, finger {fingers}, "
    f"cross base {cross_base}, cross finger {cross_finger}; smoothness priors: "
    f"pose {poses_smoothed}, velocity {velocities_smoothed}"
  )
  for t in cameras:
    anchors, cross = seat_evidence(t, world_T_anchor, cross_obs)
    print(f"[graph] {t.name} rear-seat evidence: {anchors} anchor, {cross} cross")
  return graph, values


def points_from_values(
  values: gtsam.Values,
  key_fn: Callable[[str, int], int],
  slot: str,
  tids: Iterable[int],
) -> dict[int, Float64Array]:
  """Pull the solved 3-D landmark points out of ``values`` for ``tids``."""
  return {
    tid: np.asarray(values.atPoint3(key_fn(slot, tid)), dtype=np.float64)
    for tid in tids
  }
