"""Composition of per-camera subgraphs into one joint VI-BA factor graph."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.vi_ba._mei_factor_cpp import (
  make_mei_base_tag_factor_cpp,
  make_mei_cross_finger_factor_cpp,
  make_mei_cross_tag_factor_cpp,
  make_mei_finger_factor_cpp,
  make_mei_world_point_factor_cpp,
)
from specialist.vi_ba.build import (
  LensChannel,
  LensSetup,
  add_mei_factor,
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
  tag_corners_in_frame,
  tag_corners_local,
  tag_half_angle,
)
from specialist.vi_ba.urdf_joints import FINGER_JOINTS

Float64Array = npt.NDArray[np.float64]

# One lens's surviving observations, ``{track_id: {keyframe: pixel}}``.
LandmarkTracks = dict[int, dict[int, Float64Array]]
# One lens's triangulated points, ``{track_id: world_point}``.
LandmarkPoints = dict[int, Float64Array]
# Per camera slot, the ``(front, rear)`` pair each lens map comes in.
LandmarkKeep = dict[str, tuple[LandmarkTracks, LandmarkTracks]]
__all__ = [
  "CameraKeys",
  "CrossSighting",
  "LandmarkKeep",
  "LandmarkPoints",
  "LandmarkTracks",
  "align_to_shared_world",
  "anchor_sightings",
  "anchored_keyframes",
  "assemble_joint_graph",
  "cross_sightings",
  "points_from_values",
]

BASE_TAG_CHAR = "G"
_POSE_CHAR = "p"
_VEL_CHAR = "v"
_BIAS_CHAR = "b"
_JOINT_ANGLE_CHAR = "a"
_SEC_CORNER_CHAR = "n"

# Finger corner σ (px) and the symmetry prior σ (rad) on θ_left + θ_right.
_FINGER_REPROJ_SIGMA = 1.5
_FINGER_SYMMETRY_SIGMA = 0.05
# Base-tag pose sigmas: corner reprojection (px) and loose PnP-seeded prior (rad, m).
_BASE_TAG_REPROJ_SIGMA = 1.5
_BASE_TAG_PRIOR_SIGMA_ROT = 0.3
_BASE_TAG_PRIOR_SIGMA_TRANS = 0.05
# Cross-camera corner σ (px).
_CROSS_REPROJ_SIGMA = 3.0

_MAX_KEYFRAMES_PER_CAMERA = 100_000
_MAX_LANDMARKS_PER_CAMERA = 1_000_000


class CameraKeys:
  """Collision-free GTSAM keys per camera; anchor corners are constants, keyless."""

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

  def secondary_corner_key(self, tag_id: int, corner: int) -> int:
    """Key for one freed secondary-anchor corner; tags own disjoint 4-slots."""
    return int(gtsam.symbol(_SEC_CORNER_CHAR, tag_id * 4 + corner))

  def base_tag_key(self, slot: str) -> int:
    """Key for the camera's static base-tag pose in its body frame."""
    return int(gtsam.symbol(BASE_TAG_CHAR, self._index[slot]))


def _bfs_world_alignments(
  anchor_locals_by_camera: dict[str, dict[int, gtsam.Pose3]],
  anchor_tag_ids: tuple[int, ...],
) -> tuple[dict[str, gtsam.Pose3], dict[int, gtsam.Pose3]]:
  """Camera alignments and anchor poses by BFS; raise on an unreachable camera."""
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
      align = world_T_anchor[via].compose(locals_[via].inverse())
      aligns[slot] = align
      for tag_id, local in locals_.items():
        if tag_id not in world_T_anchor:
          world_T_anchor[tag_id] = align.compose(local)
      aligned_now.append(slot)
    if not aligned_now:
      raise RuntimeError(
        f"cameras {pending} sight no anchor with an established world pose "
        f"(established anchors: {sorted(world_T_anchor)}); the camera-anchor "
        f"co-visibility graph is disconnected"
      )
    pending = [n for n in pending if n not in aligned_now]
  return aligns, world_T_anchor


def align_to_shared_world(
  cameras: list[PreparedCamera], anchor_tag_ids: tuple[int, ...]
) -> gtsam.Pose3:
  """Move every camera into the reference anchor's world; return its world pose."""
  locals_by_camera = {t.name: t.anchor_locals_by_tag for t in cameras}
  aligns, world_T_anchor = _bfs_world_alignments(locals_by_camera, anchor_tag_ids)
  for t in cameras:
    align = aligns[t.name]
    align_R = align.rotation().matrix()
    align_t = np.asarray(align.translation(), dtype=np.float64)
    t.body_poses = [align.compose(p) for p in t.body_poses]
    t.velocities = [align_R @ v for v in t.velocities]
    for ch in t.lenses:
      ch.points = {tid: align_R @ p + align_t for tid, p in ch.points.items()}
  return world_T_anchor[anchor_tag_ids[0]]


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
  base_tag_ids: list[int],
  base_tag_size: float,
) -> list[CrossSighting]:
  """Every other-camera sighting of a PnP-passing base tag or a modeled finger tag."""
  sightings: list[CrossSighting] = []
  for owner_idx, owner in enumerate(cameras):
    # The base tag, plus each finger tag with URDF joint kinematics.
    targets: dict[int, int | None] = {base_tag_ids[owner_idx]: None}
    finger_ids = (owner.left_finger_tag_id, owner.right_finger_tag_id)
    for idx, (name, finger_id) in enumerate(
      zip(FINGER_JOINTS, finger_ids, strict=True)
    ):
      if finger_id > 0 and name in owner.finger_reproj:
        targets[finger_id] = idx
    for observer in cameras:
      if observer is owner:
        continue
      n = min(observer.n_keyframes, owner.n_keyframes)
      for ch in observer.lenses:
        for kf in range(n):
          for det in ch.tags[kf]:
            if det.tag_id not in targets:
              continue
            finger = targets[det.tag_id]
            if finger is None and (
              estimate_lens_T_tag(det.corners, base_tag_size, ch.setup.params) is None
            ):
              continue  # base-tag PnP gate: drop degenerate / behind-camera
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


def anchor_sightings(
  camera: PreparedCamera,
  anchor_tag_id: int,
  max_half_angle: float,
) -> list[tuple[LensChannel, int, TagDetection]]:
  """Per-keyframe anchor-tag detections in both lenses, within ``max_half_angle``."""
  return [
    (ch, kf, det)
    for ch in camera.lenses
    for kf, dets in enumerate(ch.tags)
    for det in dets
    if det.tag_id == anchor_tag_id
    and tag_half_angle(det, ch.setup.params) <= max_half_angle
  ]


def _sighting_bearing(
  obs: dict[int, dict[str, gtsam.Pose3]],
  kf: int,
  side: str,
  det: TagDetection,
  lens: LensSetup,
) -> Float64Array:
  """Cheirality bearing: the PnP tag position, else the detection-center ray."""
  lens_T_tag = obs.get(kf, {}).get(side)
  if lens_T_tag is None:
    return tag_center_bearing(det, lens.params)
  bearing = np.asarray(lens_T_tag.translation(), dtype=np.float64)
  return bearing / np.linalg.norm(bearing)


def anchored_keyframes(camera: PreparedCamera) -> set[int]:
  """Keyframes with a PnP sighting of any configured anchor."""
  return set().union(*camera.anchor_obs_by_tag.values())


def _add_corner_factors_for_anchor(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  corner_noise: gtsam.noiseModel.Base,
  tag_id: int,
  corners_world: Float64Array,
  pnp_bearings: bool,
  corner_keys: list[int] | None,
  max_half_angle: float,
  graph: gtsam.NonlinearFactorGraph,
) -> int:
  """Add corner factors for sightings within ``max_half_angle``; return their count."""
  n_sightings = 0
  for t in cameras:
    obs = t.anchor_obs_by_tag.get(tag_id, {}) if pnp_bearings else {}
    for ch, kf, det in anchor_sightings(t, tag_id, max_half_angle):
      n_sightings += 1
      pose_key = keys.pose_key(t.name, kf)
      bearing = _sighting_bearing(obs, kf, ch.lens, det, ch.setup)
      for j, pixel in enumerate(det.corners):
        if corner_keys is not None:
          add_mei_factor(
            graph, pose_key, corner_keys[j], pixel, corner_noise, ch.setup, bearing
          )
        else:
          add_mei_tag_factor(
            graph,
            make_mei_world_point_factor_cpp,
            (pose_key,),
            pixel,
            corner_noise,
            ch.setup,
            corners_world[j],
            bearing,
          )
  return n_sightings


def _add_all_anchor_corner_factors(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  args: Args,
  anchor_corners_world: Float64Array,
  secondary_anchor_world: dict[int, gtsam.Pose3],
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
) -> None:
  """Add corner factors for the reference anchor and each seeded secondary anchor."""
  corner_noise = huber_pixel_noise(args.anchor.corner_reproj_sigma)
  n_ref = _add_corner_factors_for_anchor(
    cameras,
    keys,
    corner_noise,
    args.anchor.tag_ids[0],
    anchor_corners_world,
    pnp_bearings=True,
    corner_keys=None,
    max_half_angle=args.anchor.max_half_angle,
    graph=graph,
  )
  print(f"anchor corner reprojection: {n_ref} anchor sightings")
  sec_prior = gtsam.noiseModel.Isotropic.Sigma(
    3, args.anchor.secondary_corner_prior_sigma
  )
  for tag_id, world_T_tag in secondary_anchor_world.items():
    seed_corners = tag_corners_in_frame(world_T_tag, args.anchor.tag_size)
    corner_keys: list[int] | None = None
    if args.anchor.free_secondary:
      # Secondary corners are graph variables held near the seed by a loose prior.
      corner_keys = [keys.secondary_corner_key(tag_id, j) for j in range(4)]
      for key, xyz in zip(corner_keys, seed_corners, strict=True):
        values.insert(key, xyz)
        graph.add(gtsam.PriorFactorPoint3(key, xyz, sec_prior))
    n_sec = _add_corner_factors_for_anchor(
      cameras,
      keys,
      corner_noise,
      tag_id,
      seed_corners,
      pnp_bearings=False,
      corner_keys=corner_keys,
      max_half_angle=args.anchor.max_half_angle,
      graph=graph,
    )
    print(f"secondary anchor {tag_id} corner reprojection: {n_sec} sightings")


def _add_gripper_factors(
  cameras: list[PreparedCamera],
  keys: CameraKeys,
  args: Args,
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
) -> tuple[int, int]:
  """Add each camera's base-tag pose, its base and finger corner factors and
  the finger symmetry priors; return the base and finger factor counts."""
  finger_noise = huber_pixel_noise(_FINGER_REPROJ_SIGMA)
  base_tag_noise = huber_pixel_noise(_BASE_TAG_REPROJ_SIGMA)
  base_tag_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([_BASE_TAG_PRIOR_SIGMA_ROT] * 3 + [_BASE_TAG_PRIOR_SIGMA_TRANS] * 3)
  )
  sym_noise = gtsam.noiseModel.Isotropic.Sigma(1, _FINGER_SYMMETRY_SIGMA)
  base_tag_corners = tag_corners_local(args.gripper.tag_size)
  finger_factor_count = 0
  base_tag_factor_count = 0
  for t, base_tag_id in zip(cameras, args.base_tag_ids, strict=True):
    reprojs = {
      idx: reproj
      for idx, joint in enumerate(FINGER_JOINTS)
      if (reproj := t.finger_reproj.get(joint)) is not None
    }
    if t.body_T_base_tag is None or (not reprojs and base_tag_id <= 0):
      continue
    bt_key = keys.base_tag_key(t.name)
    values.insert(bt_key, t.body_T_base_tag)
    graph.add(gtsam.PriorFactorPose3(bt_key, t.body_T_base_tag, base_tag_prior_noise))
    # Base-tag corners reproject through body_T_base_tag into this camera's rear lens.
    if base_tag_id > 0:
      for dets in t.rear.tags:
        det = next((d for d in dets if d.tag_id == base_tag_id), None)
        if det is None:
          continue
        bearing = tag_center_bearing(det, t.rear.setup.params)
        for corner, pixel in zip(base_tag_corners, det.corners, strict=True):
          add_mei_tag_factor(
            graph,
            make_mei_base_tag_factor_cpp,
            (bt_key,),
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
        for corner, pixel in zip(reproj.joint_t_corners, obs.pixels, strict=True):
          add_mei_tag_factor(
            graph,
            make_mei_finger_factor_cpp,
            (tkey, bt_key),
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
  """Add cross-camera base and finger factors where the owner's variables exist."""
  noise = huber_pixel_noise(_CROSS_REPROJ_SIGMA)
  base_tag_corners = tag_corners_local(args.gripper.tag_size)
  camera_by_slot = {t.name: t for t in cameras}
  base_count = finger_count = 0
  for sighting in cross_obs:
    if not values.exists(keys.base_tag_key(sighting.owner)):
      continue
    obs_key = keys.pose_key(sighting.observer, sighting.keyframe)
    owner_key = keys.pose_key(sighting.owner, sighting.keyframe)
    owner_bt_key = keys.base_tag_key(sighting.owner)
    if sighting.finger is None:
      for corner, pixel in zip(base_tag_corners, sighting.corners, strict=True):
        add_mei_tag_factor(
          graph,
          make_mei_cross_tag_factor_cpp,
          (obs_key, owner_key, owner_bt_key),
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
          make_mei_cross_finger_factor_cpp,
          (obs_key, owner_key, owner_bt_key, theta_key),
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
  anchor_corners_world: Float64Array,
  secondary_anchor_world: dict[int, gtsam.Pose3],
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """Build the joint factor graph over every camera and anchor."""
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
  _add_all_anchor_corner_factors(
    cameras, keys, args, anchor_corners_world, secondary_anchor_world, graph, values
  )
  base_tags, fingers = _add_gripper_factors(cameras, keys, args, graph, values)
  cross_base, cross_finger = _add_cross_camera_factors(
    cameras, keys, args, cross_obs, graph, values
  )
  poses_smoothed, velocities_smoothed = _add_smoothness_priors(cameras, keys, graph)
  print(
    f"reprojection factors: base-tag {base_tags}, finger {fingers}, "
    f"cross base {cross_base}, cross finger {cross_finger}; smoothness priors: "
    f"pose {poses_smoothed}, velocity {velocities_smoothed}"
  )
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
