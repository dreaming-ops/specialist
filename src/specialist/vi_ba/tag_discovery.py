"""Find a camera's base and finger tags from rear-lens corners and URDF finger arcs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt
from specialist_camera import MeiParams

from specialist.vi_ba.geometry import translation
from specialist.vi_ba.tag_detection import (
  TagDetection,
  estimate_lens_T_tag,
)
from specialist.vi_ba.urdf_joints import FINGER_JOINTS, RevoluteJoint

__all__ = [
  "GripperTags",
  "TagDiscoveryError",
  "discover_gripper_tags",
  "discovery_samples",
  "uniform_picks",
  "validate_discovered",
]

_MIN_PAIR_SIGHTINGS = 5
_MAX_BASE_CORNER_SPREAD = 8.0
_MAX_FINGER_POS = 0.015
_MAX_FINGER_ROT = np.radians(25.0)

_ARC_SAMPLES = 181
PICK_COUNT = 40
# Fraction of range(n) cut from each end before uniform_picks spreads the picks.
_PICK_TRIM = 0.02


class TagDiscoveryError(RuntimeError):
  """No unambiguous gripper assignment; the message carries the evidence."""


@dataclass(frozen=True)
class GripperTags:
  """Base tag id and ``FINGER_JOINTS``-ordered finger tag ids, ``None`` if untagged."""

  base: int
  fingers: tuple[int | None, int | None]


@dataclass(frozen=True)
class _TagTrack:
  tag_id: int
  poses: dict[int, gtsam.Pose3]  # pick index -> lens_T_tag
  corner_spread: float  # median RMS corner displacement, pixels

  @property
  def rigid(self) -> bool:
    return self.corner_spread <= _MAX_BASE_CORNER_SPREAD


def uniform_picks(n: int) -> npt.NDArray[np.int64]:
  """``PICK_COUNT`` of ``range(n)`` spread uniformly, ends trimmed; all if shorter."""
  if n <= PICK_COUNT:
    return np.arange(n, dtype=np.int64)
  lo, hi = int(n * _PICK_TRIM), int(n * (1.0 - _PICK_TRIM)) - 1
  return np.unique(np.linspace(lo, hi, PICK_COUNT).round().astype(np.int64))


def discovery_samples(sync_samples: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
  """``PICK_COUNT`` of the lens's sync samples, spread uniformly."""
  return sync_samples[uniform_picks(len(sync_samples))]


def _tag_tracks(
  tags_per_pick: list[list[TagDetection]],
  params: MeiParams,
  tag_size: float,
) -> list[_TagTrack]:
  by_id: dict[int, dict[int, gtsam.Pose3]] = {}
  corners: dict[int, dict[int, npt.NDArray[np.float64]]] = {}
  for pick, dets in enumerate(tags_per_pick):
    # Two tags carrying one id leave that id's pose ambiguous.
    seen = Counter(det.tag_id for det in dets)
    for det in dets:
      if seen[det.tag_id] > 1:
        continue
      pose = estimate_lens_T_tag(det.corners, tag_size, params)
      if pose is None:
        continue
      by_id.setdefault(det.tag_id, {})[pick] = pose
      corners.setdefault(det.tag_id, {})[pick] = det.corners

  tracks = []
  for tag_id, poses in sorted(by_id.items()):
    if len(poses) < _MIN_PAIR_SIGHTINGS:
      continue
    px = np.array([corners[tag_id][s] for s in poses])
    spread = float(
      np.median(np.sqrt(((px - np.median(px, axis=0)) ** 2).sum(-1).mean(-1)))
    )
    tracks.append(_TagTrack(tag_id=tag_id, poses=poses, corner_spread=spread))
  return tracks


def _swept_arc(
  joint: RevoluteJoint,
) -> tuple[list[gtsam.Pose3], npt.NDArray[np.float64]]:
  """Finger-tag poses swept over the joint range, and their stacked translations."""
  assert joint.joint_T_tag0 is not None
  thetas = np.linspace(joint.lower, joint.upper, _ARC_SAMPLES)
  arc = [
    joint.base_T_joint.compose(
      gtsam.Pose3(gtsam.Rot3.Rz(float(th)), gtsam.Point3(0, 0, 0))
    ).compose(joint.joint_T_tag0)
    for th in thetas
  ]
  return arc, np.array([translation(p) for p in arc])


def _arc_fit(
  base: _TagTrack,
  other: _TagTrack,
  arc: list[gtsam.Pose3],
  arc_t: npt.NDArray[np.float64],
) -> tuple[float, float] | None:
  """Median arc residuals (position, rotation) of ``other`` from ``base``, or None."""
  shared = sorted(base.poses.keys() & other.poses.keys())
  if len(shared) < _MIN_PAIR_SIGHTINGS:
    return None
  pos_res, rot_res = [], []
  for f in shared:
    rel = base.poses[f].between(other.poses[f])
    d = np.linalg.norm(arc_t - translation(rel), axis=1)
    i = int(np.argmin(d))
    pos_res.append(float(d[i]))
    rot_res.append(
      float(
        np.linalg.norm(gtsam.Rot3.Logmap(arc[i].rotation().between(rel.rotation())))
      )
    )
  return float(np.median(pos_res)), float(np.median(rot_res))


def _evidence(
  tracks: list[_TagTrack],
  n_picks: int,
  anchor_tag_ids: frozenset[int] | set[int],
  fits: dict[tuple[int, str, int], tuple[float, float]],
) -> str:
  """The per-tag rigidity and per-arc residual table a refusal reports."""
  lines = [
    f"  tag {t.tag_id}: {len(t.poses)}/{n_picks} sightings, "
    f"corner spread {t.corner_spread:.2f} px"
    + (" [anchor]" if t.tag_id in anchor_tag_ids else "")
    + (" [rigid]" if t.rigid else "")
    for t in tracks
  ]
  lines += [
    f"  base {b} / {name} <- tag {o}: pos {1000 * pos:.1f} mm, "
    f"rot {np.degrees(rot):.1f} deg"
    for (b, name, o), (pos, rot) in sorted(fits.items())
  ]
  return "\n".join(lines)


def discover_gripper_tags(
  tags_per_pick: list[list[TagDetection]],
  params: MeiParams,
  joints: dict[str, RevoluteJoint],
  tag_size: float,
  anchor_tag_ids: frozenset[int] | set[int],
) -> GripperTags:
  """This camera's base and finger tags, or ``TagDiscoveryError`` with the evidence."""
  finger_joints = {
    name: j
    for name, j in joints.items()
    if name in FINGER_JOINTS and j.joint_T_tag0 is not None
  }
  if not finger_joints:
    raise TagDiscoveryError(
      f"the URDF carries no finger joint with an AprilTag frame "
      f"(need {list(FINGER_JOINTS)}); discovery cannot assign a base tag"
    )
  tracks = _tag_tracks(tags_per_pick, params, tag_size)
  candidates = [t for t in tracks if t.rigid and t.tag_id not in anchor_tag_ids]

  arcs = {name: _swept_arc(joint) for name, joint in finger_joints.items()}
  fits: dict[tuple[int, str, int], tuple[float, float]] = {}
  riders: dict[tuple[int, str], list[int]] = {}
  for base in candidates:
    for name in finger_joints:
      riders[base.tag_id, name] = []
      for other in tracks:
        if other.tag_id == base.tag_id or other.tag_id in anchor_tag_ids:
          continue
        fit = _arc_fit(base, other, *arcs[name])
        if fit is None:
          continue
        fits[base.tag_id, name, other.tag_id] = fit
        if fit[0] <= _MAX_FINGER_POS and fit[1] <= _MAX_FINGER_ROT:
          riders[base.tag_id, name].append(other.tag_id)

  left, right = FINGER_JOINTS
  untagged: list[int | None] = [None]
  found = [
    GripperTags(b.tag_id, (lf, rf))
    for b in candidates
    for lf in riders.get((b.tag_id, left)) or untagged
    for rf in riders.get((b.tag_id, right)) or untagged
    if lf != rf
  ]
  # A gripper showing one finger tag still assigns, but a full pair outranks it.
  paired = [g for g in found if None not in g.fingers]
  found = paired or found
  if len(found) != 1:
    what = (
      "no tag pair rides the URDF finger arcs from any camera-rigid base"
      if not found
      else f"{len(found)} assignments fit: {found}"
    )
    raise TagDiscoveryError(
      f"cannot identify the gripper tags in the rear lens: {what}\n"
      f"{_evidence(tracks, len(tags_per_pick), anchor_tag_ids, fits)}"
    )
  return found[0]


def validate_discovered(tags: dict[str, GripperTags]) -> None:
  """Raise unless each tag id belongs to exactly one camera."""
  seen: dict[int, str] = {}
  for slot, g in tags.items():
    for tag_id in (g.base, *g.fingers):
      if tag_id is None:
        continue
      if tag_id in seen:
        raise ValueError(
          f"{seen[tag_id]} and {slot} both discovered tag {tag_id}; each "
          f"camera must carry its own"
        )
      seen[tag_id] = slot
