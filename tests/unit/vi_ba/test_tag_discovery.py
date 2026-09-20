"""Discovery separates the gripper's own tags from every other tag in the rear lens."""

from __future__ import annotations

from dataclasses import replace

import gtsam
import numpy as np
import pytest

from specialist.vi_ba.tag_detection import TagDetection
from specialist.vi_ba.tag_discovery import (
  PICK_COUNT,
  GripperTags,
  TagDiscoveryError,
  discover_gripper_tags,
  uniform_picks,
  validate_discovered,
)
from specialist.vi_ba.urdf_joints import RevoluteJoint
from tests.mei_fixtures import synthetic_tag_detection, wide_mei_params

_PARAMS = wide_mei_params()
_TAG_SIZE = 0.01
_ANCHORS = frozenset({1})
_N = 12
_OWN = gtsam.Pose3(gtsam.Rot3.Rx(2.7), gtsam.Point3(0.01, 0.02, 0.10))


def _mirrored_joint(sign: float) -> RevoluteJoint:
  """One finger of a gripper whose joints mirror across the base-tag y axis."""
  return RevoluteJoint(
    base_T_joint=gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.04, sign * 0.035, -0.008)),
    lower=-0.262 if sign > 0 else -0.698,
    upper=0.698 if sign > 0 else 0.262,
    joint_T_tag0=gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.01, 0.0, 0.005)),
  )


_JOINTS = {"left_finger": _mirrored_joint(1.0), "right_finger": _mirrored_joint(-1.0)}


def _detect(lens_T_tag: gtsam.Pose3, tag_id: int) -> TagDetection | None:
  return synthetic_tag_detection(lens_T_tag, tag_id, _TAG_SIZE, _PARAMS)


def _finger(
  lens_T_base_tag: gtsam.Pose3, joint: RevoluteJoint, theta: float
) -> gtsam.Pose3:
  assert joint.joint_T_tag0 is not None
  swing = gtsam.Pose3(gtsam.Rot3.Rz(theta), gtsam.Point3(0, 0, 0))
  return (
    lens_T_base_tag.compose(joint.base_T_joint)
    .compose(swing)
    .compose(joint.joint_T_tag0)
  )


def _gripper(
  dets: list[TagDetection | None],
  lens_T_base_tag: gtsam.Pose3,
  ids: tuple[int, int, int],
  theta: float,
  with_base: bool = True,
  with_right: bool = True,
) -> None:
  if with_base:
    dets.append(_detect(lens_T_base_tag, ids[0]))
  dets.append(_detect(_finger(lens_T_base_tag, _JOINTS["left_finger"], theta), ids[1]))
  if with_right:
    dets.append(
      _detect(_finger(lens_T_base_tag, _JOINTS["right_finger"], -theta), ids[2])
    )


def _scene(
  own_thetas: list[float] | None = None,
  with_base: bool = True,
  with_right_finger: bool = True,
  foreign_ids: tuple[int, int, int] | None = (2003, 2004, 2005),
  foreign_still: bool = False,
  anchor: bool = True,
  rigid_anchor: bool = False,
  own_ids: tuple[int, int, int] = (2000, 2001, 2002),
) -> list[list[TagDetection]]:
  """Our gripper at ``own_ids``, the other operator's gripper and a wall anchor."""
  thetas = own_thetas if own_thetas is not None else list(np.linspace(-0.2, 0.6, _N))
  picks: list[list[TagDetection]] = []
  for f in range(_N):
    dets: list[TagDetection | None] = []
    _gripper(dets, _OWN, own_ids, thetas[f], with_base, with_right_finger)
    if foreign_ids is not None:
      swing = 0.0 if foreign_still else 0.22 * f
      drift = 0.0 if foreign_still else 0.006 * f
      _gripper(
        dets,
        gtsam.Pose3(
          gtsam.Rot3.Rx(2.7).compose(gtsam.Rot3.Rz(swing)),
          gtsam.Point3(-0.05 + drift, -0.03, 0.13 + drift),
        ),
        foreign_ids,
        thetas[f],
      )
    if anchor:
      pose = (
        gtsam.Pose3(gtsam.Rot3.Rx(2.8), gtsam.Point3(-0.03, 0.01, 0.15))
        if rigid_anchor
        else gtsam.Pose3(
          gtsam.Rot3.Rx(2.8 + 0.1 * f), gtsam.Point3(-0.03 + 0.02 * f, 0.01, 0.15)
        )
      )
      dets.append(_detect(pose, 1))
    picks.append([d for d in dets if d is not None])
  return picks


def _discover(
  picks: list[list[TagDetection]],
  joints: dict[str, RevoluteJoint] | None = None,
) -> GripperTags:
  return discover_gripper_tags(
    picks, _PARAMS, joints or _JOINTS, tag_size=_TAG_SIZE, anchor_tag_ids=_ANCHORS
  )


def test_discovers_the_gripper_triple() -> None:
  assert _discover(_scene()) == GripperTags(2000, (2001, 2002))


def test_barely_opening_gripper_still_resolves() -> None:
  # Only the arc's rotation residual tells a near-constant finger from the base.
  assert _discover(_scene(own_thetas=[0.3 + 0.002 * f for f in range(_N)])).base == 2000


def test_missing_base_refuses_with_evidence() -> None:
  with pytest.raises(TagDiscoveryError) as exc:
    _discover(_scene(with_base=False, foreign_ids=None))
  assert "2001" in str(exc.value)


def test_rigid_anchor_is_never_a_candidate() -> None:
  assert _discover(_scene(rigid_anchor=True)).base == 2000


def test_tag_zero_is_a_finger_tag_like_any_other() -> None:
  found = _discover(
    _scene(with_right_finger=False, foreign_ids=None, own_ids=(2000, 0, 2002))
  )
  assert found == GripperTags(2000, (0, None))


def test_sparse_sightings_are_dropped() -> None:
  picks = _scene()
  picks[3].append(
    _detect(gtsam.Pose3(gtsam.Rot3.Rx(2.7), gtsam.Point3(0, 0, 0.2)), 910)
  )
  assert _discover(picks).base == 2000


def test_rejects_two_cameras_sharing_any_tag() -> None:
  """A shared id would enter the graph as two cameras' factors."""
  validate_discovered(
    {"t0": GripperTags(2000, (2001, 2002)), "t1": GripperTags(2003, (2004, 2005))}
  )
  with pytest.raises(ValueError, match="both discovered tag 2000"):
    validate_discovered(
      {"t0": GripperTags(2000, (None, None)), "t1": GripperTags(2000, (None, None))}
    )
  with pytest.raises(ValueError, match="both discovered tag 0"):
    validate_discovered(
      {"t0": GripperTags(2000, (0, None)), "t1": GripperTags(2003, (0, None))}
    )


def test_uniform_picks_bounds() -> None:
  picks = uniform_picks(4000)
  assert len(picks) == PICK_COUNT and picks[0] >= 0 and picks[-1] < 4000
  assert list(uniform_picks(5)) == [0, 1, 2, 3, 4]


def test_a_gripper_with_one_finger_tagged_assigns_rather_than_crashing() -> None:
  """The untagged finger stays ``None``."""
  joints = dict(_JOINTS)
  joints["right_finger"] = replace(joints["right_finger"], joint_T_tag0=None)
  assert _discover(_scene(), joints=joints) == GripperTags(2000, (2001, None))


def test_a_second_still_gripper_refuses_rather_than_guesses() -> None:
  with pytest.raises(TagDiscoveryError, match="2 assignments fit"):
    _discover(_scene(foreign_still=True))


def test_an_anchor_id_is_never_a_base_or_a_finger() -> None:
  # The rival wears the anchor id itself, so the scene logs no separate anchor.
  for ids in ((1, 91, 92), (90, 1, 92)):
    assert _discover(
      _scene(foreign_ids=ids, foreign_still=True, anchor=False)
    ) == GripperTags(2000, (2001, 2002))


def test_a_tag_on_the_arc_but_turned_off_it_is_rejected() -> None:
  """Only the arc's rotation residual rejects it; the position residual accepts it."""
  joint = _JOINTS["left_finger"]
  picks: list[list[TagDetection]] = []
  for f in range(_N):
    dets: list[TagDetection | None] = []
    _gripper(dets, _OWN, (2000, 2001, 2002), float(np.linspace(-0.2, 0.6, _N)[f]))
    on_arc = _finger(_OWN, joint, 0.30)
    turned = gtsam.Pose3(
      on_arc.rotation().compose(gtsam.Rot3.Rz(np.radians(40.0))), on_arc.translation()
    )
    dets.append(_detect(turned, 2010))
    picks.append([d for d in dets if d is not None])
  assert _discover(picks) == GripperTags(2000, (2001, 2002))


def test_a_finger_past_the_urdf_limit_still_resolves() -> None:
  """Joint limits are the URDF's, not the hardware's."""
  over = _JOINTS["left_finger"].upper + np.radians(12.0)
  picks: list[list[TagDetection]] = []
  for _ in range(_N):
    dets: list[TagDetection | None] = []
    _gripper(dets, _OWN, (2000, 2001, 2002), float(over))
    picks.append([d for d in dets if d is not None])
  assert _discover(picks).base == 2000


def test_one_tag_id_twice_in_a_frame_refuses() -> None:
  """Two tags sharing an id make every per-frame pose ambiguous."""
  picks = _scene(foreign_ids=None)
  twin = gtsam.Pose3(gtsam.Rot3.Rx(2.7), gtsam.Point3(-0.06, 0.02, 0.14))
  for dets in picks:
    doubled = _detect(twin, 2000)
    if doubled is not None:
      dets.append(doubled)
  with pytest.raises(TagDiscoveryError) as exc:
    _discover(picks)
  assert "tag 2000" not in str(exc.value)


def test_a_minority_of_flipped_frames_does_not_lose_the_finger() -> None:
  """The arc residual is a median, so a mirrored IPPE minority cannot outvote."""
  joint = _JOINTS["left_finger"]
  thetas = np.linspace(-0.2, 0.6, _N)
  picks: list[list[TagDetection]] = []
  for f in range(_N):
    dets: list[TagDetection | None] = []
    _gripper(dets, _OWN, (2000, 2001, 2002), float(thetas[f]))
    if f % 4 == 0:  # a quarter of the frames land on the mirrored branch
      on_arc = _finger(_OWN, joint, float(thetas[f]))
      dets[1] = _detect(
        gtsam.Pose3(
          on_arc.rotation().compose(gtsam.Rot3.Rx(np.radians(100.0))),
          on_arc.translation(),
        ),
        2001,
      )
    picks.append([d for d in dets if d is not None])
  assert _discover(picks) == GripperTags(2000, (2001, 2002))


def test_discovery_samples_are_sync_samples_spread_over_the_lens() -> None:
  """Discovery takes at most ``PICK_COUNT`` sync samples spanning the lens."""
  from specialist.vi_ba.tag_discovery import discovery_samples

  sync = np.arange(0, 3000, 16, dtype=np.int64)
  picked = discovery_samples(sync)
  assert len(picked) <= PICK_COUNT
  assert set(picked.tolist()) <= set(sync.tolist())
  assert picked[0] < 200 and picked[-1] > 2700
