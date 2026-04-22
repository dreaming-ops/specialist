"""The per-fisheye camera→tag observation the cross-tracker PnP factors consume."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import gtsam

FisheyeSide = Literal["front", "back"]
TagKind = Literal["anchor", "tracker"]


@dataclass(eq=False)
class TagObservation:
  """One camera→tag SE(3) observation from one fisheye at one keyframe."""

  observer_tracker: str
  fisheye_side: FisheyeSide
  keyframe_index: int
  tag_id: str
  tag_kind: TagKind
  tag_owner_tracker: str | None
  camera_to_tag: gtsam.Pose3
