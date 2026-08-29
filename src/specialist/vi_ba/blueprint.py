"""The standard dashboards: a solve's per-slot view and the state layer's."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from specialist_dataset import schema
from specialist_dataset.fiducials import tag_corners_local

from specialist.vi_ba.urdf_joints import FINGER_JOINTS

__all__ = [
  "BODY_AXIS",
  "TAG_AXIS",
  "TRAJECTORY_TRAIL_DURATION",
  "blueprint_for_slots",
  "state_blueprint",
  "state_card_blueprint",
]

# Seconds the trajectory ribbon trails the cursor; segments and time window share it.
TRAJECTORY_TRAIL_DURATION = 1.0

# Keyframe-grid rate (Hz) of the time panel, the same at any capture rate.
_KEYFRAME_FPS = 10.0


def _panels() -> tuple[rrb.BlueprintPanel, rrb.SelectionPanel, rrb.TimePanel]:
  """The chrome every dashboard hides, and the clock they open on."""
  return (
    rrb.BlueprintPanel(state=rrb.PanelState.Collapsed),
    rrb.SelectionPanel(state=rrb.PanelState.Collapsed),
    rrb.TimePanel(
      state=rrb.PanelState.Collapsed,
      timeline=schema.TIMELINE,
      fps=_KEYFRAME_FPS,
    ),
  )


_TAG_COLOR = (255, 210, 0)

BODY_AXIS = 0.05
_LENS_AXIS = 0.03
TAG_AXIS = 0.02


def _triad_overrides(
  slots: list[str], anchor_tag_ids: Iterable[int], tag_size: float
) -> dict[rr.datatypes.EntityPath | str, Any]:
  """A triad per exported frame; an override lists all visualizers its entity keeps."""
  corners = tag_corners_local(tag_size)
  square = np.vstack([corners, corners[:1]])
  axes: dict[rr.datatypes.EntityPath | str, Any] = {
    f"/{schema.anchor(int(tag_id))}": [
      rr.TransformAxes3D(tag_size / 2.0),
      rr.LineStrips3D(strips=[square], colors=[_TAG_COLOR], radii=[0.003]),
    ]
    for tag_id in anchor_tag_ids
  }
  for slot in slots:
    axes[f"/{schema.camera(slot)}"] = rr.TransformAxes3D(BODY_AXIS)
    axes[f"/{schema.gripper(slot)}"] = rr.TransformAxes3D(BODY_AXIS)
    axes[f"/{schema.base_tag(slot)}"] = rr.TransformAxes3D(TAG_AXIS)
    for lens in schema.LENSES:
      axes[f"/{schema.lens(slot, lens)}"] = rr.TransformAxes3D(_LENS_AXIS)
    axes[f"/{schema.lens_seat(slot, 'rear')}"] = rr.TransformAxes3D(_LENS_AXIS)
    for finger in FINGER_JOINTS:
      axes[f"/{schema.gripper_part(slot, finger)}"] = rr.TransformAxes3D(TAG_AXIS)
      axes[f"/{schema.joint_tag(slot, finger)}"] = rr.TransformAxes3D(TAG_AXIS)
  return axes


_HIDDEN = (
  (schema.detection, schema.KEYPOINTS_DETECTION),
  (schema.detection, schema.REJECTED_KEYPOINTS_DETECTION),
  (schema.overlay, schema.LANDMARK_REPROJECTION_OVERLAY),
)


def _hidden_overlay_overrides(
  slot: str, lens: str
) -> dict[rr.datatypes.EntityPath | str, rrb.EntityBehavior]:
  """Hide the front lens's dense pixel layers at start; other lenses have none."""
  if lens != "front":
    return {}
  return {
    f"/{path(slot, lens, kind)}": rrb.EntityBehavior(visible=False)
    for path, kind in _HIDDEN
  }


def _without_pixels(slots: list[str]) -> list[str]:
  """3D-view exclusions: lens pixel subtrees, never a lens."""
  return [
    f"- /{schema.lens(slot, lens)}/{tail}"
    for slot in slots
    for lens in schema.LENSES
    for tail in ("video", "video/**", "overlay/**", "detection/**")
  ]


def _scene(
  slots: list[str],
  anchor_tag_ids: Iterable[int] = (),
  anchor_tag_size: float = 0.1,
  overrides: dict[rr.datatypes.EntityPath | str, Any] | None = None,
  excluded: Iterable[str] = (),
) -> rrb.Spatial3DView:
  """The shared 3D view of triads, anchors and ribbons; ``overrides`` add to these."""
  every: dict[rr.datatypes.EntityPath | str, Any] = {
    **{f"/{schema.trajectory(s)}": _trail_window() for s in slots},
    **_triad_overrides(slots, anchor_tag_ids, anchor_tag_size),
    **{
      f"/{schema.landmark(s, lens)}": rrb.EntityBehavior(visible=False)
      for s in slots
      for lens in schema.LENSES
    },
    **(overrides or {}),
  }
  # Rerun matches a wildcard only as a trailing /**, so every path is literal.
  return rrb.Spatial3DView(
    name="3D",
    origin="/",
    contents=["+ /**", *_without_pixels(slots), *excluded],
    overrides=every,
  )


def blueprint_for_slots(
  slots: list[str],
  anchor_tag_ids: Iterable[int] = (),
  anchor_tag_size: float = 0.1,
) -> rrb.Blueprint:
  """A 3D view beside one image column per camera."""
  image_columns = [
    rrb.Vertical(
      *(
        rrb.Spatial2DView(
          name=f"{slot}/{lens}",
          origin=f"/{schema.lens(slot, lens)}",
          overrides=_hidden_overlay_overrides(slot, lens),
        )
        for lens in schema.LENSES
      ),
      name=slot,
    )
    for slot in slots
  ]
  return rrb.Blueprint(
    rrb.Horizontal(
      _scene(slots, anchor_tag_ids, anchor_tag_size),
      rrb.Horizontal(*image_columns) if len(slots) > 1 else image_columns[0],
      column_shares=[2, len(slots)],
    ),
    *_panels(),
  )


def state_blueprint() -> rrb.Blueprint:
  """Both hands' images beside a 3D view of hand triads."""
  return rrb.Blueprint(
    rrb.Horizontal(
      _scene(
        [],
        overrides={
          f"/{schema.state_pose(hand)}": rr.TransformAxes3D(BODY_AXIS)
          for hand in schema.HANDS
        },
        excluded=[f"- /{schema.state_image(hand)}" for hand in schema.HANDS],
      ),
      rrb.Vertical(
        *(
          rrb.Spatial2DView(name=hand, origin=f"/{schema.state_image(hand)}")
          for hand in schema.HANDS
        ),
        name="state",
      ),
      column_shares=[3, 1],
    ),
    *_panels(),
  )


def state_card_blueprint(frame_width: int) -> rrb.Blueprint:
  """Segment-table card of a state catalog dataset: both hands' images side by side."""
  w = float(frame_width)
  return rrb.Blueprint(
    rrb.Spatial2DView(
      name="state",
      origin=f"/{schema.STATE}",
      contents=[f"+ /{schema.state_image(h)}" for h in schema.HANDS],
      visual_bounds=rrb.VisualBounds2D(x_range=(0.0, 2.0 * w), y_range=(0.0, w)),
    ),
    *_panels(),
  )


def _trail_window() -> rrb.VisibleTimeRanges:
  """The time window a trajectory ribbon draws; a view without it draws one step."""
  return rrb.VisibleTimeRanges(
    [
      rrb.VisibleTimeRange(
        schema.TIMELINE,
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-TRAJECTORY_TRAIL_DURATION),
        end=rrb.TimeRangeBoundary.cursor_relative(),
      )
    ]
  )
