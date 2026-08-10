"""The track layer: one rrd of the solve's output, on the base's ``recording_id``."""

from __future__ import annotations

from pathlib import Path

import rerun as rr

from specialist.record import schema
from specialist.vi_ba.summary import CameraSummary, Summary
from specialist.vi_ba.viz import log_multi_ba_entities, run_blueprint
from specialist.vi_ba.viz_build import RunOutputs

__all__ = ["write_track_layer"]


def _send_run_properties(summary: Summary, per_tracker: list[CameraSummary]) -> None:
  rr.send_property(
    "solver",
    rr.AnyValues(
      iterations=summary.solver_iterations,
      cross_tracker_obs=summary.cross_tracker_obs,
      secondary_anchor_move_translation=summary.secondary_anchor_move_translation,
      secondary_anchor_move_rotation=summary.secondary_anchor_move_rotation,
      non_default_flags=list(summary.non_default_flags),
    ),
  )
  for tracker in per_tracker:
    rr.send_property(
      f"quality_{tracker.name}",
      rr.AnyValues(front_rmse=tracker.front_rmse),
    )


def write_track_layer(
  rrd_path: Path,
  outputs: RunOutputs,
  summary: Summary,
  reference_tag_id: int,
  anchor_tag_size: float,
  session_id: str,
  mask_pad_px: int,
  urdf_path: Path | None = None,
  open_viewer: bool = True,
  timeline_start: float | None = None,
  frame_period: float | None = None,
) -> None:
  """Write the run's track layer (and open the viewer on it by default)."""
  rr.init(
    schema.APPLICATION_ID,
    recording_id=session_id,
    spawn=False,
    send_properties=False,
  )
  rr.send_blueprint(
    run_blueprint(outputs.viz_trackers), make_active=True, make_default=True
  )
  partial = rrd_path.parent / (rrd_path.name + ".partial")
  partial.unlink(missing_ok=True)
  if open_viewer:
    rr.spawn(connect=False)
    rr.set_sinks(rr.FileSink(str(partial)), rr.GrpcSink())
  else:
    # Headless: write the rrd directly.
    rr.save(str(partial))
  log_multi_ba_entities(
    outputs.viz_trackers,
    outputs.world_T_anchor,
    reference_tag_id,
    anchor_tag_size,
    mask_pad_px=mask_pad_px,
    urdf_path=urdf_path,
    timeline_start=timeline_start,
    frame_period=frame_period,
  )
  _send_run_properties(summary, outputs.per_tracker)
  rr.disconnect()
  partial.replace(rrd_path)
