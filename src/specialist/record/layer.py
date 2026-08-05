"""The track layer: one rrd of the solve's output, on the base's ``recording_id``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr

from specialist.record import schema
from specialist.vi_ba.summary import CameraSummary, Summary
from specialist.vi_ba.viz import (
  CameraVizData,
  log_multi_ba_entities,
  run_blueprint,
)
from specialist.vi_ba.viz_build import RunOutputs

__all__ = ["write_track_layer"]


def _finite_rms(values: list[float] | None) -> float:
  if not values:
    return float("nan")
  finite = [v for v in values if np.isfinite(v)]
  return float(np.sqrt(np.mean(np.square(finite)))) if finite else float("nan")


def _send_run_properties(
  summary: Summary,
  per_tracker: list[CameraSummary],
  cameras: list[CameraVizData],
  audio_offset_by_tracker: dict[str, float],
  session_id: str | None,
) -> None:
  rr.send_property(
    "run",
    rr.AnyValues(
      session_id=session_id or "",
      sources=list(summary.sources),
      slots=list(summary.slots),
      audio_offset=[
        float(audio_offset_by_tracker.get(name, 0.0)) for name in summary.slots
      ],
    ),
  )
  rr.send_property(
    "solver",
    rr.AnyValues(
      initial_graph_error=summary.initial_graph_error,
      final_graph_error=summary.final_graph_error,
      iterations=summary.solver_iterations,
      cross_tracker_obs=summary.cross_tracker_obs,
      secondary_anchor_move_translation=summary.secondary_anchor_move_translation,
      secondary_anchor_move_rotation=summary.secondary_anchor_move_rotation,
      non_default_flags=list(summary.non_default_flags),
    ),
  )
  by_name = {t.name: t for t in cameras}
  for tracker in per_tracker:
    viz = by_name.get(tracker.name)
    rr.send_property(
      f"quality_{tracker.name}",
      rr.AnyValues(
        n_keyframes=tracker.n_keyframes,
        anchor_obs=tracker.anchor_obs,
        path_length=tracker.path_length,
        bounding_box_diagonal=tracker.bounding_box_diagonal,
        front_rmse=_finite_rms(viz.front_rmse_per_kf if viz else None),
        rear_rmse=_finite_rms(viz.rear_rmse_per_kf if viz else None),
      ),
    )


def write_track_layer(
  rrd_path: Path,
  outputs: RunOutputs,
  summary: Summary,
  reference_tag_id: int,
  anchor_tag_size: float,
  mask_pad_px: int = 0,
  urdf_path: Path | None = None,
  session_id: str | None = None,
  open_viewer: bool = True,
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
  if open_viewer:
    rr.spawn(connect=False)
    rr.set_sinks(rr.FileSink(str(rrd_path)), rr.GrpcSink())
  else:
    # Headless: write the rrd directly.
    rr.save(str(rrd_path))
  log_multi_ba_entities(
    outputs.viz_trackers,
    outputs.world_T_anchor,
    reference_tag_id,
    anchor_tag_size,
    mask_pad_px=mask_pad_px,
    urdf_path=urdf_path,
  )
  _send_run_properties(
    summary,
    outputs.per_tracker,
    outputs.viz_trackers,
    outputs.audio_offset_by_tracker,
    session_id,
  )
  rr.disconnect()
