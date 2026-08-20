"""Track and viz layers on the base's ``recording_id``, with no ``RecordingInfo``."""

from __future__ import annotations

import socket
from dataclasses import asdict
from pathlib import Path
from typing import Any

import rerun as rr
from rerun.experimental import (
  Chunk,
  LazyChunkStream,
  OptimizationProfile,
  send_chunks,
)

from specialist.record import schema
from specialist.vi_ba.summary import CameraSummary, Summary
from specialist.vi_ba.viz import run_blueprint, solution_chunks
from specialist.vi_ba.viz_build import RunOutputs

__all__ = ["write_solution_layers"]

_VIEWER_PORT = 9876


def _split_group(
  stream: rr.RecordingStream,
  group: str,
  entity: str,
  values: dict[str, Any],
  headline: tuple[str, ...],
) -> None:
  """Headline keys become segment-table columns; the rest goes on the static entity."""
  stream.send_property(group, rr.AnyValues(**{k: values[k] for k in headline}))
  rest = {k: v for k, v in values.items() if k not in headline}
  rr.log(entity, rr.AnyValues(**rest), static=True, recording=stream)


def _send_run_properties(
  summary: Summary, per_tracker: list[CameraSummary], stream: rr.RecordingStream
) -> None:
  """Every summary number, to the layer that keeps them."""
  run = asdict(summary)
  stream.send_property("session", rr.AnyValues(sources=list(run.pop("sources"))))
  _split_group(
    stream, "solver", "run_summary", run, ("cross_tracker_obs", "solver_iterations")
  )
  for tracker in per_tracker:
    _split_group(
      stream,
      f"quality_{tracker.name}",
      f"run_summary/{tracker.name}",
      asdict(tracker),
      ("front_rmse",),
    )


def _stream(session_id: str) -> rr.RecordingStream:
  return rr.RecordingStream(
    schema.APPLICATION_ID, recording_id=session_id, send_properties=False
  )


def _viewer_sink() -> list[rr.GrpcSink]:
  """Spawn a viewer and return a sink into it, or none when it cannot be opened."""
  try:
    rr.spawn(connect=False)
    socket.create_connection(("127.0.0.1", _VIEWER_PORT), timeout=5.0).close()
  except Exception as exc:  # noqa: BLE001 - the viewer is optional; the layers are not
    print(f"  [layer] viewer not opened ({exc}); writing the layers without it")
    return []
  return [rr.GrpcSink(url=f"rerun+http://127.0.0.1:{_VIEWER_PORT}/proxy")]


def write_solution_layers(
  track_path: Path,
  viz_path: Path,
  outputs: RunOutputs,
  summary: Summary,
  reference_tag_id: int,
  anchor_tag_size: float,
  session_id: str,
  mask_pad_px: int,
  open_viewer: bool,
  urdf_path: Path | None = None,
  timeline_start: float | None = None,
  frame_period: float | None = None,
) -> None:
  """Write the run's track and viz layers (and open the viewer by default)."""
  blueprint = run_blueprint(outputs.viz_trackers)
  track_path.parent.mkdir(parents=True, exist_ok=True)
  viz_path.parent.mkdir(parents=True, exist_ok=True)
  partials = {
    path: path.parent / (path.name + ".partial") for path in (track_path, viz_path)
  }
  for partial in partials.values():
    partial.unlink(missing_ok=True)

  viewer = _viewer_sink() if open_viewer else []

  # Viz chunks stream to the file; track chunks are shaped for the object store.
  kept: list[Chunk] = []
  try:
    with _stream(session_id) as viz_stream:
      viz_stream.set_sinks(rr.FileSink(partials[viz_path]), *viewer)
      viz_stream.send_blueprint(blueprint, make_active=True, make_default=True)
      solution_chunks(
        kept.append,
        lambda chunk: send_chunks([chunk], recording=viz_stream),
        outputs.viz_trackers,
        outputs.world_T_anchor,
        reference_tag_id,
        anchor_tag_size,
        mask_pad_px=mask_pad_px,
        urdf_path=urdf_path,
        timeline_start=timeline_start,
        frame_period=frame_period,
      )

    store = LazyChunkStream.from_iter(kept).collect(
      optimize=OptimizationProfile.OBJECT_STORE
    )
    with _stream(session_id) as track_stream:
      track_stream.set_sinks(rr.FileSink(partials[track_path]), *viewer)
      send_chunks(store, recording=track_stream)
      _send_run_properties(summary, outputs.per_tracker, track_stream)
    for path, partial in partials.items():
      partial.replace(path)
  finally:
    for partial in partials.values():
      partial.unlink(missing_ok=True)
