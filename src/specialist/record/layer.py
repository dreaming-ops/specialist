"""Estimate and viz layers on the base's ``recording_id``, with no ``RecordingInfo``."""

from __future__ import annotations

import math
import socket
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import rerun as rr
from rerun.experimental import (
  Chunk,
  LazyChunkStream,
  OptimizationProfile,
  send_chunks,
)
from specialist_dataset import schema

from specialist.vi_ba.summary import CameraSummary, Summary
from specialist.vi_ba.viz import Layers, estimate_chunks, run_blueprint
from specialist.vi_ba.viz_build import RunEstimate

__all__ = ["write_estimate_layers"]

_VIEWER_PORT = 9876


def _git_sha() -> str:
  """The current commit, or ``unknown``: an empty value would write no column."""
  try:
    return subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=Path(__file__).parent,
      capture_output=True,
      text=True,
      timeout=5.0,
      check=True,
    ).stdout.strip()
  except Exception:  # noqa: BLE001 - provenance is optional; the layers are not
    return "unknown"


def _send_run_properties(
  summary: Summary,
  per_camera: list[CameraSummary],
  stream: rr.RecordingStream,
  estimated_at: str,
) -> None:
  """The run's ``estimate`` property and each camera's numbers on its ``fit`` entity."""
  for camera in per_camera:
    fields = asdict(camera)
    rr.log(
      schema.fit(fields.pop("slot")),
      rr.AnyValues(**fields),
      static=True,
      recording=stream,
    )
  # A camera the run never measured would poison max() with its NaN.
  measured = [c.front_rmse for c in per_camera if not math.isnan(c.front_rmse)]
  stream.send_property(
    schema.ESTIMATE,
    rr.AnyValues(
      **asdict(summary),
      worst_rmse=max(measured) if measured else float("nan"),
      git_sha=_git_sha(),
      estimated_at=estimated_at,
      rerun_version=rr.__version__,
    ),
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
    print(f"  [warn] [layer] viewer not opened ({exc}); writing the layers without it")
    return []
  return [rr.GrpcSink(url=f"rerun+http://127.0.0.1:{_VIEWER_PORT}/proxy")]


def write_estimate_layers(
  estimate_path: Path,
  viz_path: Path,
  estimate: RunEstimate,
  summary: Summary,
  anchor_tag_size: float,
  session_id: str,
  open_viewer: bool,
  urdf_path: Path,
  timeline_start: float,
  frame_period: float,
) -> None:
  """Write the run's estimate and viz layers, and open the viewer if asked."""
  estimated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
  estimate_path.parent.mkdir(parents=True, exist_ok=True)
  viz_path.parent.mkdir(parents=True, exist_ok=True)
  partials = {
    path: path.parent / (path.name + ".partial") for path in (estimate_path, viz_path)
  }
  for partial in partials.values():
    partial.unlink(missing_ok=True)

  viewer = _viewer_sink() if open_viewer else []

  kept: list[Chunk] = []
  try:
    with _stream(session_id) as viz_stream:
      # The dashboard goes to the live viewer only, never into a layer file.
      if viewer:
        viz_stream.set_sinks(*viewer)
        viz_stream.send_blueprint(
          run_blueprint(
            estimate.cameras, estimate.world_T_anchor.keys(), anchor_tag_size
          ),
          make_active=True,
          make_default=True,
        )
      viz_stream.set_sinks(rr.FileSink(partials[viz_path]), *viewer)
      estimate_chunks(
        Layers(
          estimate=kept.append,
          viz=lambda chunk: send_chunks([chunk], recording=viz_stream),
        ),
        estimate.cameras,
        estimate.world_T_anchor,
        estimate.reference_tag_id,
        anchor_tag_size,
        urdf_path=urdf_path,
        timeline_start=timeline_start,
        frame_period=frame_period,
      )

    store = LazyChunkStream.from_iter(kept).collect(
      optimize=OptimizationProfile.OBJECT_STORE
    )
    with _stream(session_id) as estimate_stream:
      estimate_stream.set_sinks(rr.FileSink(partials[estimate_path]), *viewer)
      send_chunks(store, recording=estimate_stream)
      _send_run_properties(summary, estimate.per_camera, estimate_stream, estimated_at)
    for path, partial in partials.items():
      partial.replace(path)
  finally:
    for partial in partials.values():
      partial.unlink(missing_ok=True)
