"""Estimate and viz layers: the base's recording id, never a ``RecordingInfo``."""

import math
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import rerun as rr
from rerun.chunk import Chunk, LazyChunkStream, OptimizationProfile
from specialist_dataset import schema

from specialist.vi_ba.summary import CameraSummary, Summary
from specialist.vi_ba.viz import estimate_chunks
from specialist.vi_ba.viz_build import RunEstimate

__all__ = ["git_sha", "write_estimate_layers"]


def git_sha() -> str:
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


def _run_chunks(
  summary: Summary, per_camera: list[CameraSummary], estimated_at: str
) -> list[Chunk]:
  """The run's ``estimate`` property and each camera's numbers on its ``fit`` entity."""
  chunks = []
  for camera in per_camera:
    fields = asdict(camera)
    entity = schema.fit(fields.pop("slot"))
    columns = cast("Any", rr.AnyValues).columns(**{k: [v] for k, v in fields.items()})
    chunks.append(Chunk.from_columns(entity, [], columns))
  # A camera the run never measured would poison max() with its NaN.
  measured = [c.front_rmse for c in per_camera if not math.isnan(c.front_rmse)]
  account = rr.AnyValues(
    **asdict(summary),
    worst_rmse=max(measured) if measured else float("nan"),
    git_sha=git_sha(),
    estimated_at=estimated_at,
    rerun_version=rr.__version__,
  )
  return [*chunks, Chunk.from_property(schema.ESTIMATE, account)]


def write_estimate_layers(
  estimate_path: Path,
  viz_path: Path,
  estimate: RunEstimate,
  summary: Summary,
  anchor_tag_size: float,
  session_id: str,
  urdf_path: Path,
  timeline_start: float,
  frame_period: float,
) -> None:
  estimated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
  estimate_path.parent.mkdir(parents=True, exist_ok=True)
  viz_path.parent.mkdir(parents=True, exist_ok=True)
  partials = {
    path: path.parent / (path.name + ".partial") for path in (estimate_path, viz_path)
  }
  layers = estimate_chunks(
    estimate.cameras,
    estimate.world_T_anchor,
    estimate.reference_tag_id,
    anchor_tag_size,
    urdf_path=urdf_path,
    timeline_start=timeline_start,
    frame_period=frame_period,
  )
  kept = layers.estimate + _run_chunks(summary, estimate.per_camera, estimated_at)
  ids = {"application_id": schema.APPLICATION_ID, "recording_id": session_id}
  try:
    LazyChunkStream.from_iter(layers.viz).write_rrd(partials[viz_path], **ids)
    LazyChunkStream.from_iter(kept).collect(
      optimize=OptimizationProfile.OBJECT_STORE
    ).write_rrd(partials[estimate_path], **ids)
    for path, partial in partials.items():
      partial.replace(path)
  finally:
    for partial in partials.values():
      partial.unlink(missing_ok=True)
