"""The run's JSON export: the run summary and the raw observations evaluation reads."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import MeiParams
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.compose import CulledPoints
from specialist.vi_ba.configs import Args
from specialist.vi_ba.single_camera import PreparedCamera
from specialist.vi_ba.tag_detection import TagDetection

Float64Array = npt.NDArray[np.float64]

__all__ = ["Summary", "CameraSummary", "build_raw_export", "write_run_json"]


@dataclass(frozen=True)
class CameraSummary:
  """One tracker's line in the run summary."""

  name: str
  n_keyframes: int
  anchor_obs: int
  bounding_box_diagonal: float
  path_length: float


@dataclass(frozen=True)
class Summary:
  """The run's header in the exported JSON: its inputs and the solve's own account."""

  sources: tuple[str, ...]
  slots: tuple[str, ...]
  cross_tracker_obs: int
  initial_graph_error: float
  final_graph_error: float
  solver_iterations: int


class _PoseDict(TypedDict):
  """A pose as the JSON carries it: row-major rotation plus translation."""

  R: list[list[float]]
  t: list[float]


def _pose_to_dict(p: gtsam.Pose3) -> _PoseDict:
  return _PoseDict(
    R=[list(map(float, row)) for row in p.rotation().matrix()],
    t=[float(x) for x in p.translation()],
  )


def _mei_to_dict(p: MeiParams) -> dict[str, float]:
  return {k: float(v) for k, v in asdict(p).items()}


def _lens_side_dict(lens: LensSetup, decoded_width: int) -> dict[str, object]:
  """One lens's body_T_lens, intrinsics and square decoded frame shape for the export."""
  return {
    "body_T_lens": _pose_to_dict(lens.body_T_lens),
    "params": _mei_to_dict(lens.params),
    "frame_shape": [decoded_width, decoded_width],
  }


def _tags_side_list(
  tags_per_kf: list[list[TagDetection]],
) -> list[list[dict[str, object]]]:
  """One lens's per-keyframe AprilTag detections (id + corners) for the export."""
  return [
    [
      {
        "tag_id": int(d.tag_id),
        "corners": [[float(p[0]), float(p[1])] for p in d.corners],
      }
      for d in tags
    ]
    for tags in tags_per_kf
  ]


def _tracks_side_dict(
  tracks: dict[int, dict[int, Float64Array]],
) -> dict[str, dict[str, list[float]]]:
  """One lens's ``{track_id: {keyframe: pixel}}`` observations for the export."""
  return {
    str(tid): {str(kf): [float(p[0]), float(p[1])] for kf, p in obs.items()}
    for tid, obs in tracks.items()
  }


def build_raw_export(
  cameras: list[PreparedCamera],
  optimized_body_poses: dict[str, list[gtsam.Pose3]],
  culled_points: CulledPoints,
  world_T_anchor: dict[int, gtsam.Pose3],
  audio_offset_by_name: dict[str, float],
  reference_tag_id: int,
  args: Args,
) -> dict[str, object]:
  """Every raw observation the run produced, as one flat JSON-ready dict."""
  raw: dict[str, object] = {}
  raw["body_poses_by_tracker"] = {
    t.name: [_pose_to_dict(p) for p in optimized_body_poses[t.name]] for t in cameras
  }
  raw["ref_anchor_pose_world"] = _pose_to_dict(world_T_anchor[reference_tag_id])
  raw["anchor_poses_world"] = {
    str(tag_id): _pose_to_dict(p) for tag_id, p in world_T_anchor.items()
  }
  raw["lens_by_tracker"] = {
    t.name: {
      "front": _lens_side_dict(t.front_lens, t.decoded_width),
      "rear": _lens_side_dict(t.rear_lens, t.decoded_width),
    }
    for t in cameras
  }
  raw["anchor_obs_by_tracker"] = {
    t.name: [
      {"tag_id": tag_id, "kf": int(kf), "side": side, "lens_T_tag": _pose_to_dict(p)}
      for tag_id, obs in t.anchor_obs_by_tag.items()
      for kf, sides in obs.items()
      for side, p in sides.items()
    ]
    for t in cameras
  }
  raw["tag_detections_by_tracker"] = {
    t.name: {
      "front": _tags_side_list(t.front_tags_per_kf),
      "rear": _tags_side_list(t.rear_tags_per_kf),
    }
    for t in cameras
  }
  raw["tracks_by_tracker"] = {
    t.name: {
      "front": _tracks_side_dict(t.front_tracks_tri),
      "rear": _tracks_side_dict(t.rear_tracks_tri),
    }
    for t in cameras
  }
  raw["landmarks_by_tracker"] = {
    t.name: {
      "front": {
        str(tid): [float(x) for x in p] for tid, p in culled_points[t.name][0].items()
      },
      "rear": {
        str(tid): [float(x) for x in p] for tid, p in culled_points[t.name][1].items()
      },
    }
    for t in cameras
  }
  raw["kf_times_by_tracker"] = {
    t.name: [float(x) for x in t.keyframe_times] for t in cameras
  }
  raw["audio_offset_by_tracker"] = audio_offset_by_name
  raw["body_T_base_tag_by_tracker"] = {
    t.name: (_pose_to_dict(t.body_T_base_tag) if t.body_T_base_tag else None)
    for t in cameras
  }
  raw["joint_angles_by_tracker"] = {
    t.name: t.joint_angles for t in cameras if t.joint_angles
  }
  raw["anchor_tag_id"] = int(reference_tag_id)
  raw["anchor_tag_size"] = float(args.anchor.tag_size)
  raw["gripper_keypoint_offset"] = [float(x) for x in args.gripper.tcp_in_base_tag]
  return raw


def write_run_json(
  path: Path,
  summary: Summary,
  per_tracker: list[CameraSummary],
  raw: dict[str, object],
) -> None:
  """Write the run's JSON: summary header, per-tracker lines and raw observations."""
  document = asdict(summary)
  document["per_tracker"] = [asdict(t) for t in per_tracker]
  document.update(raw)
  path.write_text(json.dumps(document, indent=2, default=str))
