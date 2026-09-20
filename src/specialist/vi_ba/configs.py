"""Config dataclasses: ``Args`` is the tyro CLI of ``estimate.py``; ``CameraArgs``
feeds ``prepare_camera``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from specialist.vi_ba.urdf_joints import RevoluteJoint

if TYPE_CHECKING:
  from specialist.record.session import SessionRecording

__all__ = [
  "AnchorCfg",
  "Args",
  "CameraArgs",
  "FrontendCfg",
  "GripperCfg",
  "SolverCfg",
  "validate",
]


@dataclass
class FrontendCfg:
  # ORB features per frame for revisit matching.
  orb_n_features: int = 4000
  # Target count of live flow-tracked corners.
  max_corners: int = 4000
  # GPU DIS optical flow; off, cv2 flow. Revisit matching runs on CUDA either way.
  gpu_flow: bool = True

  ransac_threshold: float = 4.5
  # Fraction of the frame the dense flow is computed at.
  flow_scale: float = 0.25

  min_track_length: int = 3
  # Keyframes a landmark's observations may span; 0 is unbounded.
  max_track_span: int = 300
  # Bearing half-angle (rad) past which the essential-matrix RANSAC drops a match.
  feature_max_half_angle: float = float(np.deg2rad(70.0))

  reproj_sigma: float = 0.5

  min_parallax: float = float(np.deg2rad(0.5))
  # Max metric depth (m) for kept landmarks in the final (scaled) triangulation.
  max_landmark_depth: float = 5.0

  # Cap on triangulated landmarks, keeping the longest tracks.
  max_landmarks: int = 4000
  # Cap on observations per track in the joint BA: endpoints plus an even middle.
  max_obs_per_track: int = 8


@dataclass
class AnchorCfg:
  # Static wall-tag ids, each a solved pose; the first names the world frame.
  tag_ids: tuple[int, ...]

  tag_size: float = 0.1

  # On-axis corner-reprojection noise floor (px), widened with corner radius.
  corner_reproj_sigma: float = 1.05


@dataclass
class GripperCfg:
  tag_size: float = 0.01
  # Tool center point (m), the jaw center between the fingertips.
  tcp_in_base_tag: tuple[float, float, float] = (0.1095, 0.0, -0.018)


@dataclass
class SolverCfg:
  max_iterations: int = 30
  velocity_prior_sigma: float = 0.1


@dataclass
class CameraArgs:
  """One camera's ``prepare_camera`` configuration."""

  session: SessionRecording
  slot: str  # the camera's slot in the base recording, e.g. cam00
  anchor: AnchorCfg
  joints: dict[str, RevoluteJoint]
  frontend: FrontendCfg
  gripper: GripperCfg


@dataclass
class Args:
  """One run; the base recording carries the session id, window and camera slots."""

  base: Path
  anchor: AnchorCfg

  output_dir: Path = Path("logs")
  urdf_path: Path | None = None
  # Replace this session's existing estimate and viz layers; off refuses.
  overwrite: bool = False

  frontend: FrontendCfg = field(default_factory=FrontendCfg)
  gripper: GripperCfg = field(default_factory=GripperCfg)
  solver: SolverCfg = field(default_factory=SolverCfg)


def validate(args: Args) -> None:
  """Raise ``ValueError`` when the run configuration breaks an invariant."""
  if args.solver.max_iterations < 1:
    raise ValueError(
      f"--solver.max-iterations must be positive, got {args.solver.max_iterations}"
    )
  if not args.anchor.tag_ids:
    raise ValueError("--anchor.tag-ids must name at least one anchor id")
  if any(t < 0 for t in args.anchor.tag_ids):
    raise ValueError(
      f"--anchor.tag-ids must be non-negative, got {list(args.anchor.tag_ids)}"
    )
  if len(set(args.anchor.tag_ids)) != len(args.anchor.tag_ids):
    raise ValueError(
      f"--anchor.tag-ids must be distinct, got {list(args.anchor.tag_ids)}"
    )
