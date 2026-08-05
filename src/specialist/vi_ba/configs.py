"""Nested tyro config dataclasses for ``track.py`` and ``prepare_camera``."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
  "AnchorCfg",
  "Args",
  "FrontendCfg",
  "GripperCfg",
  "ImuCfg",
  "SolverCfg",
  "WindowCfg",
  "validate",
]


@dataclass
class FrontendCfg:
  orb_n_features: int = 4000
  # Run DIS optical flow on the GPU, falling back loudly to cv2 without torch/CUDA.
  gpu_flow: bool = True

  ransac_threshold: float = 4.5
  # Fraction of the frame the dense flow is computed at.
  flow_scale: float = 0.25

  min_track_length: int = 3
  max_half_angle: float = float(np.deg2rad(70.0))
  reproj_sigma: float = 0.5
  # Padding (px) around a masked tag's detected box.
  mask_tag_pad: int = 45
  # Exclude non-anchor tags (the grippers' own plates) from feature detection.
  mask_tags: bool = True

  min_parallax: float = float(np.deg2rad(0.5))
  # Max metric depth (m) for kept landmarks in the final (scaled) triangulation.
  max_landmark_depth: float = 5.0

  mask_rear_rects: tuple[tuple[int, int, int, int], ...] = ((728, 1425, 2153, 2880),)
  init_min_shared_tracks: int = 30
  vo_min_registered_ratio: float = 0.7
  # Run the rear lens's landmark frontend; tag detection runs regardless.
  rear_landmarks: bool = False
  # Re-match ORB features between nearby non-adjacent keyframes to rejoin broken tracks.
  revisit: bool = True

  loop_closure_max_dist: float = 0.1
  loop_closure_min_kf_gap: int = 15
  # Triangulate the revisit landmarks from the anchor-covisible views alone.
  anchored_revisit: bool = True
  # Cap on triangulated landmarks per lens, keeping the longest tracks.
  max_landmarks_per_lens: int = 4000
  # Cap on observations per track in the joint BA: endpoints plus an even middle.
  max_obs_per_track: int = 8
  # Per-thread AprilTag detectors so concurrent tag walks overlap.
  parallel_tag_detect: bool = True


@dataclass
class ImuCfg:
  body_R_imu: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
  )


@dataclass
class AnchorCfg:
  # Static wall-tag ids; the first is the reference anchor that pins the world gauge.
  tag_ids: tuple[int, ...]
  tag_size: float = 0.1

  # The anchor's four metric corners reproject into the bundle and pin absolute scale.
  corner_reproj_sigma: float = 1.5


@dataclass
class GripperCfg:
  tag_size: float = 0.01
  # Tool center point (m), the jaw center between the fingertips.
  tcp_in_base_tag: tuple[float, float, float] = (0.1095, 0.0, -0.018)


@dataclass
class SolverCfg:
  max_iterations: int = 30
  velocity_prior_sigma: float = 0.1
  # Prefer the Schur/cuDSS GPU solver; falls back to the gtsam optimizer.
  gpu: bool = True


@dataclass
class WindowCfg:
  # Seconds of clip to process; None = the full capture.
  duration: float | None
  master_start: float = 0.0
  # Keyframe stride in frames, a multiple of the ~0.1 s lattice; 0 picks the lattice.
  keyframe_every: int = 0


@dataclass
class Args:
  """One run's full configuration: inputs, window, tag ids and per-stage knobs."""

  input_insvs: tuple[Path, ...]
  window: WindowCfg
  anchor: AnchorCfg
  output_dir: Path = Path("logs")
  slots: tuple[str, ...] = ()  # auto-derive ("tracker_0", "tracker_1", ...)
  # The base recording's id, shared by the track layer; a fresh uuid when unset.
  session_id: str | None = None
  urdf_path: Path | None = None
  open_viewer: bool = True
  # One id per tracker; cross-tracker disambiguation needs distinct ids.
  base_tag_ids: tuple[int, ...] = ()
  # Per-tracker finger tag ids; 0 disables that tracker's finger.
  left_finger_tag_ids: tuple[int, ...] = ()
  right_finger_tag_ids: tuple[int, ...] = ()
  frontend: FrontendCfg = field(default_factory=FrontendCfg)
  imu: ImuCfg = field(default_factory=ImuCfg)
  gripper: GripperCfg = field(default_factory=GripperCfg)
  solver: SolverCfg = field(default_factory=SolverCfg)


def validate(args: Args) -> None:
  """Raise ``ValueError`` when the run configuration breaks an invariant."""
  if not args.input_insvs:
    raise ValueError("at least one .insv input required")
  if args.solver.max_iterations < 1:
    raise ValueError(
      f"--solver.max-iterations must be positive, got {args.solver.max_iterations}"
    )
  if not args.base_tag_ids:
    raise ValueError(
      f"--base-tag-ids must be supplied (one per input, got "
      f"{len(args.input_insvs)} inputs)"
    )
  for field_name, ids in (
    ("base_tag_ids", args.base_tag_ids),
    ("left_finger_tag_ids", args.left_finger_tag_ids),
    ("right_finger_tag_ids", args.right_finger_tag_ids),
  ):
    if ids and len(ids) != len(args.input_insvs):
      raise ValueError(
        f"{field_name} length ({len(ids)}) must match "
        f"input_insvs ({len(args.input_insvs)})"
      )
  if len(set(args.base_tag_ids)) != len(args.base_tag_ids):
    raise ValueError(
      f"--base-tag-ids must be distinct per tracker for cross-tracker "
      f"disambiguation, got {list(args.base_tag_ids)}"
    )
  if not args.anchor.tag_ids:
    raise ValueError("--anchor.tag-ids must name at least the reference anchor id")
  if len(set(args.anchor.tag_ids)) != len(args.anchor.tag_ids):
    raise ValueError(
      f"--anchor.tag-ids must be distinct, got {list(args.anchor.tag_ids)}"
    )
  _validate_tag_id_disjointness(args)


def _validate_tag_id_disjointness(args: Args) -> None:
  """Anchor, base and nonzero finger tag ids must be pairwise disjoint."""
  groups = (
    ("--anchor.tag-ids", set(args.anchor.tag_ids)),
    ("--base-tag-ids", {int(i) for i in args.base_tag_ids}),
    (
      "--left/right-finger-tag-ids",
      {
        int(i)
        for i in (*args.left_finger_tag_ids, *args.right_finger_tag_ids)
        if int(i) > 0
      },
    ),
  )
  for (name_a, ids_a), (name_b, ids_b) in itertools.combinations(groups, 2):
    overlap = ids_a & ids_b
    if overlap:
      raise ValueError(
        f"tag ids {sorted(overlap)} appear in both {name_a} and "
        f"{name_b}; anchor, base, and finger tags must be disjoint"
      )
