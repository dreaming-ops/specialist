"""Nested tyro config dataclasses for ``track.py`` and ``prepare_camera``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrontendCfg:
  orb_n_features: int = 1000
  ransac_threshold: float = 4.5
  min_track_length: int = 3
  max_half_angle: float = float(np.deg2rad(70.0))
  reproj_sigma: float = 2.25
  mask_tag_pad: int = 45
  min_parallax: float = float(np.deg2rad(0.5))
  # Max metric depth (m) for kept landmarks in the final (scaled) triangulation.
  max_landmark_depth: float = 5.0
  mask_rear_rects: tuple[tuple[int, int, int, int], ...] = ((728, 1425, 2153, 2880),)
  init_min_shared_tracks: int = 60
  # Re-match ORB features between spatially close, non-adjacent keyframes.
  revisit: bool = True
  loop_closure_max_dist: float = 0.1
  loop_closure_min_kf_gap: int = 15
  # Run loop closure in anchor-seeded mode as well as under visual odometry.
  anchored_revisit: bool = True
  # Shape the trajectory by front-lens SfM; the anchor enters only as a graph factor.
  use_visual_odometry: bool = True
  # Cap on triangulated landmarks per lens in VO mode, keeping the longest tracks.
  vo_max_landmarks_per_lens: int = 400
  # Cap on BA observations per track in VO mode: endpoints plus evenly spaced middles.
  vo_max_obs_per_track: int = 8


@dataclass
class ImuCfg:
  # IMU noise sigmas (rad/s/sqrtHz, m/s2/sqrtHz), a few times the measured floor.
  gyro_noise_sigma: float = 0.0125
  accel_noise_sigma: float = 0.01
  gyro_random_walk: float = 1.0e-7
  accel_random_walk: float = 1.0e-7
  body_R_imu: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
  )


@dataclass
class AnchorCfg:
  tag_id: int
  tag_size: float = 0.1
  # The anchor's four metric corners reproject into the bundle and pin absolute scale.
  corner_reproj_sigma: float = 1.5


@dataclass
class GripperCfg:
  tag_size: float = 0.01
  # Tool center point (m), the jaw center between the fingertips.
  tcp_in_base_tag: tuple[float, float, float] = (0.1095, 0.0, -0.018)
  # Finger-angle sigmas: corner reprojection (px) and left-right symmetry prior (rad).
  finger_reproj_sigma: float = 1.5
  finger_symmetry_sigma: float = 0.05
  # Base-tag pose sigmas: corner reprojection (px) and loose PnP-seeded prior (rad, m).
  base_tag_reproj_sigma: float = 1.5
  base_tag_prior_sigma_rot: float = 0.3
  base_tag_prior_sigma_trans: float = 0.05


@dataclass
class SolverCfg:
  max_iterations: int = 30
  velocity_prior_sigma: float = 0.1


@dataclass
class WindowCfg:
  # Seconds of clip to process; None = the full capture.
  duration: float | None
  master_start: float = 0.0
  # Keyframe stride in frames, a multiple of the ~0.1 s lattice; 0 picks the lattice.
  keyframe_every: int = 0
