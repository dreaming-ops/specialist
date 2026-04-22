"""GTSAM factor helpers with diagonal SE(3) noise, rotation first: ``[ω, v]``."""

from __future__ import annotations

import gtsam
import numpy as np


def _sigmas(
  translation_sigma: float, rotation_sigma: float, huber_k: float | None = None
) -> gtsam.noiseModel.Base:
  sigmas = np.array(
    [
      rotation_sigma,
      rotation_sigma,
      rotation_sigma,
      translation_sigma,
      translation_sigma,
      translation_sigma,
    ],
    dtype=np.float64,
  )
  base = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
  if huber_k is None:
    return base
  if huber_k < 0:
    # Negative encodes Cauchy: |huber_k| is the Cauchy k.
    return gtsam.noiseModel.Robust.Create(
      gtsam.noiseModel.mEstimator.Cauchy.Create(-huber_k), base
    )
  return gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(huber_k), base
  )


def make_anchor_prior(
  anchor_key: int,
  anchor_pose: gtsam.Pose3,
  translation_sigma: float,
  rotation_sigma: float,
) -> gtsam.PriorFactorPose3:
  """Tight-σ prior pinning the anchor-tag pose — the gauge fix for the graph."""
  return gtsam.PriorFactorPose3(
    anchor_key, anchor_pose, _sigmas(translation_sigma, rotation_sigma)
  )


def make_odometry_factor(
  key_i: int,
  key_j: int,
  relative_pose: gtsam.Pose3,
  translation_sigma: float,
  rotation_sigma: float,
) -> gtsam.BetweenFactorPose3:
  """BetweenFactor on consecutive tracker keyframes from a noisy relative-pose measurement."""
  return gtsam.BetweenFactorPose3(
    key_i,
    key_j,
    relative_pose,
    _sigmas(translation_sigma, rotation_sigma),
  )


def make_anchor_observation_factor(
  tracker_key: int,
  anchor_key: int,
  camera_to_anchor: gtsam.Pose3,
  camera_extrinsic: gtsam.Pose3,
  translation_sigma: float,
  rotation_sigma: float,
  huber_k: float | None = None,
) -> gtsam.BetweenFactorPose3:
  """BetweenFactor from ``tracker_key`` to ``anchor_key`` for one camera observation."""
  tracker_to_anchor = camera_extrinsic.compose(camera_to_anchor)
  return gtsam.BetweenFactorPose3(
    tracker_key,
    anchor_key,
    tracker_to_anchor,
    _sigmas(translation_sigma, rotation_sigma, huber_k=huber_k),
  )


def make_cross_tracker_factor(
  observer_key: int,
  tag_owner_key: int,
  camera_to_tag: gtsam.Pose3,
  observer_camera_extrinsic: gtsam.Pose3,
  tag_on_owner_extrinsic: gtsam.Pose3,
  translation_sigma: float,
  rotation_sigma: float,
  huber_k: float | None = None,
) -> gtsam.BetweenFactorPose3:
  """BetweenFactor linking two cameras via one tracker observing the other's tag."""
  observer_to_owner = observer_camera_extrinsic.compose(camera_to_tag).compose(
    tag_on_owner_extrinsic.inverse()
  )
  return gtsam.BetweenFactorPose3(
    observer_key,
    tag_owner_key,
    observer_to_owner,
    _sigmas(translation_sigma, rotation_sigma, huber_k=huber_k),
  )
