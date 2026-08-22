"""Rotation consensus and the quaternion/matrix conversions, half turns included."""

import gtsam
import numpy as np
from specialist_dataset.poses import quats_to_matrices

from specialist.vi_ba.geometry import (
  consensus_pose,
  matrices_to_quats,
  quat_xyzw,
)


def _pose(rx: float, ry: float, rz: float, x: float, y: float, z: float) -> gtsam.Pose3:
  return gtsam.Pose3(gtsam.Rot3.Rodrigues(rx, ry, rz), gtsam.Point3(x, y, z))


def test_consensus_keeps_the_agreeing_majority() -> None:
  truth = _pose(0.1, -0.05, 0.2, 0.3, 0.4, 1.0)
  rng = np.random.default_rng(0)
  poses = [
    gtsam.Pose3(truth.rotation(), truth.translation() + rng.normal(0, 1e-3, 3))
    for _ in range(20)
  ]
  poses += [_pose(1.4, 0.9, -1.2, 0.9, -0.4, 1.6) for _ in range(6)]
  agreed = consensus_pose(poses)
  assert agreed is not None
  assert np.allclose(agreed.matrix(), truth.matrix(), atol=5e-3)


def test_consensus_refuses_a_split_pool() -> None:
  """Two modes of equal weight name two different tags, not one averaged one."""
  a = _pose(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
  b = _pose(0.8, -0.6, 0.4, 0.5, 0.5, 1.4)
  assert consensus_pose([a] * 10 + [b] * 10) is None


def test_consensus_of_nothing_is_nothing() -> None:
  assert consensus_pose([]) is None


def test_quat_xyzw_is_scipy_order() -> None:
  rotation = gtsam.Rot3.Rz(np.pi / 3)
  quat = quat_xyzw(rotation)
  np.testing.assert_allclose(quat, [0.0, 0.0, np.sin(np.pi / 6), np.cos(np.pi / 6)])
  np.testing.assert_allclose(
    quats_to_matrices(quat[None, :])[0], rotation.matrix(), atol=1e-12
  )


def test_half_turns_round_trip() -> None:
  matrices = np.stack(
    [gtsam.Rot3.Rx(np.pi).matrix(), gtsam.Rot3.Ry(np.pi).matrix(), np.eye(3)]
  )
  quats = matrices_to_quats(matrices)
  assert np.all(np.isfinite(quats))
  np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-12)
  np.testing.assert_allclose(quats_to_matrices(quats), matrices, atol=1e-12)


def test_random_rotations_round_trip() -> None:
  rng = np.random.default_rng(0)
  matrices = np.stack(
    [gtsam.Rot3.Expmap(rng.normal(size=3) * 2.0).matrix() for _ in range(16)]
  )
  np.testing.assert_allclose(
    quats_to_matrices(matrices_to_quats(matrices)), matrices, atol=1e-12
  )


def test_the_two_quaternion_encoders_agree_in_sign() -> None:
  """Both encoders resolve the ``q``/``-q`` double cover to ``w >= 0``."""
  rng = np.random.default_rng(20260820)
  quats = rng.normal(size=(5000, 4))
  quats /= np.linalg.norm(quats, axis=1, keepdims=True)
  matrices = quats_to_matrices(quats)
  from_scipy = matrices_to_quats(matrices)
  from_gtsam = np.stack([quat_xyzw(gtsam.Rot3(m)) for m in matrices])
  assert (from_scipy[:, 3] >= 0.0).all()
  assert (from_gtsam[:, 3] >= 0.0).all()
  assert np.abs(from_scipy - from_gtsam).max() < 1e-12
