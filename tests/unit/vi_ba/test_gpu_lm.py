"""``vi_ba.gpu_lm`` against gtsam's elimination and LM, and its native graph passes."""

from dataclasses import astuple

import gtsam
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from specialist.vi_ba._mei_factor_cpp import prune_factors

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.gpu_lm import SparseSystem, optimize_graph
from tests.mei_fixtures import (
  WIDE_CALIBRATED_RADIUS,
  pinhole_mei_params,
  pinned_visual_graph,
  project_point,
)
from tests.solver_fixtures import ScipyFactorization


def _scene(
  n_poses: int = 4, n_points: int = 6
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """A visual-only graph of exact projections with perturbed landmark initials."""
  rng = np.random.default_rng(7)
  params = pinhole_mei_params()
  body_T_lens = gtsam.Pose3()
  poses = [
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.25 * i, 0.0, 0.0)) for i in range(n_poses)
  ]
  points = {
    tid: np.array([0.3 * tid - 0.8, 0.2 * ((tid % 3) - 1), 2.0 + 0.1 * tid])
    for tid in range(n_points)
  }
  tracks = {
    tid: {i: project_point(poses[i], body_T_lens, xyz, params) for i in range(n_poses)}
    for tid, xyz in points.items()
  }
  noisy_points = {tid: xyz + rng.normal(0.0, 0.02, 3) for tid, xyz in points.items()}
  return pinned_visual_graph(
    poses,
    LensSetup(
      body_T_lens=body_T_lens, params=params, calibrated_radius=WIDE_CALIBRATED_RADIUS
    ),
    tracks,
    noisy_points,
  )


def _normal_equation_residual(
  system: SparseSystem, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values
) -> float:
  A, b = system.fill(values)
  H = (A.T @ A).tocsr()
  g = A.T @ b
  ref = graph.linearize(values).optimize()
  delta = np.zeros(system.n_cols)
  for k in system.keys:
    o, d = system.offsets[k]
    delta[o : o + d] = ref.at(k)
  return float(np.linalg.norm(H @ delta - g) / max(np.linalg.norm(g), 1e-12))


def test_assembled_normal_equations_match_gtsam_delta_after_refill() -> None:
  graph, values = _scene()
  system = SparseSystem(graph, values)
  moved = values.retract(graph.linearize(values).optimize())
  for at in (values, moved):
    assert _normal_equation_residual(system, graph, at) < 1e-6


def test_lm_solve_converges_like_gtsam() -> None:
  graph, values = _scene()
  params = gtsam.LevenbergMarquardtParams()
  params.setMaxIterations(25)
  reference = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
  reference.optimize()
  result = optimize_graph(graph, values, 25)
  assert result.iterations > 0
  assert 0.0 < result.final_lambda <= 1e-5
  assert result.final_error < 0.05 * result.initial_error
  assert result.final_error <= reference.error() * 1.01 + 1e-9
  assert float(graph.error(result.values)) == result.final_error


def test_lm_solve_handles_landmark_free_graph() -> None:
  graph = gtsam.NonlinearFactorGraph()
  values = gtsam.Values()
  noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
  target = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.5, 0.0, 0.0))
  graph.push_back(gtsam.PriorFactorPose3(gtsam.symbol("x", 0), gtsam.Pose3(), noise))
  graph.push_back(gtsam.PriorFactorPose3(gtsam.symbol("x", 1), target, noise))
  graph.push_back(
    gtsam.BetweenFactorPose3(gtsam.symbol("x", 0), gtsam.symbol("x", 1), target, noise)
  )
  values.insert(gtsam.symbol("x", 0), gtsam.Pose3())
  values.insert(
    gtsam.symbol("x", 1), gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.3, 0.1, 0.0))
  )
  result = optimize_graph(graph, values, 10)
  assert result.iterations > 0
  assert result.final_error < 0.01 * result.initial_error


def test_threaded_linearization_is_the_per_factor_blocks_with_its_pattern() -> None:
  """Whitened blocks come in graph order, each entry at its factor's key columns."""
  from specialist.vi_ba._mei_factor_cpp import linearize_raveled

  graph, values = _scene()
  system = SparseSystem(graph, values)
  keys = np.asarray(system.keys, dtype=np.int64)
  offsets = np.asarray([system.offsets[k] for k in system.keys], dtype=np.int64)
  lin = graph.linearize(values)
  want_vals: list[np.ndarray] = []
  want_rhs: list[np.ndarray] = []
  want_rows: list[np.ndarray] = []
  want_cols: list[np.ndarray] = []
  rowbase = 0
  for i in range(lin.size()):
    factor = lin.at(i)
    block, b = factor.jacobian()
    col_index = np.concatenate(
      [np.arange(o, o + d) for o, d in (system.offsets[k] for k in factor.keys())]
    )
    at_row, at_col = np.meshgrid(
      np.arange(block.shape[0]) + rowbase, col_index, indexing="ij"
    )
    want_vals.append(block.ravel())
    want_rhs.append(b)
    want_rows.append(at_row.ravel())
    want_cols.append(at_col.ravel())
    rowbase += block.shape[0]
  for threads in (1, 3):
    vals, rhs, rows, cols = linearize_raveled(
      graph, values, keys, offsets, threads, True
    )
    assert np.array_equal(vals, np.concatenate(want_vals))
    assert np.array_equal(rhs, np.concatenate(want_rhs))
    assert np.array_equal(rows, np.concatenate(want_rows))
    assert np.array_equal(cols, np.concatenate(want_cols))
  vals, rhs, rows, cols = linearize_raveled(graph, values, keys, offsets, 2, False)
  assert rows.size == 0 and cols.size == 0
  assert vals.size == sum(block.size for block in want_vals)


def test_threaded_graph_error_is_the_graph_error() -> None:
  from specialist.vi_ba._mei_factor_cpp import graph_error

  graph, values = _scene()
  for threads in (1, 4):
    assert graph_error(graph, values, threads) == graph.error(values)


def test_scatter_delta_splits_the_flat_step_per_key() -> None:
  graph, values = _scene()
  system = SparseSystem(graph, values)
  delta = np.random.default_rng(5).normal(size=system.n_cols)
  scattered = system.scatter(delta)
  for k in system.keys:
    off, dim = system.offsets[k]
    assert np.array_equal(scattered.at(k), delta[off : off + dim])


def test_batched_projection_factors_are_the_per_factor_graph() -> None:
  from specialist.vi_ba._mei_factor_cpp import make_mei_factor_cpp

  from specialist.vi_ba.build import add_mei_factors

  graph, values = _scene()
  params = pinhole_mei_params()
  lens = LensSetup(
    body_T_lens=gtsam.Pose3(), params=params, calibrated_radius=WIDE_CALIBRATED_RADIUS
  )
  noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.5)
  rng = np.random.default_rng(3)
  pose_keys = [int(k) for k in values.keys() if gtsam.Symbol(k).chr() == ord("x")]
  point_keys = [int(k) for k in values.keys() if gtsam.Symbol(k).chr() != ord("x")]
  assert pose_keys and point_keys
  rows = [
    (int(rng.choice(pose_keys)), int(rng.choice(point_keys)), rng.uniform(100, 500, 2))
    for _ in range(40)
  ]
  one = gtsam.NonlinearFactorGraph()
  for pose_key, point_key, pixel in rows:
    one.push_back(
      make_mei_factor_cpp(
        pose_key, point_key, pixel, noise, lens.body_T_lens, np.array(astuple(params))
      )
    )
  batch = gtsam.NonlinearFactorGraph()
  add_mei_factors(
    batch, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], noise, lens
  )
  assert batch.size() == one.size() == 40
  one_system, batch_system = SparseSystem(one, values), SparseSystem(batch, values)
  jac_one, rhs_one = one_system.fill(values)
  jac_batch, rhs_batch = batch_system.fill(values)
  assert np.array_equal(jac_one.toarray(), jac_batch.toarray())
  assert np.array_equal(rhs_one, rhs_batch)
  assert one.error(values) == batch.error(values)


def test_native_pruning_drops_exactly_the_factors_touching_dropped_keys() -> None:
  graph, values = _scene()
  keys = [int(k) for k in values.keys()]
  drop = {keys[1], keys[-1]}
  pruned = prune_factors(graph, np.fromiter(drop, dtype=np.int64))
  want = [
    i for i in range(graph.size()) if not any(k in drop for k in graph.at(i).keys())
  ]
  assert pruned.size() == len(want)
  for j, i in enumerate(want):
    assert list(pruned.at(j).keys()) == list(graph.at(i).keys())
    assert pruned.at(j).error(values) == graph.at(i).error(values)
  assert 0 < pruned.size() < graph.size()


def test_state_keyframes_follow_the_key_index() -> None:
  from specialist.vi_ba.compose import CameraKeys

  keys = CameraKeys(["cam00", "cam01"])
  values = gtsam.Values()
  values.insert(keys.pose_key("cam01", 7), gtsam.Pose3())
  values.insert(keys.velocity_key("cam00", 9), np.zeros(3))
  values.insert(keys.front_landmark_key("cam00", 3), gtsam.Point3(0, 0, 1))
  values.insert(keys.anchor_key(0), gtsam.Pose3())
  system = SparseSystem(gtsam.NonlinearFactorGraph(), values)
  by_key = {
    int(k): system.state_keyframes[np.searchsorted(system.state_cols, off)]
    for k, (off, _) in system.offsets.items()
    if off in set(system.state_cols.tolist())
  }
  assert by_key[keys.pose_key("cam01", 7)] == 7
  assert by_key[keys.velocity_key("cam00", 9)] == 9
  assert by_key[keys.anchor_key(0)] == -1
  assert len(system.state_keyframes) == len(system.state_cols) == 6 + 3 + 6


def test_normal_equation_blocks_are_canonical_csr() -> None:
  from specialist.vi_ba.gpu_lm import _NormalEquations

  graph, values = _scene()
  system = SparseSystem(graph, values)
  jacobian, rhs = system.fill(values)
  host = _NormalEquations(jacobian, rhs, system.landmark_cols, system.state_cols)
  assert host.Hss.has_canonical_format
  assert host.Hsl.has_canonical_format
  assert host.Hls.has_canonical_format


def test_schur_pcg_solves_the_reduced_state_system() -> None:
  from specialist.vi_ba.gpu_lm import _NormalEquations, _SchurPcg

  graph, values = _scene()
  system = SparseSystem(graph, values)
  jacobian, rhs = system.fill(values)
  normal = _NormalEquations(jacobian, rhs, system.landmark_cols, system.state_cols)
  schur = _SchurPcg(normal, system.state_keyframes, ScipyFactorization())
  lam = 1e-3
  got, iterations = schur.solve(lam)
  Hll_inv = normal.block_inverse(lam)
  S = normal.Hss - normal.Hsl @ Hll_inv @ normal.Hls + lam * sp.identity(len(got))
  want = spl.spsolve(S.tocsc(), normal.gs - normal.Hsl @ (Hll_inv @ normal.gl))
  np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9)
  assert iterations <= 3


def test_block_inverse_gives_an_unobserved_direction_no_step() -> None:
  from specialist.vi_ba.gpu_lm import _NormalEquations

  # A rank-2 block with a crashed solve's eigenvalues; its null one rounded negative.
  v, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
  crashed = v @ np.diag([-1.25e-9, 2.3e7, 3.07e7]) @ v.T
  full = np.diag([4.0, 5.0, 6.0]) * 1e6
  normal = _NormalEquations.__new__(_NormalEquations)
  normal.blocks = np.stack([crashed, full])
  got = normal.block_inverse(1e-9).toarray()
  assert np.isfinite(got).all()
  np.testing.assert_allclose(got[:3, :3] @ v[:, 0], 0.0, atol=1e-20)
  np.testing.assert_allclose(got[:3, :3] @ v[:, 2], v[:, 2] / 3.07e7, rtol=1e-9)
  want = np.linalg.inv(full + 1e-9 * np.eye(3))
  np.testing.assert_allclose(got[3:, 3:], want, rtol=1e-12)
