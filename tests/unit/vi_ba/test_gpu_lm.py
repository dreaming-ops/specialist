"""``vi_ba.gpu_lm`` against gtsam's elimination and LM, and its dispatch fallback."""

from __future__ import annotations

import gtsam
import numpy as np
import scipy.sparse.linalg as spl

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.gpu_lm import SparseSystem, _lm
from specialist.vi_ba.optimize import optimize_graph
from tests.mei_fixtures import pinhole_mei_params, pinned_visual_graph, project_point


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
    poses, LensSetup(body_T_lens=body_T_lens, params=params), tracks, noisy_points
  )


def _scipy_solver() -> tuple:
  return (lambda S, rhs: spl.spsolve(S.tocsc(), rhs)), (lambda: None)


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
  reference = optimize_graph(graph, values, max_iterations=25)
  vals, iters, initial_error, final_error = _lm(
    graph, values, 25, 1e-4, solver_factory=_scipy_solver
  )
  assert iters > 0
  assert final_error < 0.05 * initial_error
  assert final_error <= reference.final_error * 1.01 + 1e-9
  assert float(graph.error(vals)) == final_error


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
  vals, iters, initial_error, final_error = _lm(
    graph, values, 10, 1e-4, solver_factory=_scipy_solver
  )
  assert iters > 0
  assert final_error < 0.01 * initial_error


def test_optimize_graph_gpu_dispatch_returns_valid_result() -> None:
  """Runs the real GPU solve where the backend is installed, else the gtsam fallback."""
  graph, values = _scene()
  reference = optimize_graph(graph, values, max_iterations=10)
  result = optimize_graph(graph, values, max_iterations=10, gpu="auto")
  assert result.iterations > 0
  assert result.final_error <= reference.final_error * 1.01 + 1e-9


def test_require_mode_raises_when_backend_missing() -> None:
  """``gpu="require"`` raises on an unloadable backend."""
  graph, values = _scene()
  try:
    from specialist.vi_ba.gpu_lm import assert_backend

    assert_backend()
  except Exception:
    import pytest

    with pytest.raises(RuntimeError, match="gpu solver required"):
      optimize_graph(graph, values, max_iterations=5, gpu="require")
    return
  result = optimize_graph(graph, values, max_iterations=5, gpu="require")
  assert result.final_error <= result.initial_error
