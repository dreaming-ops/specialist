"""Schur-complement Levenberg-Marquardt, the state system factorized by cuDSS."""

from __future__ import annotations

from typing import Any, Callable

import gtsam
import numpy as np
import scipy.sparse as sp

__all__ = ["SparseSystem", "assert_backend", "_lm"]

# gtsam's LevenbergMarquardtParams defaults.
_LAMBDA_INITIAL = 1e-5
_LAMBDA_FACTOR = 10.0
_LAMBDA_UPPER_BOUND = 1e5

_LANDMARK_CHAR = "l"
_LANDMARK_DIM = 3


class SparseSystem:
  """Fixed-pattern assembly of the whitened Jacobian over one LM call."""

  def __init__(self, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values):
    self._graph = graph
    self.keys = [int(k) for k in values.keys()]
    zero = values.zeroVectors()
    self.offsets: dict[int, tuple[int, int]] = {}
    landmark_cols: list[int] = []
    state_cols: list[int] = []
    off = 0
    for k in self.keys:
      dim = zero.at(k).shape[0]
      self.offsets[k] = (off, dim)
      is_landmark = chr(gtsam.Symbol(k).chr()) == _LANDMARK_CHAR
      (landmark_cols if is_landmark else state_cols).extend(range(off, off + dim))
      off += dim
    self.n_cols = off
    self.landmark_cols = np.asarray(landmark_cols, dtype=np.intp)
    self.state_cols = np.asarray(state_cols, dtype=np.intp)
    self._pattern: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    self._n_rows = 0

  def fill(self, values: gtsam.Values) -> tuple[sp.csr_matrix, np.ndarray]:
    lin = self._graph.linearize(values)
    if self._pattern is None:
      self._build_pattern(lin)
    return self._assemble(lin)

  def _build_pattern(self, lin: gtsam.GaussianFactorGraph) -> None:
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    rowbase = 0
    for i in range(lin.size()):
      f = lin.at(i)
      nr = f.jacobian()[0].shape[0]
      col_index = np.concatenate(
        [np.arange(off, off + dim) for off, dim in (self.offsets[k] for k in f.keys())]
      )
      rr, cc = np.meshgrid(np.arange(nr) + rowbase, col_index, indexing="ij")
      rows.append(rr.ravel())
      cols.append(cc.ravel())
      rowbase += nr
    self._n_rows = rowbase
    # Converting a triplet index matrix the same way gives the CSR gather permutation.
    row_cat = np.concatenate(rows)
    col_cat = np.concatenate(cols)
    order = sp.coo_matrix(
      (np.arange(len(row_cat), dtype=np.float64), (row_cat, col_cat)),
      shape=(self._n_rows, self.n_cols),
    ).tocsr()
    self._pattern = (order.data.astype(np.intp), order.indices, order.indptr)

  def _assemble(
    self, lin: gtsam.GaussianFactorGraph
  ) -> tuple[sp.csr_matrix, np.ndarray]:
    assert self._pattern is not None
    perm, indices, indptr = self._pattern
    vals = np.empty(len(perm))
    rhs = np.zeros(self._n_rows)
    rowbase = 0
    pos = 0
    for i in range(lin.size()):
      A, b = lin.at(i).jacobian()
      nr = A.shape[0]
      rhs[rowbase : rowbase + nr] = b
      vals[pos : pos + A.size] = A.ravel()
      rowbase += nr
      pos += A.size
    jacobian = sp.csr_matrix(
      (vals[perm], indices, indptr), shape=(self._n_rows, self.n_cols)
    )
    return jacobian, rhs

  def scatter(self, delta: np.ndarray) -> gtsam.VectorValues:
    """Split a flat tangent vector into per-key ``VectorValues``."""
    delta_values = gtsam.VectorValues()
    for k in self.keys:
      off, dim = self.offsets[k]
      delta_values.insert(k, delta[off : off + dim])
    return delta_values


# A factorization plan is reusable only while the Schur pattern matches.
SolverFactory = Callable[
  [], tuple[Callable[[sp.csr_matrix, np.ndarray], np.ndarray], Callable[[], None]]
]


def _cudss_solver() -> tuple[
  Callable[[sp.csr_matrix, np.ndarray], np.ndarray], Callable[[], None]
]:
  import cupy
  import cupyx.scipy.sparse as cpsp
  from nvmath.sparse.advanced import DirectSolver

  solver: Any = None
  dev_matrix: Any = None
  dev_rhs: Any = None
  pattern: tuple[np.ndarray, np.ndarray] | None = None

  def solve(S: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
    nonlocal solver, dev_matrix, dev_rhs, pattern
    if (
      solver is None
      or pattern is None
      or not (
        np.array_equal(pattern[0], S.indptr) and np.array_equal(pattern[1], S.indices)
      )
    ):
      if solver is not None:
        solver.free()
      dev_matrix = cpsp.csr_matrix(S)
      dev_rhs = cupy.asarray(rhs).reshape(-1, 1)
      solver = DirectSolver(dev_matrix, dev_rhs)
      solver.plan()
      pattern = (S.indptr.copy(), S.indices.copy())
    else:
      dev_matrix.data.set(S.data)
      dev_rhs.set(rhs.reshape(-1, 1))
    solver.factorize()
    x = solver.solve()
    cupy.cuda.Stream.null.synchronize()
    host: np.ndarray = cupy.asnumpy(x)
    return host.ravel()

  def close() -> None:
    if solver is not None:
      solver.free()

  return solve, close


def assert_backend() -> None:
  """Import the CUDA solve stack, raising when any piece is missing."""
  _solve, close = _cudss_solver()
  close()


def _lm(
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
  max_iterations: int,
  relative_error_tol: float,
  solver_factory: SolverFactory = _cudss_solver,
  initial_error: float | None = None,
) -> tuple[gtsam.Values, int, float, float]:
  """``(values, iterations, initial_error, final_error)`` of the Schur LM loop."""
  system = SparseSystem(graph, values)
  solve, close = solver_factory()
  landmark_cols, state_cols = system.landmark_cols, system.state_cols
  n_landmarks = len(landmark_cols) // _LANDMARK_DIM
  eye3 = np.eye(_LANDMARK_DIM)
  eye_state = sp.identity(len(state_cols), format="csr")
  lam = _LAMBDA_INITIAL
  err = float(graph.error(values)) if initial_error is None else initial_error
  initial_error = err
  iters = 0

  try:
    while iters < max_iterations:
      jacobian, rhs = system.fill(values)
      H = (jacobian.T @ jacobian).tocsr()
      g = jacobian.T @ rhs
      H_state = H[state_cols]
      Hll = H[landmark_cols][:, landmark_cols]
      Hsl = H_state[:, landmark_cols]
      Hss = H_state[:, state_cols]
      gl, gs = g[landmark_cols], g[state_cols]
      # Landmarks are mutually unconnected, so Hll is block-diagonal in 3x3 blocks.
      Hll_coo = Hll.tocoo()
      Hll_blocks = np.zeros((n_landmarks, _LANDMARK_DIM, _LANDMARK_DIM))
      Hll_blocks[
        Hll_coo.row // _LANDMARK_DIM,
        Hll_coo.row % _LANDMARK_DIM,
        Hll_coo.col % _LANDMARK_DIM,
      ] = Hll_coo.data

      accepted = False
      while lam <= _LAMBDA_UPPER_BOUND:
        block_inv = np.linalg.inv(Hll_blocks + lam * eye3)
        Hll_inv = sp.bsr_matrix(
          (block_inv, np.arange(n_landmarks), np.arange(n_landmarks + 1)),
          shape=Hll.shape,
        ).tocsr()
        S = Hss - Hsl @ Hll_inv @ Hsl.T
        S = 0.5 * (S + S.T) + lam * eye_state
        rhs_state = gs - Hsl @ (Hll_inv @ gl)
        x_state = solve(S, rhs_state)

        delta = np.zeros(system.n_cols)
        delta[state_cols] = x_state
        delta[landmark_cols] = Hll_inv @ (gl - Hsl.T @ x_state)
        candidate = values.retract(system.scatter(delta))
        candidate_error = float(graph.error(candidate))
        if candidate_error < err:
          relative_decrease = (err - candidate_error) / max(err, 1e-12)
          values = candidate
          err = candidate_error
          # Floor keeps lambda positive; gtsam's default lower bound is 0.
          lam = max(lam / _LAMBDA_FACTOR, 1e-10)
          iters += 1
          accepted = True
          if relative_decrease < relative_error_tol:
            return values, iters, initial_error, err
          break
        lam *= _LAMBDA_FACTOR
      if not accepted:
        break
  finally:
    close()
  return values, iters, initial_error, err
