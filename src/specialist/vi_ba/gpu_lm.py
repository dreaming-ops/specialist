"""Schur-complement Levenberg-Marquardt, the state system solved by conjugate
gradients preconditioned with a cuDSS factorization of its banded truncation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import gtsam
import numpy as np
import scipy.sparse as sp

from specialist.vi_ba._mei_factor_cpp import (
  graph_error,
  linearize_raveled,
  scatter_delta,
)
from specialist.vi_ba.features import cpu_budget
from specialist.vi_ba.symbols import (
  LANDMARK_CHAR,
  MAX_KEYFRAMES_PER_CAMERA,
  PER_KEYFRAME_CHARS,
)

__all__ = [
  "OptimizeResult",
  "SparseSystem",
  "assert_backend",
  "optimize_graph",
]

# gtsam's LevenbergMarquardtParams defaults.
_LAMBDA_INITIAL = 1e-5
_LAMBDA_FACTOR = 10.0
_LAMBDA_UPPER_BOUND = 1e5
# Stop once an accepted step improves the error by less than this fraction.
_RELATIVE_ERROR_TOL = 1e-4

_LANDMARK_DIM = 3

# Threads the factor-graph passes (linearization, error) split over.
_GRAPH_THREADS = 16


class SparseSystem:
  """Fixed-pattern assembly of the whitened Jacobian over one LM call."""

  def __init__(self, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values):
    self._graph = graph
    self.keys = [int(k) for k in values.keys()]
    zero = values.zeroVectors()
    self.offsets: dict[int, tuple[int, int]] = {}
    landmark_cols: list[int] = []
    state_cols: list[int] = []
    keyframes: list[int] = []
    off = 0
    for k in self.keys:
      dim = zero.at(k).shape[0]
      self.offsets[k] = (off, dim)
      symbol = gtsam.Symbol(k)
      char = chr(symbol.chr())
      if char == LANDMARK_CHAR:
        landmark_cols.extend(range(off, off + dim))
      else:
        state_cols.extend(range(off, off + dim))
        kf = (
          symbol.index() % MAX_KEYFRAMES_PER_CAMERA
          if char in PER_KEYFRAME_CHARS
          else -1
        )
        keyframes.extend([kf] * dim)
      off += dim
    self.n_cols = off
    self.landmark_cols = np.asarray(landmark_cols, dtype=np.intp)
    self.state_cols = np.asarray(state_cols, dtype=np.intp)
    # Per state column, its keyframe; -1 for a variable shared by the whole session.
    self.state_keyframes = np.asarray(keyframes, dtype=np.int64)
    self._key_array = np.asarray(self.keys, dtype=np.int64)
    self._offset_array = np.asarray(
      [self.offsets[k] for k in self.keys], dtype=np.int64
    ).reshape(-1, 2)
    self._threads = min(_GRAPH_THREADS, cpu_budget())
    self._pattern: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    self._n_rows = 0

  def fill(self, values: gtsam.Values) -> tuple[sp.csr_matrix, np.ndarray]:
    vals, rhs, rows, cols = linearize_raveled(
      self._graph,
      values,
      self._key_array,
      self._offset_array,
      self._threads,
      self._pattern is None,
    )
    if self._pattern is None:
      self._n_rows = len(rhs)
      # A triplet index matrix through the same conversion is the gather permutation.
      order = sp.coo_matrix(
        (np.arange(len(rows), dtype=np.float64), (rows, cols)),
        shape=(self._n_rows, self.n_cols),
      ).tocsr()
      self._pattern = (order.data.astype(np.intp), order.indices, order.indptr)
    perm, indices, indptr = self._pattern
    jacobian = sp.csr_matrix(
      (vals[perm], indices, indptr), shape=(self._n_rows, self.n_cols)
    )
    return jacobian, rhs

  def error(self, values: gtsam.Values) -> float:
    """``graph.error(values)``, evaluated on threads."""
    return float(graph_error(self._graph, values, self._threads))

  def scatter(self, delta: np.ndarray) -> gtsam.VectorValues:
    """Split a flat tangent vector into per-key ``VectorValues``."""
    return scatter_delta(self._key_array, self._offset_array, delta)


# Conjugate gradients: residual reached, iteration cap, preconditioner's keyframe span.
_PCG_TOL = 1e-4
_PCG_MAX_ITER = 500
_PRECONDITIONER_BAND = 300
# Row chunks of the Schur product: at least this many, each under this many entries.
_SCHUR_CHUNKS = 32
_SCHUR_CHUNK_NNZ = 1 << 20


class Factorization(Protocol):
  """A sparse SPD factorization in one array library, applied to its vectors."""

  xp: Any
  xsp: Any

  def factorize(self, S: sp.csr_matrix) -> None: ...

  def apply(self, rhs: Any) -> Any: ...

  def to_host(self, array: Any) -> np.ndarray: ...

  def close(self) -> None: ...


def _cudss_threading_layer() -> str | None:
  """The path of the cuDSS wheel's OpenMP threading layer, or ``None``."""
  try:
    import nvidia
  except ImportError:
    return None
  for root in nvidia.__path__:
    found = sorted(Path(root).glob("cu*/lib/libcudss_mtlayer_gomp.so*"))
    if found:
      return str(found[0])
  return None


class _Cudss:
  """cuDSS on the device: Cholesky of a host CSR matrix, LU once it refuses."""

  def __init__(self) -> None:
    import cupy
    import cupyx.scipy.sparse as cpsp
    from nvmath.sparse import advanced

    self.xp, self.xsp, self._adv = cupy, cpsp, advanced
    self._solver: Any = None
    self._matrix: Any = None
    self._rhs: Any = None
    self._pattern: tuple[np.ndarray, np.ndarray] | None = None
    self._general = False
    self._threading = _cudss_threading_layer()
    free, total = cupy.cuda.runtime.memGetInfo()
    print(f"  [solve] device memory free {free / 1e9:.1f} of {total / 1e9:.1f} GB")

  def _plan(self, S: sp.csr_matrix) -> None:
    adv = self._adv
    self.close()
    kind, view = (
      (adv.DirectSolverMatrixType.GENERAL, adv.DirectSolverMatrixViewType.FULL)
      if self._general
      else (adv.DirectSolverMatrixType.SPD, adv.DirectSolverMatrixViewType.UPPER)
    )
    self._matrix = self.xsp.csr_matrix(S)
    self._rhs = self.xp.zeros((S.shape[0], 1))
    self._solver = adv.DirectSolver(
      self._matrix,
      self._rhs,
      options=adv.DirectSolverOptions(
        sparse_system_type=kind,
        sparse_system_view=view,
        multithreading_lib=self._threading,
      ),
    )
    self._solver.plan_config.reordering_algorithm = adv.DirectSolverReorderingAlg.AMD
    self._solver.plan()
    self._pattern = (S.indptr.copy(), S.indices.copy())

  def factorize(self, S: sp.csr_matrix) -> None:
    self.xp.get_default_memory_pool().free_all_blocks()
    if (
      self._solver is None
      or self._pattern is None
      or not (
        np.array_equal(self._pattern[0], S.indptr)
        and np.array_equal(self._pattern[1], S.indices)
      )
    ):
      self._plan(S)
    else:
      self._matrix.data.set(S.data)
    self._solver.factorize()
    if self._general:
      return
    if not bool(self.xp.isfinite(self.apply(self.xp.ones(S.shape[0]))).all()):
      print("  [solve] Cholesky refused the state system; using LU")
      self._general = True
      self._plan(S)
      self._solver.factorize()

  def apply(self, rhs: Any) -> Any:
    self._rhs[:, 0] = rhs
    return self._solver.solve().ravel().copy()

  def to_host(self, array: Any) -> np.ndarray:
    return np.asarray(self.xp.asnumpy(array))

  def close(self) -> None:
    if self._solver is not None:
      self._solver.free()
      self._solver = None


def assert_backend() -> None:
  """Import the CUDA solve stack, raising when any piece is missing."""
  _Cudss().close()


def _pcg(
  apply_system: Callable[[Any], Any],
  apply_preconditioner: Callable[[Any], Any],
  b: Any,
  xp: Any,
) -> tuple[Any, int]:
  """PCG from zero to relative residual ``_PCG_TOL``, returning ``(x, iterations)``."""
  x = xp.zeros_like(b)
  stop = _PCG_TOL * float(xp.linalg.norm(b))
  if stop == 0.0:
    return x, 0
  r = b.copy()
  z = apply_preconditioner(r)
  p = z.copy()
  rz = float(r @ z)
  for it in range(1, _PCG_MAX_ITER + 1):
    sp_ = apply_system(p)
    alpha = rz / float(p @ sp_)
    x += alpha * p
    r -= alpha * sp_
    if float(xp.linalg.norm(r)) <= stop:
      return x, it
    z = apply_preconditioner(r)
    rz_next = float(r @ z)
    p = z + (rz_next / rz) * p
    rz = rz_next
  return x, _PCG_MAX_ITER


class _NormalEquations:
  """One linearization's normal equations on the host, in state and landmark blocks."""

  def __init__(
    self,
    jacobian: sp.csr_matrix,
    rhs: np.ndarray,
    landmark_cols: np.ndarray,
    state_cols: np.ndarray,
  ) -> None:
    H = (jacobian.T @ jacobian).tocsr()
    g = jacobian.T @ rhs
    H_state = H[state_cols]
    Hll = H[landmark_cols][:, landmark_cols]
    self.Hsl = H_state[:, landmark_cols].tocsr()
    self.Hss = H_state[:, state_cols].tocsr()
    self.Hls = self.Hsl.T.tocsr()
    # Canonical CSR uploads as is; otherwise the device sorts it on first use.
    for block in (self.Hsl, self.Hss, self.Hls):
      block.sum_duplicates()
    self.gl, self.gs = g[landmark_cols], g[state_cols]
    # Hll is block-diagonal, one 3x3 block per landmark.
    Hll_coo = Hll.tocoo()
    self.blocks = np.zeros(
      (len(landmark_cols) // _LANDMARK_DIM, _LANDMARK_DIM, _LANDMARK_DIM)
    )
    self.blocks[
      Hll_coo.row // _LANDMARK_DIM,
      Hll_coo.row % _LANDMARK_DIM,
      Hll_coo.col % _LANDMARK_DIM,
    ] = Hll_coo.data
    self._Hll_inv: sp.csr_matrix | None = None

  def block_inverse(self, lam: float) -> sp.csr_matrix:
    """The damped landmark Hessian's pseudo-inverse, kept for this damping's step."""
    # A direction a landmark's observations cannot see gets no step.
    block_inv = np.linalg.pinv(
      self.blocks + lam * np.eye(_LANDMARK_DIM), rcond=1e-9, hermitian=True
    )
    n = len(self.blocks)
    self._Hll_inv = sp.bsr_matrix(
      (block_inv, np.arange(n), np.arange(n + 1)),
      shape=(n * _LANDMARK_DIM, n * _LANDMARK_DIM),
    ).tocsr()
    return self._Hll_inv

  def landmarks(self, x_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(state step, landmark step)`` for the solved state."""
    assert self._Hll_inv is not None
    return x_state, self._Hll_inv @ (self.gl - self.Hls @ x_state)


class _SchurPcg:
  """One linearization's state system solved by conjugate gradients in the
  factorization's array library, preconditioned by it on the banded truncation."""

  def __init__(
    self, normal: _NormalEquations, keyframes: np.ndarray, factorization: Factorization
  ) -> None:
    self._normal = normal
    self._f = factorization
    xp, xsp = factorization.xp, factorization.xsp
    self.Hss, self.Hsl, self.Hls = (
      xsp.csr_matrix(block) for block in (normal.Hss, normal.Hsl, normal.Hls)
    )
    self.gs, self.gl = xp.asarray(normal.gs), xp.asarray(normal.gl)
    self._keyframes = xp.asarray(keyframes)
    chunks = max(_SCHUR_CHUNKS, -(-normal.Hsl.nnz // _SCHUR_CHUNK_NNZ))
    self._bounds = np.linspace(0, normal.Hss.shape[0], chunks + 1).astype(int)
    self._factored = False

  def _factor_banded(self, lam: float, Hll_inv: Any) -> None:
    xp, xsp, host = self._f.xp, self._f.xsp, self._f.to_host
    n_state = self.Hss.shape[0]
    X = Hll_inv @ self.Hls
    data: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    indptr = np.zeros(n_state + 1, dtype=np.int64)
    for a, b in zip(self._bounds[:-1], self._bounds[1:], strict=True):
      damping = xsp.diags(
        xp.full(b - a, lam, dtype=np.float64), offsets=int(a), shape=(b - a, n_state)
      )
      chunk = (self.Hss[a:b] - self.Hsl[a:b] @ X + damping).tocoo()
      kf_row, kf_col = self._keyframes[chunk.row + a], self._keyframes[chunk.col]
      keep = (
        (kf_row < 0) | (kf_col < 0) | (xp.abs(kf_row - kf_col) <= _PRECONDITIONER_BAND)
      )
      part = xsp.csr_matrix(
        (chunk.data[keep], (chunk.row[keep], chunk.col[keep])), shape=(b - a, n_state)
      )
      part.sort_indices()
      data.append(host(part.data))
      indices.append(host(part.indices))
      indptr[a + 1 : b + 1] = indptr[a] + host(part.indptr[1:])
      del chunk, part
    del X
    self._f.factorize(
      sp.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), indptr),
        shape=(n_state, n_state),
      )
    )

  def solve(self, lam: float) -> tuple[np.ndarray, int]:
    """The state step for damping ``lam`` and its PCG iterations; the landmark step waits in ``normal``."""
    Hll_inv = self._f.xsp.csr_matrix(self._normal.block_inverse(lam))
    if not self._factored:
      self._factor_banded(lam, Hll_inv)
      self._factored = True
    rhs = self.gs - self.Hsl @ (Hll_inv @ self.gl)

    def apply_system(x: Any) -> Any:
      return self.Hss @ x - self.Hsl @ (Hll_inv @ (self.Hls @ x)) + lam * x

    x, iterations = _pcg(apply_system, self._f.apply, rhs, self._f.xp)
    return self._f.to_host(x), iterations


def _lm(
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
  max_iterations: int,
  initial_error: float,
  lam: float,
) -> tuple[gtsam.Values, int, float, float]:
  """``(values, iterations, final_error, last accepted lambda)``."""
  system = SparseSystem(graph, values)
  factorization = _Cudss()
  landmark_cols, state_cols = system.landmark_cols, system.state_cols
  accepted_lambda = lam
  # Accepted steps to hold lambda for, how many were held, whether it just fell.
  hold = 0
  held = 0
  fell = False
  err = initial_error
  iters = 0
  pcg_iterations: list[int] = []
  try:
    while iters < max_iterations:
      jacobian, rhs = system.fill(values)
      normal = _NormalEquations(jacobian, rhs, landmark_cols, state_cols)
      schur = _SchurPcg(normal, system.state_keyframes, factorization)
      rejected = 0
      while lam <= _LAMBDA_UPPER_BOUND:
        x_state, pcg = schur.solve(lam)
        pcg_iterations.append(pcg)
        delta = np.zeros(system.n_cols)
        delta[state_cols], delta[landmark_cols] = normal.landmarks(x_state)
        candidate = values.retract(system.scatter(delta))
        candidate_error = system.error(candidate)
        if candidate_error < err:
          relative_decrease = (err - candidate_error) / max(err, 1e-12)
          values = candidate
          err = candidate_error
          accepted_lambda = lam
          if held < hold:
            held += 1
            fell = False
          else:
            # Floor keeps lambda positive.
            lam = max(lam / _LAMBDA_FACTOR, 1e-10)
            held = 0
            fell = True
          iters += 1
          if relative_decrease < _RELATIVE_ERROR_TOL:
            return values, iters, err, accepted_lambda
          break
        rejected += 1
        lam *= _LAMBDA_FACTOR**rejected
        if fell:
          hold = max(1, 2 * hold)
          held = 0
          fell = False
      else:
        break
  finally:
    if pcg_iterations:
      counts = np.asarray(pcg_iterations)
      print(
        f"  [solve] conjugate gradients: {len(counts)} solves, iterations "
        f"median {int(np.median(counts))} max {counts.max()}"
      )
    factorization.close()
  return values, iters, err, accepted_lambda


@dataclass(frozen=True)
class OptimizeResult:
  values: gtsam.Values
  iterations: int
  initial_error: float
  final_error: float
  # The damping of the last accepted step.
  final_lambda: float


def optimize_graph(
  graph: gtsam.NonlinearFactorGraph,
  initial_values: gtsam.Values,
  max_iterations: int,
  *,
  initial_lambda: float = _LAMBDA_INITIAL,
) -> OptimizeResult:
  """Schur LM solve of ``graph``; raise when the initial error is not finite."""
  initial_error = float(graph.error(initial_values))
  if not np.isfinite(initial_error):
    raise RuntimeError(
      "graph initial error is not finite: a factor evaluates to inf/NaN at "
      "the initial values, so the optimizer would silently return them"
    )
  begin = time.perf_counter()
  values, iterations, final_error, final_lambda = _lm(
    graph, initial_values, max_iterations, initial_error, initial_lambda
  )
  print(
    f"  [solve] {iterations} iters in {time.perf_counter() - begin:.1f}s, "
    f"error {initial_error:.0f} -> {final_error:.0f}"
  )
  return OptimizeResult(values, iterations, initial_error, final_error, final_lambda)
