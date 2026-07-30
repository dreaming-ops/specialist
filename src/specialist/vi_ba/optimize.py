"""Levenberg-Marquardt wrapper that returns both values and run metadata."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import gtsam

__all__ = ["OptimizeResult", "optimize_graph"]

# oneTBB sizes its arena from the affinity mask, where the OS supports affinity.
_SOLVE_CORES = 16

# Stop once an accepted step improves the error by less than this fraction.
_RELATIVE_ERROR_TOL = 1e-4


_get_affinity = getattr(os, "sched_getaffinity", None)
_set_affinity = getattr(os, "sched_setaffinity", None)


@contextmanager
def _bounded_solve_affinity() -> Iterator[None]:
  if _get_affinity is None or _set_affinity is None:
    yield
    return
  previous = _get_affinity(0)
  if len(previous) <= _SOLVE_CORES:
    yield
    return
  _set_affinity(0, set(sorted(previous)[:_SOLVE_CORES]))
  try:
    yield
  finally:
    _set_affinity(0, previous)


@dataclass(frozen=True)
class OptimizeResult:
  values: gtsam.Values
  iterations: int
  initial_error: float
  final_error: float


def optimize_graph(
  graph: gtsam.NonlinearFactorGraph,
  initial_values: gtsam.Values,
  max_iterations: int,
) -> OptimizeResult:
  """Run Levenberg-Marquardt and return values + run stats."""
  if max_iterations <= 0:
    # Return the init untouched, to evaluate it without bundle adjustment.
    initial_error = float(graph.error(initial_values))
    return OptimizeResult(
      values=initial_values,
      iterations=0,
      initial_error=initial_error,
      final_error=initial_error,
    )
  params = gtsam.LevenbergMarquardtParams()
  params.setMaxIterations(max_iterations)
  params.setRelativeErrorTol(_RELATIVE_ERROR_TOL)
  optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
  # The optimizer tracks its own error; this skips a full graph evaluation.
  initial_error = float(optimizer.error())
  begin = time.perf_counter()
  with _bounded_solve_affinity():
    result = optimizer.optimize()
  iterations = int(optimizer.iterations())
  final_error = float(optimizer.error())
  print(
    f"  [LM call] {iterations} iters in {time.perf_counter() - begin:.1f}s, "
    f"error {initial_error:.0f} -> {final_error:.0f}",
    flush=True,
  )
  return OptimizeResult(
    values=result,
    iterations=iterations,
    initial_error=initial_error,
    final_error=final_error,
  )
