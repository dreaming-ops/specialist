"""Levenberg-Marquardt wrapper that returns both values and run metadata."""

from __future__ import annotations

import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import gtsam
import numpy as np

from specialist.vi_ba.configs import GpuMode

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
  gpu: GpuMode = "off",
) -> OptimizeResult:
  """LM solve; raises on a non-finite initial error, or a ``"require"`` GPU failure."""
  initial_error = float(graph.error(initial_values))
  if not np.isfinite(initial_error):
    raise RuntimeError(
      "graph initial error is not finite: a factor evaluates to inf/NaN at "
      "the initial values, so the optimizer would silently return them"
    )
  begin = time.perf_counter()
  values: gtsam.Values | None = None
  note = ""
  if gpu != "off":
    try:
      from specialist.vi_ba.gpu_lm import _lm

      with _bounded_solve_affinity():
        values, iterations, _initial_error, final_error = _lm(
          graph,
          initial_values,
          max_iterations,
          _RELATIVE_ERROR_TOL,
          initial_error=initial_error,
        )
      note = " (gpu)"
    except ImportError as exc:
      if gpu == "require":
        raise RuntimeError(
          f"gpu solver required but its backend cannot load: {exc}"
        ) from exc
      print(f"  [LM call] gpu backend unavailable ({exc}); using gtsam", flush=True)
    except Exception:
      if gpu == "require":
        raise
      traceback.print_exc()
      print("  [LM call] gpu solve failed; using gtsam", flush=True)
  if values is None:
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iterations)
    params.setRelativeErrorTol(_RELATIVE_ERROR_TOL)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
    begin = time.perf_counter()  # exclude any failed gpu attempt from the timing
    with _bounded_solve_affinity():
      values = optimizer.optimize()
    iterations = int(optimizer.iterations())
    final_error = float(optimizer.error())
  print(
    f"  [LM call] {iterations} iters in {time.perf_counter() - begin:.1f}s, "
    f"error {initial_error:.0f} -> {final_error:.0f}{note}",
    flush=True,
  )
  return OptimizeResult(
    values=values,
    iterations=iterations,
    initial_error=initial_error,
    final_error=final_error,
  )
