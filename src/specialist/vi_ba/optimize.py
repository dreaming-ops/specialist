"""Levenberg-Marquardt wrapper that returns both values and run metadata."""

from __future__ import annotations

from dataclasses import dataclass

import gtsam

__all__ = ["OptimizeResult", "optimize_graph"]


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
  params = gtsam.LevenbergMarquardtParams()
  params.setMaxIterations(max_iterations)
  initial_error = float(graph.error(initial_values))
  optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
  result = optimizer.optimize()
  return OptimizeResult(
    values=result,
    iterations=int(optimizer.iterations()),
    initial_error=initial_error,
    final_error=float(graph.error(result)),
  )
