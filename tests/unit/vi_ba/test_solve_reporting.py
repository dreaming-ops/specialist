"""What the two-pass solve reports about itself."""

from __future__ import annotations

import gtsam
import pytest

from specialist.vi_ba import reference_solve
from specialist.vi_ba.compose import CameraKeys
from specialist.vi_ba.optimize import OptimizeResult
from specialist.vi_ba.reference_solve import (
  _PRECULL_MAX_ITERATIONS,
  _solve_with_cull_reoptimize,
)

_MAX_ITERATIONS = 30


def _canned(passes: list[OptimizeResult], monkeypatch: pytest.MonkeyPatch) -> list[int]:
  """Feed the solve fixed results and return the budget each pass that ran received."""
  budgets: list[int] = []

  def stub(
    graph: gtsam.NonlinearFactorGraph,
    initial_values: gtsam.Values,
    max_iterations: int,
    gpu: str = "off",
  ) -> OptimizeResult:
    budgets.append(max_iterations)
    return passes[len(budgets) - 1]

  monkeypatch.setattr(reference_solve, "optimize_graph", stub)
  return budgets


def _run() -> OptimizeResult:
  """No cameras, so nothing is culled and the reporting stands alone."""
  result, _keep = _solve_with_cull_reoptimize(
    gtsam.NonlinearFactorGraph(),
    gtsam.Values(),
    [],
    CameraKeys([]),
    _MAX_ITERATIONS,
    "off",
  )
  return result


def test_a_topped_up_solve_reports_the_error_the_run_started_from(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The top-up starts where the short ranking pass stopped, not where the run began."""
  budgets = _canned(
    [
      OptimizeResult(gtsam.Values(), _PRECULL_MAX_ITERATIONS, 1_800_006.0, 243.0),
      OptimizeResult(gtsam.Values(), 1, 243.0, 100.0),
    ],
    monkeypatch,
  )
  result = _run()

  assert budgets == [_PRECULL_MAX_ITERATIONS, _MAX_ITERATIONS]
  assert result.initial_error == 1_800_006.0
  assert result.iterations == _PRECULL_MAX_ITERATIONS + 1
  assert result.final_error == 100.0


def test_a_converged_ranking_solve_is_the_whole_answer(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A first pass that converges with nothing to cull runs no second pass."""
  budgets = _canned([OptimizeResult(gtsam.Values(), 4, 891.0, 0.0)], monkeypatch)
  result = _run()

  assert budgets == [_PRECULL_MAX_ITERATIONS]
  assert result.initial_error == 891.0
  assert result.iterations == 4
