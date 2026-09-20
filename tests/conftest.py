"""Tests solve on the host, with scipy standing in for the cuDSS factorization."""

from __future__ import annotations

import pytest

from specialist.vi_ba import gpu_lm
from tests.solver_fixtures import ScipyFactorization


@pytest.fixture(autouse=True)
def _host_solver(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(gpu_lm, "_Cudss", ScipyFactorization)
