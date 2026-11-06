"""Tests solve on the host, with scipy standing in for the cuDSS factorization."""

from pathlib import Path

import pytest

from specialist.vi_ba import gpu_lm
from tests.record_fixtures import camera_estimate, urdf_file, write_run
from tests.solver_fixtures import ScipyFactorization


@pytest.fixture(autouse=True)
def _host_solver(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(gpu_lm, "_Cudss", ScipyFactorization)


@pytest.fixture(scope="session")
def estimated_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
  """The estimate and viz layers of one solved two-camera run."""
  tmp = tmp_path_factory.mktemp("solved")
  return write_run(
    tmp,
    "sess_route",
    [camera_estimate("cam00"), camera_estimate("cam01", offset=1.0)],
    urdf_path=urdf_file(tmp),
    frame_period=1.0 / 30.0,
  )
