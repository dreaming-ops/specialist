"""Every CLI driver runs by path, as the README invokes it."""

import subprocess
import sys
from pathlib import Path

import pytest

_DRIVERS = (
  "estimate",
  "convert",
  "state",
  "catalog",
  "imu",
  "video_imu_timing",
  "convert_mocap",
  "score_mocap",
)


@pytest.mark.parametrize("name", _DRIVERS)
def test_every_driver_runs_by_path(name: str) -> None:
  """The README's ``python scripts/x.py`` form puts scripts/ first on the path."""
  result = subprocess.run(
    [sys.executable, f"scripts/{name}.py", "--help"],
    capture_output=True,
    cwd=Path(__file__).resolve().parents[2],
  )
  assert result.returncode == 0, result.stderr.decode()[-500:]
