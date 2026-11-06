"""The run-configuration invariants ``validate`` enforces before any prep."""

from pathlib import Path

import pytest

from specialist.vi_ba.configs import AnchorCfg, Args, SolverCfg, validate


def _args(**overrides: object) -> Args:
  fields: dict[str, object] = {
    "base": Path("session_base.rrd"),
    "anchor": AnchorCfg(tag_ids=(1,)),
  }
  fields.update(overrides)
  return Args(**fields)  # type: ignore[arg-type]


def test_accepts_a_minimal_single_camera_run() -> None:
  validate(_args())


def test_rejects_nonpositive_solver_iterations() -> None:
  """A zero or negative LM budget silently produces an unrefined solve."""
  with pytest.raises(ValueError, match="max-iterations"):
    validate(_args(solver=SolverCfg(max_iterations=0)))


def test_rejects_empty_anchor_tag_ids() -> None:
  with pytest.raises(ValueError, match="at least one anchor"):
    validate(_args(anchor=AnchorCfg(tag_ids=())))


def test_rejects_a_negative_anchor_tag_id() -> None:
  with pytest.raises(ValueError, match="must be non-negative"):
    validate(_args(anchor=AnchorCfg(tag_ids=(1, -1))))


def test_rejects_duplicate_anchor_tag_ids() -> None:
  with pytest.raises(ValueError, match="must be distinct"):
    validate(_args(anchor=AnchorCfg(tag_ids=(1, 1))))
