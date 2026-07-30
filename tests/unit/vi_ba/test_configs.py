"""The run-configuration invariants ``validate`` enforces before any prep."""

from __future__ import annotations

from pathlib import Path

import pytest

from specialist.vi_ba.configs import AnchorCfg, Args, SolverCfg, WindowCfg, validate


def _args(**overrides: object) -> Args:
  fields: dict[str, object] = {
    "input_insvs": (Path("a.insv"),),
    "window": WindowCfg(duration=None),
    "anchor": AnchorCfg(tag_ids=(1,)),
    "base_tag_ids": (2000,),
  }
  fields.update(overrides)
  return Args(**fields)  # type: ignore[arg-type]


def test_accepts_a_minimal_single_camera_run() -> None:
  validate(_args())


def test_rejects_no_input() -> None:
  with pytest.raises(ValueError, match="at least one .insv"):
    validate(_args(input_insvs=()))


def test_rejects_missing_base_tag_ids() -> None:
  """A base tag identifies each tracker, so every input needs one."""
  with pytest.raises(ValueError, match="base-tag-ids"):
    validate(_args(base_tag_ids=()))


@pytest.mark.parametrize(
  "field", ["base_tag_ids", "left_finger_tag_ids", "right_finger_tag_ids"]
)
def test_rejects_per_tracker_list_shorter_than_the_inputs(field: str) -> None:
  """A short id list would silently pair the wrong tag with the wrong clip."""
  overrides: dict[str, object] = {
    "input_insvs": (Path("a.insv"), Path("b.insv")),
    "base_tag_ids": (2000, 2003),
  }
  overrides[field] = (2000,)
  with pytest.raises(ValueError, match="must match"):
    validate(_args(**overrides))


def test_rejects_duplicate_base_tag_ids() -> None:
  """Cross-tracker factors cannot tell apart two cameras wearing one base tag."""
  with pytest.raises(ValueError, match="distinct per tracker"):
    validate(
      _args(input_insvs=(Path("a.insv"), Path("b.insv")), base_tag_ids=(2000, 2000))
    )


def test_rejects_nonpositive_solver_iterations() -> None:
  """A zero or negative LM budget silently produces an unrefined solve."""
  with pytest.raises(ValueError, match="max-iterations"):
    validate(_args(solver=SolverCfg(max_iterations=0)))


def test_rejects_empty_anchor_tag_ids() -> None:
  with pytest.raises(ValueError, match="at least the reference anchor"):
    validate(_args(anchor=AnchorCfg(tag_ids=())))


def test_rejects_duplicate_anchor_tag_ids() -> None:
  with pytest.raises(ValueError, match="must be distinct"):
    validate(_args(anchor=AnchorCfg(tag_ids=(1, 1))))


def test_tag_id_disjointness_validated() -> None:
  """Anchor, base and nonzero finger ids are disjoint; a zero finger id is disabled."""
  validate(_args(anchor=AnchorCfg(tag_ids=(1, 0))))
  validate(_args(anchor=AnchorCfg(tag_ids=(1, 0)), left_finger_tag_ids=(0,)))
  with pytest.raises(ValueError, match="2000"):
    validate(_args(anchor=AnchorCfg(tag_ids=(1, 2000))))
  with pytest.raises(ValueError, match="1"):
    validate(_args(anchor=AnchorCfg(tag_ids=(1, 0)), left_finger_tag_ids=(1,)))
  with pytest.raises(ValueError, match="2000"):
    validate(_args(anchor=AnchorCfg(tag_ids=(1, 0)), right_finger_tag_ids=(2000,)))
