"""A session names itself from the capture that defines its clock."""

from pathlib import Path

import pytest
from scripts.convert import session_name


def test_session_name_pairs_the_start_instant_with_the_device() -> None:
  master = Path("VID_20260908_044638_00_001.insv")
  assert session_name(master, "IAQEB2601UVVAJ") == "20260908_044638_IAQEB2601UVVAJ"
  assert session_name(master, "IAQEB2601UVVAJ") < session_name(
    Path("VID_20260908_044840_00_002.insv"), "IAQEB2601UVVAJ"
  )


def test_session_name_refuses_a_capture_it_cannot_name() -> None:
  with pytest.raises(SystemExit, match="no capture instant or serial"):
    session_name(Path("clip.insv"), "IAQEB2601UVVAJ")
  with pytest.raises(SystemExit, match="no capture instant or serial"):
    session_name(Path("VID_20260908_044638_00_001.insv"), "")
