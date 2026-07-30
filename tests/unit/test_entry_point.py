"""Import coverage for the CLI entry point, which no other test imports."""

from __future__ import annotations

import importlib


def test_entry_point_imports() -> None:
  track = importlib.import_module("scripts.track")
  assert callable(track.main)
