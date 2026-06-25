"""Hatchling build hook: compile the native MEI factor at install time."""

from __future__ import annotations

import runpy
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
  def initialize(self, version: str, build_data: dict) -> None:
    script = Path(self.root) / "src" / "specialist" / "vi_ba" / "_buildext.py"
    runpy.run_path(str(script))["build"]()
    build_data["pure_python"] = False
    build_data["infer_tag"] = True
