"""Every module this member imports is a dependency it declares."""

from __future__ import annotations

import ast
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SOURCE = _ROOT / "src" / "specialist_dataset"
_OWN = "specialist_dataset"


def _declared() -> set[str]:
  pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
  names = [
    requirement.split(">")[0].split("=")[0].split("[")[0].strip()
    for requirement in pyproject["project"]["dependencies"]
  ]
  return {name.replace("-", "_").lower() for name in names}


def _imported_roots() -> set[str]:
  roots: set[str] = set()
  for source in _SOURCE.rglob("*.py"):
    for node in ast.walk(ast.parse(source.read_text())):
      if isinstance(node, ast.Import):
        roots.update(alias.name.split(".")[0] for alias in node.names)
      elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        roots.add(node.module.split(".")[0])
  return roots


def test_every_imported_module_is_a_declared_dependency() -> None:
  declared, provided = _declared(), packages_distributions()
  for root in sorted(_imported_roots()):
    if root in sys.stdlib_module_names or root == _OWN:
      continue
    distributions = {
      name.replace("-", "_").lower() for name in provided.get(root, [root])
    }
    assert distributions & declared, (
      f"{root} is imported but no distribution providing it is declared"
    )
