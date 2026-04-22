"""The ``.rrd`` + JSON summary tail the probe scripts share."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import rerun as rr


def finalize_probe(
  name: str,
  app_id: str,
  output_dir: Path,
  summary: Any,
  log_fn: Callable[[], None],
  *,
  extra_lines: list[str] | None = None,
) -> None:
  """Save a probe's ``.rrd``, logged by ``log_fn``, and its JSON summary."""
  stamp = time.strftime("%Y%m%d_%H%M%S")
  rrd_path = output_dir / f"{name}_{stamp}.rrd"
  json_path = output_dir / f"{name}_{stamp}.json"

  rr.init(app_id)
  rr.save(str(rrd_path))
  log_fn()

  payload = json.dumps(asdict(summary), indent=2)
  json_path.write_text(payload)
  print(payload)
  print(f"\nWrote {rrd_path} and {json_path}")
  for line in extra_lines or []:
    print(line)
  print(f"Verdict: {summary.verdict}")
