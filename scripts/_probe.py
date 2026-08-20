"""The ``.rrd`` + JSON summary tail the probe scripts share."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import rerun as rr


def verdict(problems: Sequence[str]) -> str:
  return "green" if not problems else "yellow: " + "; ".join(problems)


def finalize_probe(
  name: str,
  output_dir: Path,
  summary: Any,
  log_fn: Callable[[], None],
) -> None:
  """Save a probe's ``.rrd``, logged by ``log_fn``, and its JSON summary."""
  output_dir.mkdir(parents=True, exist_ok=True)
  stamp = time.strftime("%Y%m%d_%H%M%S")
  rrd_path = output_dir / f"{name}_{stamp}.rrd"
  json_path = output_dir / f"{name}_{stamp}.json"

  rr.init(f"specialist-{name}")
  rr.save(str(rrd_path))
  log_fn()
  rr.disconnect()

  payload = json.dumps(asdict(summary), indent=2)
  json_path.write_text(payload)
  print(payload)
  print(f"\nWrote {rrd_path} and {json_path}")
  print(f"Verdict: {summary.verdict}")
