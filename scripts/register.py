"""Register base recordings and their layers on a Rerun catalog dataset.

Usage:
    uv run python scripts/register.py --catalog-url rerun+http://host:port \
        --dataset gripper_dataset --base ft005_base.rrd --track track_x.rrd
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import tyro
from rerun.catalog import CatalogClient
from rerun.experimental import RrdReader

from specialist.record.schema import APPLICATION_ID
from specialist.vi_ba.viz import blueprint_for_names


@dataclass
class Args:
  """Catalog registration of base recordings and layers."""

  catalog_url: str = ""
  dataset: str = ""
  base: tuple[Path, ...] = ()
  track: tuple[Path, ...] = ()
  # Extra layers (e.g. ground truth), registered under their stem's suffix.
  extra_layer: tuple[str, ...] = ()  # name=path pairs
  slots: tuple[str, ...] = ("tracker_0", "tracker_1")


def recording_id(rrd: Path) -> str:
  """The single recording id carried by one record file."""
  entries = RrdReader(str(rrd)).recordings()
  if len(entries) != 1:
    raise ValueError(f"{rrd} holds {len(entries)} recordings; expected exactly 1")
  return str(entries[0].recording_id)


def check_layers_attach(bases: dict[Path, str], layers: dict[Path, str]) -> None:
  """Refuse layers whose recording id matches no base segment."""
  base_ids = set(bases.values())
  orphans = {p: rid for p, rid in layers.items() if rid not in base_ids}
  if orphans:
    lines = ", ".join(f"{p.name} ({rid})" for p, rid in orphans.items())
    raise ValueError(f"layers match no base segment: {lines}")


def main(args: Args) -> None:
  if not args.catalog_url or not args.dataset or not args.base:
    raise SystemExit("--catalog-url, --dataset, and --base are required")
  bases = {p: recording_id(p) for p in args.base}
  solutions = {p: recording_id(p) for p in args.track}
  extra = {}
  for pair in args.extra_layer:
    name, sep, raw = pair.partition("=")
    if not sep:
      raise SystemExit(f"--extra-layer expects name=path, got {pair!r}")
    extra[(name, Path(raw))] = recording_id(Path(raw))

  client = CatalogClient(args.catalog_url)
  if args.dataset in client.dataset_names():
    entry = client.get_dataset(args.dataset)
    known = {str(s) for s in entry.segment_ids()}
  else:
    entry = client.create_dataset(args.dataset)
    known = set()
  check_layers_attach(
    bases | {Path(f"registered-{rid}"): rid for rid in known},
    solutions | {path: rid for (_, path), rid in extra.items()},
  )

  if bases:
    entry.register([p.resolve().as_uri() for p in bases], layer_name="base")
    print(f"registered {len(bases)} base recording(s)")
  if solutions:
    entry.register([p.resolve().as_uri() for p in solutions], layer_name="track")
    print(f"registered {len(solutions)} track layer(s)")
  for (name, path), _ in extra.items():
    entry.register([path.resolve().as_uri()], layer_name=name)
    print(f"registered {path.name} as layer {name!r}")

  with tempfile.TemporaryDirectory() as tmp:
    rbl = Path(tmp) / "run.rbl"
    blueprint_for_names(list(args.slots)).save(APPLICATION_ID, str(rbl))
    entry.register_blueprint(rbl.resolve().as_uri(), set_default=True)
  print(f"installed the default blueprint on {args.dataset!r}")


if __name__ == "__main__":
  main(tyro.cli(Args))
