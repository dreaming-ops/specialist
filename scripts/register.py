"""Register a dataset's layers on a Rerun catalog dataset, one layer per directory.

Usage:
    uv run python scripts/register.py --catalog-url rerun+http://host:port \
        --dataset gripper_dataset --root dataset
"""

from __future__ import annotations

import functools
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tyro
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from rerun.experimental import RrdReader

from specialist.record.read import check_frame_alignment
from specialist.record.session import SessionRecording
from specialist.vi_ba.viz import blueprint_for_names, segment_card_blueprint


@dataclass
class Args:
  """Catalog registration of a layer-first dataset root."""

  # The catalog server, e.g. rerun+http://host:port.
  catalog_url: str
  # Catalog dataset to register into; created if it does not exist.
  dataset: str
  # Dataset root: base/<session>.rrd plus one directory per additional layer.
  root: Path
  # Session ids of re-converted bases; their segments register with REPLACE.
  replace_base: tuple[str, ...] = ()


def recording_id(rrd: Path) -> str:
  """The single recording id carried by one record file."""
  entries = RrdReader(str(rrd)).recordings()
  if len(entries) != 1:
    raise ValueError(f"{rrd} holds {len(entries)} recordings; expected exactly 1")
  return str(entries[0].recording_id)


def dataset_layers(root: Path) -> tuple[list[Path], dict[str, list[Path]]]:
  """``(base_files, {layer_name: files})``; exits when ``base`` holds no rrd."""
  bases = sorted((root / "base").glob("*.rrd"))
  if not bases:
    raise SystemExit(f"{root / 'base'} holds no .rrd; a dataset needs its base layer")
  layers = {
    d.name: files
    for d in sorted(root.iterdir())
    if d.is_dir() and d.name != "base" and (files := sorted(d.glob("*.rrd")))
  }
  return bases, layers


def check_layers_attach(base_ids: set[str], layers: dict[Path, str]) -> None:
  """Refuse layers whose recording id matches no base segment."""
  orphans = {p: rid for p, rid in layers.items() if rid not in base_ids}
  if orphans:
    lines = ", ".join(f"{p.name} ({rid})" for p, rid in orphans.items())
    raise ValueError(f"layers match no base segment: {lines}")


_recording = functools.cache(SessionRecording)


def register_batch(
  entry: DatasetEntry,
  layer_of: dict[Path, str],
  on_duplicate: OnDuplicateSegmentLayer,
  failures: list[str],
) -> tuple[set[str], Counter[str]]:
  """Register one batch and return its accepted segment ids and failures per layer."""
  if not layer_of:
    return set(), Counter()
  layer_by_uri = {p.resolve().as_uri(): layer for p, layer in layer_of.items()}
  handle = entry.register(
    list(layer_by_uri),
    layer_name=list(layer_by_uri.values()),
    on_duplicate=on_duplicate,
  )
  accepted: set[str] = set()
  failed: Counter[str] = Counter()
  for result in handle.iter_results(timeout_secs=1800):
    if result.is_error:
      layer = layer_by_uri[result.uri]
      failed[layer] += 1
      failures.append(f"{layer}: {result.uri}: {result.error}")
      print(f"FAILED {layer}: {result.uri}: {result.error}")
    else:
      accepted.add(str(result.segment_id))
  return accepted, failed


def main(args: Args) -> None:
  base_files, layer_files = dataset_layers(args.root)
  bases = {p: recording_id(p) for p in base_files}

  client = CatalogClient(args.catalog_url)
  if args.dataset in client.dataset_names():
    entry = client.get_dataset(args.dataset)
    known = {str(s) for s in entry.segment_ids()}
  else:
    entry = client.create_dataset(args.dataset)
    known = set()
  layers = {p: recording_id(p) for files in layer_files.values() for p in files}
  # Only the track layer shares the base's frame clock, so only its frames are checked.
  base_by_id = {rid: p for p, rid in bases.items()}
  for path in layer_files.get("track", []):
    base = base_by_id.get(layers[path])
    if base is None:
      print(
        f"WARNING: {path.name} attaches to already-registered base "
        f"{layers[path]}; frame alignment NOT checked (restore that base "
        "under the root to verify)"
      )
      continue
    check_frame_alignment(_recording(base).frame_times, path)

  unknown = set(args.replace_base) - set(bases.values())
  if unknown:
    raise SystemExit(
      f"--replace-base names no base under {args.root}: {sorted(unknown)}"
    )

  # Duplicate bases SKIP; REPLACE rebuilds the catalog's byte-range index.
  failures: list[str] = []
  replace = set(args.replace_base)
  keep = {p: "base" for p, rid in bases.items() if rid not in replace}
  redo = {p: "base" for p, rid in bases.items() if rid in replace}
  accepted, _ = register_batch(entry, keep, OnDuplicateSegmentLayer.SKIP, failures)
  redone, _ = register_batch(entry, redo, OnDuplicateSegmentLayer.REPLACE, failures)
  accepted |= redone
  print(f"registered {len(bases) - len(failures)} of {len(bases)} base recording(s)")
  check_layers_attach(known | accepted, layers)
  # `layer_name` takes one name per URI.
  layer_of = {p: name for name, files in layer_files.items() for p in files}
  _, failed = register_batch(entry, layer_of, OnDuplicateSegmentLayer.REPLACE, failures)
  for name, files in layer_files.items():
    print(f"registered {len(files) - failed[name]} {name} layer(s)")

  names = sorted({n for p in bases for n in _recording(p).slots})
  # Segments take the catalog dataset id as application id; a blueprint must match it.
  blueprint_app_id = str(entry.id)
  with tempfile.TemporaryDirectory() as tmp:
    rbl = Path(tmp) / f"{args.dataset}.rbl"
    blueprint_for_names(names).save(blueprint_app_id, str(rbl))
    entry.register_blueprint(rbl.resolve().as_uri(), set_default=True)
    # The segment table renders every session, so its blueprint decodes no video.
    cards = Path(tmp) / f"{args.dataset}-cards.rbl"
    segment_card_blueprint(names).save(blueprint_app_id, str(cards))
    entry.register_blueprint(
      cards.resolve().as_uri(), set_default=True, segment_table=True
    )
  print(f"installed the default + segment-table blueprints on {args.dataset!r}")
  if failures:
    raise SystemExit(f"{len(failures)} registration(s) failed")


if __name__ == "__main__":
  main(tyro.cli(Args))
