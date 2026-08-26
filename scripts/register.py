"""Register a dataset's layers on a Rerun catalog dataset, refusing name collisions.

Usage:
    uv run python scripts/register.py --catalog-url rerun+http://host:port \
        --dataset gripper_dataset --root dataset
"""

from __future__ import annotations

import functools
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tyro
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from rerun.experimental import RrdReader
from specialist_dataset.read import check_frame_alignment
from specialist_dataset.schema import (
  APPLICATION_ID,
  ESTIMATE,
  anchor_id_of,
  anchor_ids_in,
  slots_in,
)

from specialist.record.session import SessionRecording
from specialist.vi_ba.blueprint import blueprint_for_slots, segment_card_blueprint


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


def anchor_tag_sizes(rrds: list[Path]) -> set[float]:
  """The anchor tag widths that the estimate layers record."""
  sizes = set()
  for rrd in rrds:
    for chunk in RrdReader(str(rrd)).stream():
      if anchor_id_of(str(chunk.entity_path)) is None:
        continue
      batch = chunk.to_record_batch()
      if "tag_size" in batch.schema.names:
        sizes |= {v for row in batch.column("tag_size").to_pylist() if row for v in row}
  return sizes


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


def written_names(rrd: Path) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
  """Every ``(entity, component)`` and ``(property group, key)`` one file writes."""
  columns = RrdReader(str(rrd)).store().schema().component_columns()
  entities = {(c.entity_path, c.component) for c in columns if not c.is_property}
  properties = {(c.entity_path, c.component) for c in columns if c.is_property}
  return entities, properties


def check_no_collisions(files_by_session: dict[str, list[Path]]) -> None:
  """Refuse a session whose layers write the same name twice."""
  clashes = []
  for session, files in sorted(files_by_session.items()):
    if len(files) < 2:
      continue
    seen: dict[tuple[str, str], Path] = {}
    seen_properties: dict[tuple[str, str], Path] = {}
    for path in files:
      entities, properties = written_names(path)
      for owned, names in ((seen, entities), (seen_properties, properties)):
        for name in sorted(names):
          if name in owned:
            clashes.append(
              f"{session}: {name[0]}:{name[1]} written by both "
              f"{owned[name].parent.name}/ and {path.parent.name}/"
            )
          else:
            owned[name] = path
  if clashes:
    raise SystemExit("layers collide:\n  " + "\n  ".join(clashes))


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
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
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
  by_session: dict[str, list[Path]] = {}
  for path, rid in {**bases, **layers}.items():
    by_session.setdefault(rid, []).append(path)
  check_no_collisions(by_session)
  # Only the estimate layer shares the base's frame clock.
  base_by_id = {rid: p for p, rid in bases.items()}
  for path in layer_files.get("estimate", []):
    base = base_by_id.get(layers[path])
    if base is None:
      print(
        f"[warn] {path.name} attaches to already-registered base "
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
  accepted, kept_bad = register_batch(
    entry, keep, OnDuplicateSegmentLayer.SKIP, failures
  )
  redone, redone_bad = register_batch(
    entry, redo, OnDuplicateSegmentLayer.REPLACE, failures
  )
  accepted |= redone
  bad = sum(kept_bad.values()) + sum(redone_bad.values())
  print(f"registered {len(bases) - bad} of {len(bases)} base recording(s)")
  if failures:
    raise SystemExit(f"{len(failures)} registration(s) failed; no blueprint installed")
  check_layers_attach(known | accepted, layers)
  # `layer_name` takes one name per URI.
  layer_of = {p: name for name, files in layer_files.items() for p in files}
  _, failed = register_batch(entry, layer_of, OnDuplicateSegmentLayer.REPLACE, failures)
  for name, files in layer_files.items():
    print(f"registered {len(files) - failed[name]} {name} layer(s)")
  if failures:
    raise SystemExit(f"{len(failures)} registration(s) failed; no blueprint installed")

  paths = [c.entity_path for c in entry.schema().component_columns()]
  slots = slots_in(paths)
  if not slots:
    raise SystemExit(
      "the catalog dataset holds no camera slots; no blueprint installed"
    )
  anchors = anchor_ids_in(paths)
  if not anchors:
    print(
      "  [warn] the catalog dataset holds no world anchors; the blueprint draws none"
    )
  sizes = anchor_tag_sizes(layer_files.get(ESTIMATE, []))
  if len(sizes) > 1:
    raise SystemExit(f"the dataset solved anchors at {sorted(sizes)} m; no blueprint")
  if anchors and not sizes:
    print("  [warn] the estimate layers record no anchor tag size; drawing 0.1 m")
  tag_size = sizes.pop() if sizes else 0.1
  # Segments take the catalog dataset id as application id; a blueprint must match it.
  blueprint_app_id = str(entry.id)
  with tempfile.TemporaryDirectory() as tmp:
    rbl = Path(tmp) / f"{args.dataset}.rbl"
    blueprint_for_slots(slots, anchor_tag_ids=anchors, anchor_tag_size=tag_size).save(
      blueprint_app_id, str(rbl)
    )
    entry.register_blueprint(rbl.resolve().as_uri(), set_default=True)
    # The segment table renders every session, so its blueprint decodes no video.
    cards = Path(tmp) / f"{args.dataset}-cards.rbl"
    segment_card_blueprint(slots).save(blueprint_app_id, str(cards))
    entry.register_blueprint(
      cards.resolve().as_uri(), set_default=True, segment_table=True
    )
  print(f"installed the default + segment-table blueprints on {args.dataset!r}")
  # One .rbl under the recordings' app id, for session files opened directly.
  blueprints_dir = args.root / "blueprints"
  blueprints_dir.mkdir(exist_ok=True)
  dataset_rbl = blueprints_dir / "specialist.rbl"
  blueprint_for_slots(slots, anchor_tag_ids=anchors, anchor_tag_size=tag_size).save(
    APPLICATION_ID, str(dataset_rbl)
  )
  print(
    f"wrote {dataset_rbl.resolve()}: slots {slots}, "
    f"anchor tags {anchors} at {tag_size} m"
  )


if __name__ == "__main__":
  main(tyro.cli(Args))
