"""Register a state layer as a Rerun catalog dataset, with its assets and blueprints.

Usage:
    uv run python scripts/catalog.py dataset/state_s8_r100 \
        --catalog-url rerun+http://host:port
"""

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tyro
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from specialist_dataset.read import Recording
from specialist_dataset.schema import (
  APPLICATION_ID,
  HANDS,
  POSE_TRANSLATION,
  STATE,
  state_image,
)

from specialist.record.asset import register_assets
from specialist.vi_ba.blueprint import state_blueprint, state_card_blueprint


@dataclass
class Args:
  """Catalog registration of one state configuration."""

  # A state layer directory, e.g. dataset/state_s8_r100; blueprints land beside it.
  state: tyro.conf.Positional[Path]
  # The catalog server, e.g. rerun+http://host:port.
  catalog_url: str


def state_files(state: Path) -> list[Path]:
  """A state directory's ``.rrd`` files; refuses another directory or an empty one."""
  if state.name != STATE and not state.name.startswith(f"{STATE}_"):
    raise SystemExit(f"{state} is not a state layer directory")
  files = sorted(state.glob("*.rrd"))
  if not files:
    present = sorted(
      d.name
      for d in state.parent.iterdir()
      if d.is_dir()
      and (d.name == STATE or d.name.startswith(f"{STATE}_"))
      and any(d.glob("*.rrd"))
    )
    raise SystemExit(f"{state} holds no .rrd; state layers present: {present}")
  return files


def frame_width(rrd: Path) -> int:
  """The state images' width, the x offset of the right hand's frames."""
  placed = Recording(rrd).row(state_image(HANDS[1])).get(POSE_TRANSLATION)
  if placed is None:
    raise SystemExit(f"{rrd} places no {state_image(HANDS[1])} frames")
  return int(placed[0][0])


def check_one_configuration(entry: DatasetEntry, state: str) -> None:
  """Refuse a catalog dataset with any layer but ``state``: configurations clash."""
  held = {layer.name for layer in entry.info().layers}
  if held - {state}:
    raise SystemExit(
      f"{sorted(held)} is already on the catalog dataset; a catalog dataset "
      "holds one state configuration and no other layer"
    )


def register_all(entry: DatasetEntry, files: list[Path], state: str) -> None:
  """Register every file as a segment, replacing any duplicate."""
  by_uri = {p.resolve().as_uri(): p for p in files}
  handle = entry.register(
    list(by_uri),
    layer_name=[state] * len(by_uri),
    on_duplicate=OnDuplicateSegmentLayer.REPLACE,
  )
  failed = 0
  for result in handle.iter_results(timeout_secs=1800):
    if result.is_error:
      failed += 1
      print(f"FAILED {by_uri[result.uri].name}: {result.error}")
  print(f"registered {len(files) - failed} of {len(files)} {state} recording(s)")
  if failed:
    raise SystemExit(f"{failed} registration(s) failed; no blueprint installed")


def main(args: Args) -> None:
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  files = state_files(args.state)
  # The catalog dataset, created if absent, bears the dataset root's name.
  name = args.state.resolve().parent.name
  client = CatalogClient(args.catalog_url)
  if name in client.dataset_names():
    entry = client.get_dataset(name)
  else:
    entry = client.create_dataset(name)
  check_one_configuration(entry, args.state.name)
  register_all(entry, files, args.state.name)
  register_assets(entry, args.state.parent)

  dashboard = state_blueprint()
  local = args.state.parent / "blueprints" / "state.rbl"
  local.parent.mkdir(exist_ok=True)
  dashboard.save(APPLICATION_ID, str(local))
  print(f"wrote {local.resolve()}")
  # Segments take the catalog dataset id as application id; a blueprint must match it.
  with tempfile.TemporaryDirectory() as tmp:
    for kind, blueprint, table in (
      ("dashboard", dashboard, False),
      # The card shows the two frames, which tell one session from another.
      ("card", state_card_blueprint(frame_width(files[0])), True),
    ):
      rbl = Path(tmp) / f"{name}-{kind}.rbl"
      blueprint.save(str(entry.id), str(rbl))
      entry.register_blueprint(
        rbl.resolve().as_uri(), set_default=True, segment_table=table
      )
  print(f"installed the default + segment-table blueprints on {name!r}")


if __name__ == "__main__":
  main(tyro.cli(Args))
