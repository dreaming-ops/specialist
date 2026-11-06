"""The gripper model as one static recording of every slot, named by the URDF digest."""

import os
from hashlib import sha256
from pathlib import Path

import pyarrow as pa
import rerun as rr
from rerun.catalog import DatasetEntry
from rerun.chunk import Chunk, LazyChunkStream, OptimizationProfile, RrdReader
from rerun.urdf import UrdfTree
from specialist_dataset import schema

__all__ = ["register_assets", "resolve_urdf_path", "urdf_sha", "write_gripper_asset"]


def resolve_urdf_path(explicit: Path | None) -> Path:
  """The gripper URDF: an explicit path, else the installed description asset."""
  if explicit is not None:
    return explicit
  try:
    from specialist_description import get_urdf_path

    return Path(get_urdf_path("specialist_handheld"))
  except ImportError as exc:
    raise SystemExit(
      f"the gripper URDF is unavailable ({exc}); pass --urdf-path"
    ) from exc


def urdf_sha(urdf_path: Path) -> str:
  """The URDF's sha256, the id every layer records the model by."""
  return sha256(urdf_path.read_bytes()).hexdigest()


def _gripper_chunks(urdf_path: Path, slot: str) -> list[Chunk]:
  """One slot's static model: meshes, fixed-joint edges and revolute links' frames."""
  tree = UrdfTree.from_file_path(
    str(urdf_path), entity_path_prefix=slot, frame_prefix=f"{slot}_"
  )
  gripper = f"/{schema.gripper(slot)}"
  # The loader escapes slashes in a prefix, so build at a one-segment root and move.
  model_root = f"/{slot}/{tree.name}"
  chunks = [
    chunk.with_entity_path(gripper + chunk.entity_path.removeprefix(model_root))
    for chunk in tree.stream(include_joint_transforms=False)
    .drop(content=f"{model_root}/collision_geometries/**")
    .to_chunks()
  ]
  fixed = [joint.name for joint in tree.joints() if joint.joint_type == "fixed"]
  edges = tree.compute_joint_transform_batches(
    pa.array([fixed]), pa.array([[0.0] * len(fixed)])
  ).to_pylist()[0]
  fields = ("translation", "quaternion", "parent_frame", "child_frame")
  columns = rr.Transform3D.columns(**{f: [edge[f] for edge in edges] for f in fields})
  tf_static = f"/{schema.gripper_part(slot, 'tf_static')}"
  chunks.append(Chunk.from_columns(tf_static, indexes=[], columns=columns))
  chunks += [
    Chunk.from_columns(
      f"/{schema.gripper_part(slot, joint.name)}",
      indexes=[],
      columns=rr.CoordinateFrame.columns(
        frame=[schema.link_frame(slot, tree.get_joint_child(joint).name)]
      ),
    )
    for joint in tree.joints()
    if joint.joint_type == "revolute"
  ]
  return chunks


def write_gripper_asset(urdf_path: Path, slots: list[str], dataset: Path) -> Path:
  """This URDF's asset under ``dataset/assets``, rewritten only when it lacks a slot."""
  path = schema.asset_rrd(dataset, f"gripper-{urdf_sha(urdf_path)[:12]}")
  if path.exists():
    chunks = RrdReader(str(path)).store().stream().to_chunks()
    held = {chunk.entity_path.split("/")[2] for chunk in chunks}
    if held >= set(slots):
      return path
    slots = sorted(held | set(slots))
  path.parent.mkdir(parents=True, exist_ok=True)
  store = LazyChunkStream.from_iter(
    chunk for slot in slots for chunk in _gripper_chunks(urdf_path, slot)
  ).collect(optimize=OptimizationProfile.OBJECT_STORE)
  # Concurrent solves write the same asset, so each writes its own file and renames.
  written = path.with_suffix(f".{os.getpid()}")
  try:
    store.write_rrd(
      written, application_id=schema.APPLICATION_ID, recording_id=path.stem
    )
    written.replace(path)
  finally:
    written.unlink(missing_ok=True)
  return path


def register_assets(entry: DatasetEntry, dataset: Path) -> None:
  """Register the model under ``dataset/assets`` and unregister the rest."""
  files = sorted((dataset / schema.ASSETS).glob("*.rrd"))
  if len(files) > 1:
    raise SystemExit(
      f"{dataset / schema.ASSETS} holds several models {[f.stem for f in files]}"
    )
  for stale in set(entry.assets()) - {f.stem for f in files}:
    entry.unregister_asset(stale)
  for f in files:
    entry.register_asset(f.resolve().as_uri())
  print(f"registered {len(files)} asset(s): {[f.stem for f in files]}", flush=True)
