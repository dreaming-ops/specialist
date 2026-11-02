"""The estimate layer round-trips exactly through the layer writer and record reader."""

from pathlib import Path

import gtsam
import numpy as np
import pytest
from rerun.chunk import RrdReader
from specialist_dataset import schema
from specialist_dataset.read import read_keyframe_poses

from specialist.vi_ba.summary import CameraSummary
from tests.record_fixtures import camera_estimate, urdf_file, write_run


@pytest.fixture(scope="module")
def estimate_rrd(tmp_path_factory: pytest.TempPathFactory) -> Path:
  """The estimate layer of a solved two-camera run."""
  cameras = [camera_estimate("cam00"), camera_estimate("cam01", offset=1.0)]
  estimate, _viz = write_run(
    tmp_path_factory.mktemp("roundtrip"),
    "sess_x",
    cameras,
    per_camera=[
      CameraSummary(
        slot=c.slot,
        n_keyframes=4,
        anchor_obs=3,
        front_rmse=0.4,
        base_tag_id=c.base_tag_id,
        seat_rotation=(0.0, 0.0, 0.05),
        seat_anchor_obs=12,
        seat_cross_obs=7,
      )
      for c in cameras
    ],
    urdf_path=urdf_file(tmp_path_factory.mktemp("urdf")),
    frame_period=1.0 / 30.0,
  )
  return estimate


def test_estimate_roundtrip(estimate_rrd: Path) -> None:
  run = read_keyframe_poses(estimate_rrd)
  assert run.slots == ["cam00", "cam01"]
  for slot, offset in (("cam00", 0.0), ("cam01", 1.0)):
    times = run.kf_times_by_slot[slot]
    np.testing.assert_allclose(times, [0.0, 0.1, 0.2, 0.3], atol=1e-9)
    expected_t = np.array([[offset + 0.1 * i, 0.02 * i, 1.0] for i in range(4)])
    np.testing.assert_allclose(run.kf_translations_by_slot[slot], expected_t, atol=1e-6)
    expected_r = np.stack([gtsam.Rot3.Ry(0.05 * i).matrix() for i in range(4)])
    np.testing.assert_allclose(run.kf_rotations_by_slot[slot], expected_r, atol=1e-6)


def test_the_run_account_is_a_property_not_an_entity(estimate_rrd: Path) -> None:
  """A catalog reads it without opening the segment, so no entity carries a copy."""
  store = RrdReader(str(estimate_rrd)).store()
  entities, fields = set(), set()
  for chunk in store.stream().filter(is_static=True).to_chunks():
    entities.add(chunk.entity_path)
    if chunk.entity_path == "/__properties/estimate":
      fields |= {
        n for n in chunk.to_record_batch().schema.names if not n.startswith("rerun.")
      }
  assert f"/{schema.ESTIMATE}" not in entities
  assert {"final_graph_error", "solver_iterations", "worst_rmse", "git_sha"} <= fields


def _worst_rmse(estimate: Path) -> float:
  """The selector a catalog filters on, read back off the layer."""
  for chunk in RrdReader(str(estimate)).store().stream().to_chunks():
    if chunk.entity_path != "/__properties/estimate":
      continue
    batch = chunk.to_record_batch()
    if "worst_rmse" in batch.schema.names:
      column = batch.column(batch.schema.names.index("worst_rmse")).to_pylist()
      return float(np.asarray(column[0]).reshape(-1)[0])
  raise AssertionError("the estimate layer carries no worst_rmse")


def test_worst_rmse_is_the_worst_camera_that_was_measured(tmp_path: Path) -> None:
  """An unmeasured camera's NaN RMSE never wins and never poisons the comparison."""
  cameras = [camera_estimate("cam00"), camera_estimate("cam01", offset=1.0)]
  estimate, _viz = write_run(
    tmp_path,
    "sess_rmse",
    cameras,
    per_camera=[
      # The NaN camera comes first.
      CameraSummary(
        slot=cameras[0].slot,
        n_keyframes=4,
        anchor_obs=3,
        front_rmse=float("nan"),
        base_tag_id=2000,
        seat_rotation=(0.0, 0.0, 0.05),
        seat_anchor_obs=12,
        seat_cross_obs=7,
      ),
      CameraSummary(
        slot=cameras[1].slot,
        n_keyframes=4,
        anchor_obs=3,
        front_rmse=0.4,
        base_tag_id=2003,
        seat_rotation=(0.0, 0.0, 0.05),
        seat_anchor_obs=12,
        seat_cross_obs=7,
      ),
    ],
    urdf_path=urdf_file(tmp_path),
    frame_period=1.0 / 30.0,
  )
  assert _worst_rmse(estimate) == pytest.approx(0.4)


def test_worst_rmse_is_nan_when_no_camera_was_measured(tmp_path: Path) -> None:
  cameras = [camera_estimate("cam00")]
  estimate, _viz = write_run(
    tmp_path,
    "sess_unmeasured",
    cameras,
    per_camera=[
      CameraSummary(
        slot=cameras[0].slot,
        n_keyframes=4,
        anchor_obs=3,
        front_rmse=float("nan"),
        base_tag_id=2000,
        seat_rotation=(0.0, 0.0, 0.05),
        seat_anchor_obs=12,
        seat_cross_obs=7,
      )
    ],
    urdf_path=urdf_file(tmp_path),
    frame_period=1.0 / 30.0,
  )
  assert np.isnan(_worst_rmse(estimate))


def test_layer_identity_and_no_recording_info(estimate_rrd: Path) -> None:
  reader = RrdReader(str(estimate_rrd))
  assert [e.recording_id for e in reader.recordings()] == ["sess_x"]
  chunk_paths = {c.entity_path for c in reader.store().stream().to_chunks()}
  # send_properties=False writes no RecordingInfo chunk to collide on catalog merge.
  assert "/__properties" not in chunk_paths
  # Path parity with the schema builders.
  assert f"/{schema.state('cam00', 'pose')}" in chunk_paths
  assert f"/{schema.base_tag('cam00')}" in chunk_paths


def test_linear_velocity_lands_on_its_own_entity(estimate_rrd: Path) -> None:
  """The keyframe velocities sit on their own entity, one value per keyframe."""
  rows = 0
  for chunk in RrdReader(str(estimate_rrd)).store().stream().to_chunks():
    if chunk.entity_path != f"/{schema.state('cam00', 'lin_vel')}":
      continue
    batch = chunk.to_record_batch()
    assert "Transform3D:translation" not in batch.schema.names
    if "lin_vel" not in batch.schema.names:
      continue
    rows += chunk.num_rows
    value = np.asarray(
      batch.column(batch.schema.names.index("lin_vel")).to_pylist()[0]
    ).reshape(3)
    np.testing.assert_allclose(value, [0.1, 0.0, 0.0], atol=1e-12)
  assert rows == 4
