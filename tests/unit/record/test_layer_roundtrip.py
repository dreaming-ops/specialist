"""The estimate layer round-trips exactly through the layer writer and record reader."""

from pathlib import Path

import gtsam
import numpy as np
import pytest
from rerun.chunk import RrdReader
from specialist_dataset import schema
from specialist_dataset.read import Recording, read_keyframe_poses

from tests.record_fixtures import camera_estimate, urdf_file, write_run


def test_estimate_roundtrip(estimated_run: tuple[Path, Path]) -> None:
  run = read_keyframe_poses(estimated_run[0])
  assert run.slots == ["cam00", "cam01"]
  for slot, offset in (("cam00", 0.0), ("cam01", 1.0)):
    times = run.kf_times_by_slot[slot]
    np.testing.assert_allclose(times, [0.0, 0.1, 0.2, 0.3], atol=1e-9)
    expected_t = np.array([[offset + 0.1 * i, 0.02 * i, 1.0] for i in range(4)])
    np.testing.assert_allclose(run.kf_translations_by_slot[slot], expected_t, atol=1e-6)
    expected_r = np.stack([gtsam.Rot3.Ry(0.05 * i).matrix() for i in range(4)])
    np.testing.assert_allclose(run.kf_rotations_by_slot[slot], expected_r, atol=1e-6)


def test_the_run_account_is_a_property_not_an_entity(
  estimated_run: tuple[Path, Path],
) -> None:
  """A catalog reads it without opening the segment, so no entity carries a copy."""
  columns = RrdReader(str(estimated_run[0])).store().schema().component_columns()
  fields = {
    c.component
    for c in columns
    if c.entity_path == "/__properties/estimate" and c.is_static
  }
  assert f"/{schema.ESTIMATE}" not in {c.entity_path for c in columns}
  assert {"final_graph_error", "solver_iterations", "worst_rmse", "git_sha"} <= fields


@pytest.mark.parametrize(
  ("front_rmse", "worst"), [([float("nan"), 0.4], 0.4), ([float("nan")], float("nan"))]
)
def test_worst_rmse_is_the_worst_camera_that_was_measured(
  tmp_path: Path, front_rmse: list[float], worst: float
) -> None:
  """An unmeasured camera's NaN RMSE never wins and never poisons the comparison."""
  cameras = [camera_estimate("cam00"), camera_estimate("cam01", offset=1.0)]
  estimate, _viz = write_run(
    tmp_path,
    "sess",
    cameras[: len(front_rmse)],
    front_rmse=front_rmse,
    urdf_path=urdf_file(tmp_path),
    frame_period=1.0 / 30.0,
  )
  with Recording(estimate) as recording:
    value = recording.property(schema.ESTIMATE)["worst_rmse"]
  np.testing.assert_allclose(np.asarray(value).reshape(-1)[0], worst)


def test_layer_identity_and_no_recording_info(estimated_run: tuple[Path, Path]) -> None:
  reader = RrdReader(str(estimated_run[0]))
  assert [e.recording_id for e in reader.recordings()] == ["sess_route"]
  chunk_paths = {c.entity_path for c in reader.store().stream().to_chunks()}
  # No RecordingInfo chunk collides with the base segment on catalog merge.
  assert "/__properties" not in chunk_paths
  # Path parity with the schema builders.
  assert f"/{schema.state('cam00', 'pose')}" in chunk_paths
  assert f"/{schema.base_tag('cam00')}" in chunk_paths


def test_linear_velocity_lands_on_its_own_entity(
  estimated_run: tuple[Path, Path],
) -> None:
  """The keyframe velocities sit on their own entity, one value per keyframe."""
  rows = 0
  for chunk in RrdReader(str(estimated_run[0])).store().stream().to_chunks():
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
