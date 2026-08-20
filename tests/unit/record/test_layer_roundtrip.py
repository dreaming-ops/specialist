"""The track layer round-trips exactly through the layer writer and record reader."""

from pathlib import Path

import gtsam
import numpy as np
import pytest
from rerun.experimental import RrdReader

from specialist.record import schema
from specialist.record.read import read_track
from specialist.vi_ba.summary import CameraSummary
from tests.record_fixtures import BASE_TAG, viz_tracker, write_run


@pytest.fixture(scope="module")
def track_rrd(tmp_path_factory: pytest.TempPathFactory) -> Path:
  """The track layer of a solved two-tracker run."""
  cameras = [viz_tracker("tracker_0"), viz_tracker("tracker_1", offset=1.0)]
  track, _viz = write_run(
    tmp_path_factory.mktemp("roundtrip"),
    "sess_x",
    cameras,
    per_tracker=[
      CameraSummary(t.name, 4, 3, bounding_box_diagonal=0.5, path_length=0.3)
      for t in cameras
    ],
  )
  return track


def test_track_roundtrip(track_rrd: Path) -> None:
  run = read_track(track_rrd)
  assert run.slots == ["tracker_0", "tracker_1"]
  for name, offset in (("tracker_0", 0.0), ("tracker_1", 1.0)):
    times = run.kf_times_by_tracker[name]
    np.testing.assert_allclose(times, [0.0, 0.1, 0.2, 0.3], atol=1e-9)
    expected_t = np.array([[offset + 0.1 * i, 0.02 * i, 1.0] for i in range(4)])
    np.testing.assert_allclose(
      run.kf_translations_by_tracker[name], expected_t, atol=1e-6
    )
    expected_r = np.stack([gtsam.Rot3.Ry(0.05 * i).matrix() for i in range(4)])
    np.testing.assert_allclose(run.kf_rotations_by_tracker[name], expected_r, atol=1e-6)
    base_tag = run.body_T_base_tag_by_tracker[name]
    np.testing.assert_allclose(
      base_tag[:3, :3], BASE_TAG.rotation().matrix(), atol=1e-6
    )
    np.testing.assert_allclose(
      base_tag[:3, 3], np.asarray(BASE_TAG.translation()), atol=1e-6
    )


def test_layer_identity_and_no_recording_info(track_rrd: Path) -> None:
  reader = RrdReader(str(track_rrd))
  assert [e.recording_id for e in reader.recordings()] == ["sess_x"]
  chunk_paths = {c.entity_path for c in reader.store().stream().to_chunks()}
  # send_properties=False writes no RecordingInfo chunk to collide on catalog merge.
  assert "/__properties" not in chunk_paths
  assert "/__properties/solver" in chunk_paths
  assert "/__properties/quality_tracker_0" in chunk_paths
  # Path parity with the schema builders.
  assert f"/{schema.body_keyframes_entity('tracker_0')}" in chunk_paths
  assert f"/{schema.base_tag_entity('tracker_0')}" in chunk_paths


def test_velocity_column_lands(track_rrd: Path) -> None:
  rows = 0
  for chunk in RrdReader(str(track_rrd)).store().stream().to_chunks():
    if chunk.entity_path != "/tracker_0/body" or chunk.is_static:
      continue
    batch = chunk.to_record_batch()
    if "velocity" in batch.schema.names:
      rows += chunk.num_rows
      value = np.asarray(
        batch.column(batch.schema.names.index("velocity")).to_pylist()[0]
      ).reshape(3)
      np.testing.assert_allclose(value, [0.1, 0.0, 0.0], atol=1e-12)
  assert rows == 4
