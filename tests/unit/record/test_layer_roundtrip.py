"""The track layer round-trips exactly through the layer writer and record reader."""

from pathlib import Path

import gtsam
import numpy as np
from rerun.experimental import RrdReader

from specialist.record import schema
from specialist.record.layer import write_track_layer
from specialist.record.read import read_track
from specialist.vi_ba.summary import CameraSummary, Summary
from specialist.vi_ba.viz import CameraVizData
from specialist.vi_ba.viz_build import RunOutputs

_BASE_TAG = gtsam.Pose3(gtsam.Rot3.Rz(0.3), gtsam.Point3(0.01, -0.02, 0.05))


def _tracker(name: str, offset: float) -> CameraVizData:
  times = np.array([0.0, 0.1, 0.2, 0.3])
  poses = [
    gtsam.Pose3(gtsam.Rot3.Ry(0.05 * i), gtsam.Point3(offset + 0.1 * i, 0.02 * i, 1.0))
    for i in range(4)
  ]
  return CameraVizData(
    name=name,
    body_poses=poses,
    keyframe_times=times,
    velocities=[np.array([0.1, 0.0, 0.0])] * 4,
    body_T_base_tag=_BASE_TAG,
  )


def _write(tmp_path: Path, session_id: str) -> Path:
  rrd = tmp_path / "layer.rrd"
  cameras = [_tracker("tracker_0", 0.0), _tracker("tracker_1", 1.0)]
  summary = Summary(
    sources=("a.insv", "b.insv"),
    slots=("tracker_0", "tracker_1"),
    cross_tracker_obs=7,
    initial_graph_error=10.0,
    final_graph_error=1.0,
    solver_iterations=5,
  )
  per_tracker = [
    CameraSummary(t.name, 4, 3, bounding_box_diagonal=0.5, path_length=0.3)
    for t in cameras
  ]
  outputs = RunOutputs(
    viz_trackers=cameras,
    per_tracker=per_tracker,
    world_T_anchor={0: gtsam.Pose3()},
  )
  write_track_layer(
    rrd_path=rrd,
    outputs=outputs,
    summary=summary,
    reference_tag_id=0,
    anchor_tag_size=0.1,
    session_id=session_id,
    # -1 disables tag masking; these cameras carry no tag detections anyway.
    mask_pad_px=-1,
    open_viewer=False,
  )
  return rrd


def test_track_roundtrip(tmp_path: Path) -> None:
  rrd = _write(tmp_path, session_id="sess_x")
  run = read_track(rrd)
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
      base_tag[:3, :3], _BASE_TAG.rotation().matrix(), atol=1e-6
    )
    np.testing.assert_allclose(
      base_tag[:3, 3], np.asarray(_BASE_TAG.translation()), atol=1e-6
    )


def test_layer_identity_and_no_recording_info(tmp_path: Path) -> None:
  rrd = _write(tmp_path, session_id="sess_x")
  reader = RrdReader(str(rrd))
  assert [e.recording_id for e in reader.recordings()] == ["sess_x"]
  chunk_paths = {c.entity_path for c in reader.store().stream().to_chunks()}
  # send_properties=False writes no RecordingInfo chunk to collide on catalog merge.
  assert "/__properties" not in chunk_paths
  assert "/__properties/solver" in chunk_paths
  assert "/__properties/solver" in chunk_paths
  assert "/__properties/quality_tracker_0" in chunk_paths
  # Path parity with the schema builders.
  assert f"/{schema.body_keyframes_entity('tracker_0')}" in chunk_paths
  assert f"/{schema.base_tag_entity('tracker_0')}" in chunk_paths


def test_velocity_column_lands(tmp_path: Path) -> None:
  rrd = _write(tmp_path, session_id="sess_y")
  rows = 0
  for chunk in RrdReader(str(rrd)).store().stream().to_chunks():
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
