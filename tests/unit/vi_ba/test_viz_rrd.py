"""The rrd written by the layer writer round-trips through the reader."""

from pathlib import Path

import gtsam
import numpy as np
from rerun.experimental import RrdReader

from specialist.record.layer import write_track_layer
from specialist.vi_ba.summary import Summary
from specialist.vi_ba.viz import CameraVizData, _overlay_index
from specialist.vi_ba.viz_build import RunOutputs


def _tracker(name: str, with_highrate: bool) -> CameraVizData:
  times = np.array([0.0, 0.1, 0.2])
  poses = [gtsam.Pose3(gtsam.Rot3(), np.array([0.1 * i, 0.0, 0.0])) for i in range(3)]
  data = CameraVizData(name=name, body_poses=poses, keyframe_times=times)
  if with_highrate:
    hr_t = np.linspace(0.0, 0.2, 41)
    zeros = np.zeros_like(hr_t)
    data.highrate_times = hr_t
    data.highrate_positions = np.stack([hr_t, zeros, zeros], axis=1)
    data.highrate_quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(hr_t), 1))
    data.highrate_vels = np.ones((len(hr_t), 3))
  return data


def test_rrd_roundtrip(tmp_path: Path) -> None:
  rrd = tmp_path / "run.rrd"
  cameras = [_tracker("tracker_0", True), _tracker("tracker_1", False)]
  outputs = RunOutputs(
    viz_trackers=cameras,
    per_tracker=[],
    world_T_anchor={0: gtsam.Pose3()},
  )
  summary = Summary(
    sources=("a.insv",),
    slots=("tracker_0", "tracker_1"),
    cross_tracker_obs=0,
    initial_graph_error=1.0,
    final_graph_error=0.5,
    solver_iterations=3,
  )
  write_track_layer(
    rrd_path=rrd,
    outputs=outputs,
    summary=summary,
    reference_tag_id=0,
    anchor_tag_size=0.1,
    session_id="sess_viz",
    # -1 disables tag masking; these cameras carry no tag detections anyway.
    mask_pad_px=-1,
    open_viewer=False,
  )
  reader = RrdReader(str(rrd))
  assert len(reader.recordings()) == 1
  paths = set(reader.store().schema().entity_paths())
  expected = {
    "/world",
    "/world/anchor_tag",
    "/tracker_0/body",
    "/tracker_0/body_keyframes",
    "/tracker_0/trajectory",
    "/tracker_1/body",
    "/tracker_1/trajectory",
  }
  missing = expected - paths
  assert not missing, f"missing entities {sorted(missing)} in {sorted(paths)}"


def test_overlay_index_bounds_each_keyframe_to_one_frame() -> None:
  """Clearing rows sit a frame period after each keyframe, on the video clock alone."""
  kf_times = np.array([0.0, 0.1, 0.2])
  kf_col = object()
  data, clear = _overlay_index(kf_times, 1.0 / 30.0, [], kf_col)  # type: ignore[arg-type]
  assert len(data) == 2 and data[1] is kf_col
  assert clear is not None and len(clear) == 1
  fallback = [object()]
  same, none = _overlay_index(None, 1.0 / 30.0, fallback, None)  # type: ignore[arg-type]
  assert same is fallback and none is None
