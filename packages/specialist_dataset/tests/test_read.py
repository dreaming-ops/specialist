"""Reading a layer back: statics, lenses, grippers and anchors."""

from pathlib import Path

import pytest
import rerun as rr
from rerun.experimental import Chunk, send_chunks
from specialist_dataset import schema
from specialist_dataset.read import (
  Recording,
  read_anchors,
  read_grippers,
  read_lens,
  read_tag_ids,
)


def _lens_chunk() -> Chunk:
  """One stored lens entity: intrinsics in a 2880-wide stitched frame."""
  return Chunk.from_columns(
    schema.lens("cam00", "front"),
    indexes=[],
    columns=[
      *rr.Transform3D.columns(
        translation=[[0.0, 0.0, 0.0]], quaternion=[[0.0, 0.0, 0.0, 1.0]]
      ),
      *rr.AnyValues.columns(
        fx=[1000.0], fy=[1000.0], cx=[720.0], cy=[720.0],
        k1=[0.0], k2=[0.0], k3=[0.0], p1=[0.0], p2=[0.0],
        xi=[1.9], frame_width=[2880.0],
      ),
    ],
  )  # fmt: skip


def _keyframes() -> Chunk:
  return Chunk.from_columns(
    schema.state("cam00", "pose"),
    indexes=[rr.TimeColumn(schema.TIMELINE, duration=[0.0])],
    columns=rr.Transform3D.columns(
      translation=[[0.0, 0.0, 0.0]], quaternion=[[0.0, 0.0, 0.0, 1.0]]
    ),
  )


def _write(rrd: Path, chunks: list[Chunk]) -> None:
  with rr.RecordingStream("test", recording_id="s") as stream:
    stream.save(str(rrd))
    send_chunks(chunks, recording=stream)


def test_a_static_written_twice_resolves_to_the_last_write(tmp_path: Path) -> None:
  def calibration(fx: float) -> Chunk:
    return Chunk.from_columns(
      schema.lens("cam00", "front"),
      indexes=[],
      columns=rr.AnyValues.columns(fx=[fx]),
    )

  rrd = tmp_path / "base.rrd"
  _write(rrd, [calibration(2.0), calibration(1.0)])
  row = Recording(rrd).row(schema.lens("cam00", "front"))
  assert list(row["fx"]) == [1.0]


def test_a_lens_reads_back_at_the_decode_resolution(tmp_path: Path) -> None:
  rrd = tmp_path / "base.rrd"
  with rr.RecordingStream("test", recording_id="s") as stream:
    stream.save(str(rrd))
    send_chunks([_lens_chunk()], recording=stream)
  lens = read_lens(rrd, "cam00", "front", 1440)
  assert lens.params.fx == pytest.approx(1000.0)
  assert lens.body_T_lens.shape == (4, 4)


def _write_stamped(rrd: Path, chunks: list[Chunk]) -> None:
  with rr.RecordingStream("test", recording_id="s") as stream:
    stream.save(str(rrd))
    send_chunks(chunks, recording=stream)


def _angles(side: str, times: list[float], values: list[float]) -> Chunk:
  return Chunk.from_columns(
    schema.gripper_part("cam00", f"{side}_finger"),
    indexes=[rr.TimeColumn(schema.TIMELINE, duration=times)],
    columns=rr.AnyValues.columns(angle=values),
  )


def _tagged(entity: str, tag_id: int) -> Chunk:
  return Chunk.from_columns(
    entity,
    indexes=[],
    columns=[
      *rr.Transform3D.columns(
        translation=[[1.0, 2.0, 3.0]], quaternion=[[0.0, 0.0, 0.0, 1.0]]
      ),
      *rr.AnyValues.columns(tag_id=[tag_id]),
    ],
  )  # fmt: skip


def _anchor(tag_id: int, is_reference: bool) -> Chunk:
  return Chunk.from_columns(
    f"{schema.WORLD}/anchor/{tag_id}",
    indexes=[],
    columns=[
      *rr.Transform3D.columns(
        translation=[[float(tag_id), 0.0, 0.0]], quaternion=[[0.0, 0.0, 0.0, 1.0]]
      ),
      *rr.AnyValues.columns(tag_id=[tag_id], is_reference=[is_reference]),
    ],
  )  # fmt: skip


def test_a_finger_series_spanning_chunks_reads_whole_and_in_time_order(
  tmp_path: Path,
) -> None:
  """It writes the later half first: a writer may split a column in any chunk order."""
  rrd = tmp_path / "estimate.rrd"
  _write_stamped(
    rrd,
    [
      _keyframes(),
      _tagged(schema.base_tag("cam00"), 2000),
      _angles("left", [2.0, 3.0], [0.2, 0.3]),
      _angles("left", [0.0, 1.0], [0.0, 0.1]),
      _angles("right", [0.0, 1.0, 2.0, 3.0], [0.0, 0.1, 0.2, 0.3]),
    ],
  )
  times, angles = read_grippers(rrd)["cam00"].fingers["left"]
  assert list(times) == [0.0, 1.0, 2.0, 3.0]
  assert list(angles) == pytest.approx([0.0, 0.1, 0.2, 0.3])


def test_a_gripper_carries_the_base_tag_pose(tmp_path: Path) -> None:
  rrd = tmp_path / "estimate.rrd"
  _write_stamped(
    rrd,
    [
      _keyframes(),
      _tagged(schema.base_tag("cam00"), 2000),
      _angles("left", [0.0], [0.0]),
      _angles("right", [0.0], [0.0]),
    ],
  )
  base_tag = read_grippers(rrd)["cam00"].base_tag
  assert base_tag.shape == (4, 4)
  assert list(base_tag[:3, 3]) == pytest.approx([1.0, 2.0, 3.0])


def test_anchors_name_the_one_the_solve_held_fixed(tmp_path: Path) -> None:
  rrd = tmp_path / "estimate.rrd"
  _write_stamped(rrd, [_keyframes(), _anchor(0, True), _anchor(1, False)])
  anchors = read_anchors(rrd)
  assert anchors.reference == 0
  assert sorted(anchors.poses) == [0, 1]
  assert anchors.poses[1][0, 3] == pytest.approx(1.0)


def test_a_joint_carrying_no_tag_is_omitted(tmp_path: Path) -> None:
  rrd = tmp_path / "estimate.rrd"
  _write_stamped(
    rrd,
    [
      _keyframes(),
      _tagged(schema.base_tag("cam00"), 2000),
      _tagged(schema.joint_tag("cam00", "left_finger"), 0),
      _tagged(schema.joint_tag("cam00", "right_finger"), 2002),
    ],
  )
  assert read_tag_ids(rrd)["cam00"] == {"base_tag": 2000, "right_finger_tag": 2002}
