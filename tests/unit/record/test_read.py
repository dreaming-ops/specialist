"""Static components return the newest write even when an older chunk sits later."""

from pathlib import Path

import numpy as np
import rerun as rr
from rerun.experimental import Chunk, send_chunks

from specialist.record import schema
from specialist.record.read import read_track
from specialist.record.session import SessionRecording


def _base_tag(translation: list[float]) -> Chunk:
  return Chunk.from_columns(
    "tracker_0/body/base_tag",
    indexes=[],
    columns=rr.Transform3D.columns(
      translation=[translation], quaternion=[[0.0, 0.0, 0.0, 1.0]]
    ),
  )


def _keyframes() -> Chunk:
  return Chunk.from_columns(
    "tracker_0/body_keyframes",
    indexes=[rr.TimeColumn(schema.TIMELINE, duration=[0.0])],
    columns=rr.Transform3D.columns(
      translation=[[0.0, 0.0, 0.0]], quaternion=[[0.0, 0.0, 0.0, 1.0]]
    ),
  )


def _write(rrd: Path, chunks: list[Chunk]) -> None:
  with rr.RecordingStream("test", recording_id="s") as stream:
    stream.save(str(rrd))
    send_chunks(chunks, recording=stream)


def test_base_tag_takes_the_newest_write_regardless_of_file_order(
  tmp_path: Path,
) -> None:
  older = _base_tag([1.0, 0.0, 0.0])
  newer = _base_tag([2.0, 0.0, 0.0])
  rrd = tmp_path / "layer.rrd"
  _write(rrd, [_keyframes(), newer, older])
  run = read_track(rrd)
  np.testing.assert_allclose(
    run.body_T_base_tag_by_tracker["tracker_0"][:3, 3], [2.0, 0.0, 0.0]
  )


def test_base_tag_takes_the_last_row_of_a_merged_chunk(tmp_path: Path) -> None:
  merged = Chunk.from_columns(
    "tracker_0/body/base_tag",
    indexes=[],
    columns=rr.Transform3D.columns(
      translation=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
      quaternion=[[0.0, 0.0, 0.0, 1.0]] * 2,
    ),
  )
  rrd = tmp_path / "layer.rrd"
  _write(rrd, [_keyframes(), merged])
  run = read_track(rrd)
  np.testing.assert_allclose(
    run.body_T_base_tag_by_tracker["tracker_0"][:3, 3], [2.0, 0.0, 0.0]
  )


def test_entity_row_takes_the_newest_write_regardless_of_file_order(
  tmp_path: Path,
) -> None:
  def calibration(fx: float) -> Chunk:
    return Chunk.from_columns(
      "calibration/tracker_0/front",
      indexes=[],
      columns=rr.AnyValues.columns(fx=[fx]),
    )

  older = calibration(1.0)
  newer = calibration(2.0)
  rrd = tmp_path / "base.rrd"
  _write(rrd, [newer, older])
  row = SessionRecording(rrd)._entity_row("calibration/tracker_0/front")
  assert float(np.asarray(row["fx"]).reshape(-1)[0]) == 2.0
