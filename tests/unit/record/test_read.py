"""Static components return the newest write even when an older chunk sits later."""

from pathlib import Path

import numpy as np
import rerun as rr
from rerun.experimental import Chunk, send_chunks

from specialist.record import schema
from specialist.record.session import SessionRecording


def _base_tag(translation: list[float]) -> Chunk:
  return Chunk.from_columns(
    schema.base_tag("cam00"),
    indexes=[],
    columns=rr.Transform3D.columns(
      translation=[translation], quaternion=[[0.0, 0.0, 0.0, 1.0]]
    ),
  )


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


def test_entity_row_takes_the_newest_write_regardless_of_file_order(
  tmp_path: Path,
) -> None:
  def calibration(fx: float) -> Chunk:
    return Chunk.from_columns(
      schema.lens("cam00", "front"),
      indexes=[],
      columns=rr.AnyValues.columns(fx=[fx]),
    )

  older = calibration(1.0)
  newer = calibration(2.0)
  rrd = tmp_path / "base.rrd"
  _write(rrd, [newer, older])
  row = SessionRecording(rrd)._entity_row(schema.lens("cam00", "front"))
  assert float(np.asarray(row["fx"]).reshape(-1)[0]) == 2.0
