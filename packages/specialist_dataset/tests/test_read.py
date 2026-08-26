"""Reading a layer back: statics, lenses, grippers and anchors."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from rerun.experimental import Chunk, RrdReader, send_chunks
from specialist_dataset import schema
from specialist_dataset.read import newest_values, read_lens


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
  store = RrdReader(str(rrd)).store()
  content = f"/{schema.lens('cam00', 'front')}"
  row = newest_values(store.stream().filter(content=content).to_chunks())
  assert float(np.asarray(row["fx"]).reshape(-1)[0]) == 2.0


def test_a_lens_reads_back_at_the_decode_resolution(tmp_path: Path) -> None:
  rrd = tmp_path / "base.rrd"
  with rr.RecordingStream("test", recording_id="s") as stream:
    stream.save(str(rrd))
    send_chunks([_lens_chunk()], recording=stream)
  lens = read_lens(rrd, "cam00", "front", 1440)
  assert lens.params.fx == pytest.approx(1000.0)
  assert lens.body_T_lens.shape == (4, 4)
