"""Frame decoding out of a base recording's video stream."""

from pathlib import Path

import pyarrow as pa
import pytest
from specialist_dataset import frames, schema


def test_lens_video_refuses_markerless_streams(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  table = pa.table({schema.TIMELINE: pa.array([0, 1], type=pa.duration("ns"))})
  monkeypatch.setattr(
    frames, "_scan_video", lambda path: {"camera/cam00/front/video": (table, [])}
  )
  with pytest.raises(RuntimeError, match="no sync-sample markers"):
    frames.LensVideo(Path("base.rrd"), "cam00", "front")


def _time_batch(times: list[int], values: list[int]) -> pa.RecordBatch:
  return pa.record_batch(
    {
      schema.TIMELINE: pa.array(times, type=pa.duration("ns")),
      "v": pa.array(values, type=pa.int64()),
    }
  )


def test_time_ordered_returns_a_sorted_table_untouched() -> None:
  table = pa.Table.from_batches(
    [_time_batch([0, 1, 2], [0, 1, 2]), _time_batch([3, 4, 5], [3, 4, 5])]
  )
  assert frames._time_ordered(table) is table


def test_time_ordered_sorts_a_cross_batch_inversion() -> None:
  table = pa.Table.from_batches(
    [_time_batch([0, 1, 2], [0, 1, 2]), _time_batch([1, 7, 8], [9, 7, 8])]
  )
  ordered = frames._time_ordered(table)
  assert ordered.equals(table.sort_by(schema.TIMELINE))
  assert ordered.column(schema.TIMELINE).to_pylist() == sorted(
    table.column(schema.TIMELINE).to_pylist()
  )
