"""The Rerun review layer round-trips with source-clock boundaries."""

from pathlib import Path

from rerun.chunk import RrdReader
from specialist_dataset import schema
from specialist_dataset.episodes import Episode, Marker

from specialist.record.episodes import write_review


def test_review_layer_roundtrip(tmp_path: Path) -> None:
  path = tmp_path / "review.rrd"
  write_review(
    path,
    "session",
    ["cam00", "cam01"],
    [Episode("ep0001", 3, 12, 13), Episode("ep0002", 30, 40, 41)],
    [Marker(2, 2.6), Marker(12, 12.6)],
  )
  assert RrdReader(str(path)).recordings()[0].recording_id == "session"
  chunks = list(RrdReader(str(path)).stream())
  for episode_id, start, end in [("ep0001", 3, 12), ("ep0002", 30, 40)]:
    batches = [
      c.to_record_batch()
      for c in chunks
      if c.entity_path.strip("/") == f"episodes/{episode_id}" and not c.is_static
    ]
    times = [v.value / 1e9 for b in batches for v in b.column(schema.TIMELINE)]
    assert times == [start, end]
  assert schema.TIMELINE == "video_time"
