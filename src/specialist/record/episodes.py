"""Review layer: green candidate spans on a time plot beside the source videos."""

from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb
from specialist_dataset import schema
from specialist_dataset.episodes import Episode, Marker


def write_review(
  path: Path,
  session_id: str,
  slots: list[str],
  episodes: list[Episode],
  markers: list[Marker],
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with rr.RecordingStream(schema.APPLICATION_ID, recording_id=session_id) as stream:
    stream.save(str(path))
    for e in episodes:
      entity = f"episodes/{e.episode_id}"
      stream.log(
        entity,
        rr.SeriesLines(colors=[(40, 210, 100, 100)], widths=16, names=[e.episode_id]),
        static=True,
      )
      stream.send_columns(
        entity,
        indexes=[
          rr.TimeColumn(schema.TIMELINE, duration=[e.capture_start, e.task_end])
        ],
        columns=rr.Scalars.columns(scalars=[1.0, 1.0]),
      )
    for index, marker in enumerate(markers, start=1):
      # Keep every boundary visible even when an odd count prevents pairing.
      entity = f"episodes/markers/{index:04d}"
      stream.log(
        entity,
        rr.SeriesLines(
          colors=[(40, 210, 100, 150)], widths=8, names=[f"Triple tap {index}"]
        ),
        static=True,
      )
      stream.send_columns(
        entity,
        indexes=[rr.TimeColumn(schema.TIMELINE, duration=[marker.first, marker.last])],
        columns=rr.Scalars.columns(scalars=[0.5, 0.5]),
      )
      stream.set_time(schema.TIMELINE, duration=marker.first)
      stream.log("episode_markers", rr.TextLog(f"Triple tap {marker.first:.3f}s"))
    stream.send_blueprint(
      rrb.Blueprint(
        rrb.Vertical(
          rrb.Horizontal(
            *(
              rrb.Spatial2DView(
                origin=f"/{schema.lens(s, lens)}",
                name=f"{s}/{lens}",
              )
              for s in slots
              for lens in schema.LENSES
            )
          ),
          rrb.TimeSeriesView(
            origin="/episodes", name="Candidate episodes (review required)"
          ),
          row_shares=[4, 1],
        ),
        rrb.TimePanel(timeline=schema.TIMELINE, state=rrb.PanelState.Expanded),
      )
    )
