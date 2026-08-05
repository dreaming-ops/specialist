"""The run summary the track layer carries as properties."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Summary", "CameraSummary"]


@dataclass(frozen=True)
class CameraSummary:
  """One tracker's line in the run summary."""

  name: str
  n_keyframes: int
  anchor_obs: int
  bounding_box_diagonal: float
  path_length: float


@dataclass(frozen=True)
class Summary:
  """The run's header: what it processed and the solve's account of how it went."""

  sources: tuple[str, ...]
  slots: tuple[str, ...]
  cross_tracker_obs: int
  initial_graph_error: float
  final_graph_error: float
  solver_iterations: int
  # Worst secondary-anchor move off its seed (meters, radians); NaN when none exists.
  secondary_anchor_move_translation: float = float("nan")
  secondary_anchor_move_rotation: float = float("nan")
  # Overridden defaults as "dotted.name=value"; two empty tuples make layers comparable.
  non_default_flags: tuple[str, ...] = ()
