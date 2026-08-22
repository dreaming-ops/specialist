"""The run summary the estimate layer carries."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CameraSummary", "Summary"]


@dataclass(frozen=True)
class CameraSummary:
  """How one camera's own reprojections closed, written to its ``fit`` entity."""

  slot: str
  n_keyframes: int
  anchor_obs: int
  # Reprojection RMSE in pixels over the whole take, per lens.
  front_rmse: float
  rear_rmse: float
  # The gripper base tag this camera solves against.
  base_tag_id: int
  # Rear-lens rotation off its factory block, radians.
  seat_rotation: tuple[float, float, float]
  # Rear-lens anchor sightings and cross sightings that measure the seat rotation.
  seat_anchor_obs: int
  seat_cross_obs: int


@dataclass(frozen=True)
class Summary:
  """The run's account of the estimate, written as the ``estimate`` property."""

  cross_camera_obs: int
  initial_graph_error: float
  final_graph_error: float
  solver_iterations: int
  # How far the solve moves the worst anchor off its seed (meters, radians).
  anchor_move_translation: float
  anchor_move_rotation: float
