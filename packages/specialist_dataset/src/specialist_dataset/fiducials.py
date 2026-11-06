"""The dataset fiducial convention: the tag family, and where corners sit."""

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]

__all__ = ["TAG_FAMILY", "tag_corners_in_frame", "tag_corners_local"]

TAG_FAMILY = "tagStandard41h12"


def tag_corners_local(tag_size: float) -> Float64Array:
  """``(4, 3)`` corners in pupil_apriltags order BL, BR, TR, TL; tag +x right, +y up."""
  half = float(tag_size / 2.0)
  return np.array(
    [
      [-half, -half, 0.0],
      [+half, -half, 0.0],
      [+half, +half, 0.0],
      [-half, +half, 0.0],
    ],
    dtype=np.float64,
  )


def tag_corners_in_frame(frame_T_tag: Float64Array, tag_size: float) -> Float64Array:
  """A tag's ``(4, 3)`` corners in the frame of ``frame_T_tag``."""
  return tag_corners_local(tag_size) @ frame_T_tag[:3, :3].T + frame_T_tag[:3, 3]
