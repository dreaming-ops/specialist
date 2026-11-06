"""The lens as the policy sees it: block-averaged and masked to the fisheye disc."""

import numpy as np
import numpy.typing as npt

Uint8Array = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]

__all__ = ["MASK_RADIUS", "block_mean", "disc_mask", "to_image"]

# The fisheye disc's radius as a fraction of the lens width.
MASK_RADIUS = 0.5


def block_mean(image: Uint8Array, scale: int) -> Uint8Array:
  """``image`` averaged over ``scale x scale`` blocks, rounded half up."""
  height, width = image.shape[:2]
  if scale < 1 or height % scale or width % scale:
    raise ValueError(f"scale {scale} does not divide the {width}x{height} image")
  rows = image.reshape(height // scale, scale, width, *image.shape[2:])
  total = rows.sum(axis=1, dtype=np.uint32)
  total = total.reshape(height // scale, width // scale, scale, *image.shape[2:])
  total = total.sum(axis=2, dtype=np.uint32)
  out: Uint8Array = ((total + scale * scale // 2) // (scale * scale)).astype(np.uint8)
  return out


def disc_mask(height: int, width: int, radius: float = MASK_RADIUS) -> BoolArray:
  """Pixels whose centers lie within ``radius`` widths of the frame center."""
  x = np.arange(width) + 0.5 - width / 2.0
  y = np.arange(height) + 0.5 - height / 2.0
  out: BoolArray = x[None, :] ** 2 + y[:, None] ** 2 <= (radius * width) ** 2
  return out


def to_image(image: Uint8Array, scale: int, radius: float = MASK_RADIUS) -> Uint8Array:
  """One lens frame as the policy sees it, in the frame's own shape."""
  small = block_mean(image, scale)
  small[~disc_mask(small.shape[0], small.shape[1], radius)] = 0
  return small
