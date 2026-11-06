"""Lowe-ratio ORB matches for many keyframe pairs at once on the GPU."""

from typing import Any

import numpy as np
import numpy.typing as npt

from specialist.vi_ba.features import Frame

__all__ = ["match_pairs"]

_PAIRS_PER_LAUNCH = 2048


def match_pairs(
  orb: list[Frame], pairs: list[tuple[int, int]], ratio: float
) -> list[npt.NDArray[np.int64]]:
  """Per pair, the ``(query, train)`` index pairs passing Lowe's ratio test."""
  import torch

  from specialist.vi_ba.gpu_flow import device_use

  try:
    with device_use():
      return _match_pairs_hamming(orb, pairs, ratio)
  finally:
    # The device buffers go back to the driver for the solver that follows.
    torch.cuda.empty_cache()


def _packed_rows(
  orb: list[Frame], device: Any
) -> tuple[Any, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
  """``(words, starts, counts)``: all descriptors as ``(N, 8)`` int32 device words."""
  import torch

  counts = np.asarray([f.descriptors.shape[0] for f in orb], dtype=np.int64)
  starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
  packed = torch.from_numpy(
    np.concatenate([f.descriptors for f in orb] or [np.empty((0, 32), dtype=np.uint8)])
  ).to(device)
  return packed.view(torch.int32), starts, counts


def _match_pairs_hamming(
  orb: list[Frame], pairs: list[tuple[int, int]], ratio: float
) -> list[npt.NDArray[np.int64]]:
  import torch

  from specialist.vi_ba import gpu_hamming

  device = torch.device("cuda")
  empty = np.empty((0, 2), dtype=np.int64)
  words, starts, counts = _packed_rows(orb, device)
  out: list[npt.NDArray[np.int64]] = [empty] * len(pairs)
  live = np.asarray(
    [
      (index, i, j)
      for index, (i, j) in enumerate(pairs)
      if counts[i] > 0 and counts[j] >= 2
    ],
    dtype=np.int64,
  ).reshape(-1, 3)
  with torch.no_grad():
    for begin in range(0, len(live), _PAIRS_PER_LAUNCH):
      batch = live[begin : begin + _PAIRS_PER_LAUNCH]
      q_count = counts[batch[:, 1]]
      out_first = np.concatenate(([0], np.cumsum(q_count)[:-1]))
      spans = [starts[batch[:, 1]], q_count, starts[batch[:, 2]], counts[batch[:, 2]]]
      best, at, second = gpu_hamming.top2(
        words, *(torch.as_tensor(a, dtype=torch.int64, device=device) for a in spans)
      )
      good = best.to(torch.float64) < ratio * second.to(torch.float64)
      hit = torch.nonzero(good).squeeze(1)
      rows = hit.cpu().numpy()
      train = at[hit].cpu().numpy().astype(np.int64)
      pair = np.searchsorted(out_first, rows, side="right") - 1
      query = rows - out_first[pair]
      bounds = np.searchsorted(pair, np.arange(len(batch) + 1))
      for k, (index, _, _) in enumerate(batch):
        a, b = bounds[k], bounds[k + 1]
        out[int(index)] = np.stack([query[a:b], train[a:b]], axis=1)
  return out
