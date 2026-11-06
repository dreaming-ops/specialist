"""Triton two-nearest Hamming search; ties go to the lower train index, as in cv2."""

from typing import Any

import triton
import triton.language as tl
from triton.language.extra import libdevice

# Query rows per program and train rows per inner step.
_BLOCK_Q = 32
_BLOCK_T = 64
# Larger than any Hamming distance; marks a missing train row.
_FAR = 1 << 30


@triton.jit  # type: ignore[untyped-decorator]
def _top2_kernel(  # type: ignore[no-untyped-def]
  words_ptr,
  q_first_ptr,
  q_count_ptr,
  t_first_ptr,
  t_count_ptr,
  out_first_ptr,
  best_ptr,
  index_ptr,
  second_ptr,
  FAR: tl.constexpr,
  BLOCK_Q: tl.constexpr,
  BLOCK_T: tl.constexpr,
):
  pair = tl.program_id(1)
  n_q = tl.load(q_count_ptr + pair)
  local = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
  q_mask = local < n_q
  rows = tl.load(q_first_ptr + pair) + local
  n_t = tl.load(t_count_ptr + pair)
  t_ptr = words_ptr + tl.load(t_first_ptr + pair) * 8
  slot = tl.load(out_first_ptr + pair) + local
  q0 = tl.load(words_ptr + rows * 8 + 0, mask=q_mask, other=0)
  q1 = tl.load(words_ptr + rows * 8 + 1, mask=q_mask, other=0)
  q2 = tl.load(words_ptr + rows * 8 + 2, mask=q_mask, other=0)
  q3 = tl.load(words_ptr + rows * 8 + 3, mask=q_mask, other=0)
  q4 = tl.load(words_ptr + rows * 8 + 4, mask=q_mask, other=0)
  q5 = tl.load(words_ptr + rows * 8 + 5, mask=q_mask, other=0)
  q6 = tl.load(words_ptr + rows * 8 + 6, mask=q_mask, other=0)
  q7 = tl.load(words_ptr + rows * 8 + 7, mask=q_mask, other=0)
  best = tl.full((BLOCK_Q,), FAR, tl.int32)
  second = tl.full((BLOCK_Q,), FAR, tl.int32)
  index = tl.zeros((BLOCK_Q,), tl.int32)
  lanes = tl.arange(0, BLOCK_T)
  for start in range(0, n_t, BLOCK_T):
    cols = start + lanes
    t_mask = cols < n_t
    t0 = tl.load(t_ptr + cols * 8 + 0, mask=t_mask, other=0)
    t1 = tl.load(t_ptr + cols * 8 + 1, mask=t_mask, other=0)
    t2 = tl.load(t_ptr + cols * 8 + 2, mask=t_mask, other=0)
    t3 = tl.load(t_ptr + cols * 8 + 3, mask=t_mask, other=0)
    t4 = tl.load(t_ptr + cols * 8 + 4, mask=t_mask, other=0)
    t5 = tl.load(t_ptr + cols * 8 + 5, mask=t_mask, other=0)
    t6 = tl.load(t_ptr + cols * 8 + 6, mask=t_mask, other=0)
    t7 = tl.load(t_ptr + cols * 8 + 7, mask=t_mask, other=0)
    d = (
      libdevice.popc(q0[:, None] ^ t0[None, :])
      + libdevice.popc(q1[:, None] ^ t1[None, :])
      + libdevice.popc(q2[:, None] ^ t2[None, :])
      + libdevice.popc(q3[:, None] ^ t3[None, :])
      + libdevice.popc(q4[:, None] ^ t4[None, :])
      + libdevice.popc(q5[:, None] ^ t5[None, :])
      + libdevice.popc(q6[:, None] ^ t6[None, :])
      + libdevice.popc(q7[:, None] ^ t7[None, :])
    )
    d = tl.where(t_mask[None, :], d, FAR)
    chunk_best = tl.min(d, axis=1)
    chunk_index = tl.argmin(d, axis=1, tie_break_left=True)
    rest = tl.where(lanes[None, :] == chunk_index[:, None], FAR, d)
    chunk_second = tl.min(rest, axis=1)
    better = chunk_best < best
    second = tl.where(
      better, tl.minimum(best, chunk_second), tl.minimum(second, chunk_best)
    )
    index = tl.where(better, (start + chunk_index).to(tl.int32), index)
    best = tl.where(better, chunk_best, best)
  tl.store(best_ptr + slot, best, mask=q_mask)
  tl.store(index_ptr + slot, index, mask=q_mask)
  tl.store(second_ptr + slot, second, mask=q_mask)


def top2(
  words: Any, q_first: Any, q_count: Any, t_first: Any, t_count: Any
) -> tuple[Any, Any, Any]:
  """``(best, index, second)`` per query row, every pair's rows end to end."""
  import torch

  n_pairs = q_first.shape[0]
  out_first = torch.cumsum(q_count, 0) - q_count
  n_out = int(q_count.sum())
  best = torch.empty(n_out, dtype=torch.int32, device=words.device)
  index = torch.empty(n_out, dtype=torch.int32, device=words.device)
  second = torch.empty(n_out, dtype=torch.int32, device=words.device)
  if n_out == 0:
    return best, index, second
  grid = (triton.cdiv(int(q_count.max()), _BLOCK_Q), n_pairs)
  _top2_kernel[grid](
    words,
    q_first,
    q_count,
    t_first,
    t_count,
    out_first,
    best,
    index,
    second,
    FAR=_FAR,
    BLOCK_Q=_BLOCK_Q,
    BLOCK_T=_BLOCK_T,
  )
  return best, index, second
