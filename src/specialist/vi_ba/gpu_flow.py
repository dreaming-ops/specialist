"""GPU (PyTorch) DIS flow: the same stream ``features._iter_dis_flows`` yields."""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Iterator, Sequence
from contextlib import suppress
from typing import Any

import numpy as np
import numpy.typing as npt

_FlowPair = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]

__all__ = ["unavailable_reason", "prime", "shutdown"]

# Batch from card memory and frame area, budgets in bytes; skip a GPU under _MIN_FREE.
_MAX_BATCH = 24
_MIN_FREE = 8 * 2**30
_RESERVE = 3 * 2**30
_PAIR_MEMORY = 1.15 * 2**30
_REFERENCE_PIXELS = 2880 * 2880

# Distinct GPUs one process claims; past the cap, extra streams share the most-free GPU.
_MAX_GPUS = 8

# Page-locked host staging with non-blocking copies.
_PINNED = True

_claim_lock = threading.Lock()
_backend: tuple[Any, Any] | None = None
_claimed: dict[int, int] = {}  # device_index -> streams assigned in this process
_bad: set[int] = set()  # GPUs that failed to warm (shared-box context errors)
_engines: list[_Engine] = []
# The stream a prep thread has already warmed, reused across its capture's lenses.
_stream = threading.local()


def _import_backend() -> tuple[Any, Any]:
  """Torch and the vendored wavefront, cached; raises ImportError without torch."""
  global _backend
  if _backend is not None:
    return _backend
  # Must be set before torch initializes its CUDA allocator.
  os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
  import torch
  import torch._dynamo

  # Dynamo caches one compiled variant per GPU; the default recompile limit is too low.
  for attr in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit"):
    if hasattr(torch._dynamo.config, attr):
      setattr(torch._dynamo.config, attr, 256)

  from specialist.vi_ba.gpu_dis.torch_dis_prop_fast import TorchDISBatchedPropFast

  _backend = (torch, TorchDISBatchedPropFast)
  return _backend


def _device_batch(total: float, pair_memory: float) -> int:
  """Per-stream batch for a ``total``-byte card at ``pair_memory`` per pair, capped."""
  batch = int((total - _RESERVE) / pair_memory)
  return max(0, min(_MAX_BATCH, batch))


def unavailable_reason() -> str | None:
  """``None`` when torch imports and a GPU has room to run, else why it cannot."""
  try:
    torch = _import_backend()[0]
  except ImportError as exc:
    return f"{exc} (install the 'gpu' extra)"
  except Exception as exc:  # a broken CUDA install must not break the cv2 path
    return f"torch failed to import: {exc!r}"
  try:
    if not torch.cuda.is_available():
      return "torch reports no CUDA device"
    for i in range(torch.cuda.device_count()):
      free, total = torch.cuda.mem_get_info(i)
      if free >= _MIN_FREE and _device_batch(total, _PAIR_MEMORY) >= 1:
        return None
    return "no GPU has enough free memory"
  except Exception as exc:
    return f"CUDA probe failed: {exc!r}"


def _claim_device(pair_memory: float) -> tuple[int, int]:
  """``(gpu, batch)``, preferring unclaimed GPUs by free memory; raises if none fits."""
  torch = _import_backend()[0]
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")
  with _claim_lock:
    info = {
      i: torch.cuda.mem_get_info(i)
      for i in range(torch.cuda.device_count())
      if i not in _bad
    }
    free = {i: f for i, (f, _) in info.items()}
    batch = {i: _device_batch(t, pair_memory) for i, (_, t) in info.items()}
    shareable = [i for i in free if free[i] >= _MIN_FREE and batch[i] >= 1]
    if not shareable:
      raise RuntimeError("no GPU has enough free memory for the flow backend")
    unclaimed = [i for i in shareable if i not in _claimed]
    pool = unclaimed if unclaimed and len(_claimed) < _MAX_GPUS else shareable
    idx = max(pool, key=lambda i: free[i])
    _claimed[idx] = _claimed.get(idx, 0) + 1
    return idx, batch[idx]


def _drop_claim(idx: int) -> None:
  """Release this stream's claim on ``idx`` and mark the GPU bad."""
  with _claim_lock:
    _bad.add(idx)
    count = _claimed.get(idx, 0) - 1
    if count <= 0:
      _claimed.pop(idx, None)
    else:
      _claimed[idx] = count


class _Worker(threading.Thread):
  """One GPU stream that computes batches of pair flows off a private job queue."""

  def __init__(self, device_index: int, batch: int, fwd_vr: int, bwd_vr: int):
    super().__init__(daemon=True)
    self.device_index = device_index
    self.batch = batch
    self._fwd_vr = fwd_vr
    self._bwd_vr = bwd_vr
    self._jobs: queue.Queue[Any] = queue.Queue()
    self._fwd_dis: Any = None
    self._bwd_dis: Any = None
    self._pinned = _PINNED
    self._pin_prev: Any = None
    self._pin_cur: Any = None
    self._dev_prev: Any = None
    self._dev_cur: Any = None
    self._pin_down: Any = None
    self.start()

  def submit(self, job: Any) -> None:
    self._jobs.put(job)

  def run(self) -> None:
    torch = _import_backend()[0]
    torch.cuda.set_device(self.device_index)
    while True:
      job = self._jobs.get()
      if job is None:
        self._release()
        return
      frames, tile, result, error, done = job
      try:
        result.update(self._compute_tile(frames, tile))
      except Exception as exc:  # surfaced to the dispatcher's caller
        error.append(exc)
      finally:
        done.set()
      # The job pins flow fields and the lens's frames; an idle worker would hold them.
      del job, frames, tile, result, error, done

  def _alloc_buffers(self, h: int, w: int) -> None:
    """Allocate page-locked staging buffers, else fall back to the pageable path."""
    torch = _import_backend()[0]
    dev = f"cuda:{self.device_index}"
    b = self.batch
    try:
      self._pin_prev = torch.empty(b, 1, h, w, dtype=torch.uint8, pin_memory=True)
      self._pin_cur = torch.empty(b, 1, h, w, dtype=torch.uint8, pin_memory=True)
      self._pin_down = torch.empty(b, h, w, 2, dtype=torch.float32, pin_memory=True)
    except RuntimeError:
      self._pinned = False
      self._pin_prev = self._pin_cur = self._pin_down = None
      return
    self._dev_prev = torch.empty(b, 1, h, w, dtype=torch.uint8, device=dev)
    self._dev_cur = torch.empty(b, 1, h, w, dtype=torch.uint8, device=dev)

  def _ensure(self, h: int, w: int) -> None:
    if self._fwd_dis is not None:
      return
    torch, batched_cls = _import_backend()
    dev = f"cuda:{self.device_index}"
    torch.cuda.set_device(self.device_index)
    while True:
      try:
        fwd = batched_cls(vr_iter=self._fwd_vr, device=dev)
        bwd = batched_cls(vr_iter=self._bwd_vr, device=dev)
        # Warm the kernels and prove the full batch fits; an OOM here halves the batch.
        blank = torch.zeros(self.batch, 1, h, w, device=dev)
        fwd.calc(blank, blank)
        bwd.calc(blank, blank)
        torch.cuda.synchronize(self.device_index)
        self._fwd_dis, self._bwd_dis = fwd, bwd
        if self._pinned:
          self._alloc_buffers(h, w)
        return
      except torch.cuda.OutOfMemoryError:
        self._fwd_dis = self._bwd_dis = None
        torch.cuda.empty_cache()
        if self.batch <= 1:
          raise
        self.batch = max(1, self.batch // 2)

  def _compute_tile(
    self, frames: Sequence[np.ndarray], tile: list[int]
  ) -> dict[int, _FlowPair]:
    torch = _import_backend()[0]
    h, w = frames[tile[0]].shape[:2]
    self._ensure(h, w)
    n = len(tile)
    if not self._pinned:
      dev = f"cuda:{self.device_index}"
      prev = torch.from_numpy(np.stack([frames[k - 1] for k in tile])[:, None]).to(dev)
      cur = torch.from_numpy(np.stack([frames[k] for k in tile])[:, None]).to(dev)
      fwd = self._fwd_dis.calc(prev, cur).to("cpu", torch.float32).numpy()
      bwd = self._bwd_dis.calc(cur, prev).to("cpu", torch.float32).numpy()
      return {k: (fwd[j].copy(), bwd[j].copy()) for j, k in enumerate(tile)}

    prev_np = self._pin_prev.numpy()
    cur_np = self._pin_cur.numpy()
    for j, k in enumerate(tile):
      prev_np[j, 0] = frames[k - 1]
      cur_np[j, 0] = frames[k]
    # Upload once (prev, cur); the forward and backward passes reuse both.
    self._dev_prev[:n].copy_(self._pin_prev[:n], non_blocking=True)
    self._dev_cur[:n].copy_(self._pin_cur[:n], non_blocking=True)
    down = self._pin_down.numpy()  # view over the reused download buffer
    fwd_out = self._fwd_dis.calc(self._dev_prev[:n], self._dev_cur[:n])
    self._pin_down[:n].copy_(fwd_out, non_blocking=True)
    torch.cuda.synchronize(self.device_index)
    fwd_fields = [down[j].copy() for j in range(n)]
    bwd_out = self._bwd_dis.calc(self._dev_cur[:n], self._dev_prev[:n])
    self._pin_down[:n].copy_(bwd_out, non_blocking=True)
    torch.cuda.synchronize(self.device_index)
    return {k: (fwd_fields[j], down[j].copy()) for j, k in enumerate(tile)}

  def _release(self) -> None:
    torch = _import_backend()[0]
    self._fwd_dis = self._bwd_dis = None
    self._pin_prev = self._pin_cur = self._pin_down = None
    self._dev_prev = self._dev_cur = None
    with suppress(Exception):
      torch.cuda.empty_cache()


class _Engine:
  """One lens stream bound to one GPU through a single worker."""

  def __init__(self, fwd_vr: int, bwd_vr: int, pair_memory: float):
    self.device_index, batch = _claim_device(pair_memory)
    self._worker = _Worker(self.device_index, batch, fwd_vr, bwd_vr)

  @property
  def batch(self) -> int:
    return self._worker.batch

  def prime(self, frames: Sequence[np.ndarray]) -> None:
    """Warm the worker on ``frames``' shape; raises RuntimeError if the GPU fails."""
    error: list[Exception] = []
    done = threading.Event()
    self._worker.submit((frames, [1], {}, error, done))
    done.wait()
    if error:
      msg = f"GPU {self.device_index} failed to warm the flow backend: {error[0]!r}"
      raise RuntimeError(msg) from error[0]

  def iter_flows(self, frames: Sequence[np.ndarray], ks: range) -> Iterator[_FlowPair]:
    """``(forward, backward)`` flow per pair ``(k-1, k)``, ``k`` ascending over ``ks``."""
    pair_ks = list(ks)
    worker = self._worker
    windows = [
      pair_ks[s : s + worker.batch] for s in range(0, len(pair_ks), worker.batch)
    ]

    def submit(
      chunk: list[int],
    ) -> tuple[list[int], dict[int, _FlowPair], list[Exception], threading.Event]:
      result: dict[int, _FlowPair] = {}
      error: list[Exception] = []
      done = threading.Event()
      worker.submit((frames, chunk, result, error, done))
      return chunk, result, error, done

    # One window stays in flight: the worker computes the next while the caller reads.
    inflight = submit(windows[0]) if windows else None
    for i in range(len(windows)):
      assert inflight is not None  # windows[i] went out one step ahead
      chunk, result, error, done = inflight
      inflight = submit(windows[i + 1]) if i + 1 < len(windows) else None
      done.wait()
      if error:
        raise error[0]
      for k in chunk:  # ascending, matching the cv2 stream order
        yield result[k]

  def shutdown(self) -> None:
    self._worker.submit(None)
    self._worker.join(timeout=30.0)


def prime(frames: Sequence[np.ndarray], fwd_vr: int, bwd_vr: int) -> _Engine:
  """A warm engine for ``frames``' shape, trying the next-most-free GPU on failure."""
  h, w = frames[0].shape[:2]
  key = ((h, w), fwd_vr, bwd_vr)
  warm: _Engine | None = getattr(_stream, "engine", None)
  if warm is not None and getattr(_stream, "key", None) == key and warm in _engines:
    return warm
  torch = _import_backend()[0]
  tries = torch.cuda.device_count() if torch.cuda.is_available() else 1
  last_exc: Exception | None = None
  pair_memory = _PAIR_MEMORY * (h * w) / _REFERENCE_PIXELS
  for _ in range(tries):
    engine = _Engine(fwd_vr, bwd_vr, pair_memory)  # raises when no GPU is claimable
    try:
      engine.prime(frames)
    except Exception as exc:
      last_exc = exc
      _drop_claim(engine.device_index)
      engine.shutdown()
      continue
    with _claim_lock:
      _engines.append(engine)
    _stream.engine, _stream.key = engine, key
    return engine
  raise last_exc or RuntimeError("no GPU could warm the flow backend")


def shutdown() -> None:
  """Release every stream's worker and GPU memory, and clear the GPU claims."""
  global _engines, _claimed, _bad
  with _claim_lock:
    engines = _engines
    _engines = []
    _claimed = {}
    _bad = set()
  for engine in engines:
    engine.shutdown()
