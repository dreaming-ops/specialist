"""``vi_ba.gpu_flow``'s stream bookkeeping, which imports and runs without CUDA."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from specialist.vi_ba import gpu_flow


@pytest.fixture(autouse=True)
def _clean_streams() -> Iterator[None]:
  for state in (gpu_flow._claimed, gpu_flow._bad, gpu_flow._engines):
    state.clear()
  gpu_flow._stream.__dict__.clear()
  yield
  for state in (gpu_flow._claimed, gpu_flow._bad, gpu_flow._engines):
    state.clear()
  gpu_flow._stream.__dict__.clear()


def _frames(h: int = 64, w: int = 64) -> list[np.ndarray]:
  return [np.zeros((h, w), dtype=np.uint8) for _ in range(2)]


def _pretend_warm(engine: Any, shape: tuple[int, int], fwd: int, bwd: int) -> None:
  gpu_flow._engines.append(engine)
  gpu_flow._stream.engine = engine
  gpu_flow._stream.key = (shape, fwd, bwd)


def test_a_captures_second_lens_reuses_the_stream_its_first_lens_warmed() -> None:
  """Each stream holds gigabytes of page-locked host staging."""
  engine = object()
  _pretend_warm(engine, (64, 64), 10, 2)
  assert gpu_flow.prime(_frames(), 10, 2) is engine


def _refuse_to_build(monkeypatch: pytest.MonkeyPatch) -> list[int]:
  """Fail every backend import a fresh stream makes and return the attempts list."""
  attempts: list[int] = []

  def _no_backend() -> tuple[Any, Any]:
    attempts.append(1)
    raise RuntimeError("no backend in this test")

  monkeypatch.setattr(gpu_flow, "_import_backend", _no_backend)
  return attempts


def test_a_different_frame_shape_does_not_reuse_the_warm_stream(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The staging buffers fit one frame shape."""
  attempts = _refuse_to_build(monkeypatch)
  _pretend_warm(object(), (64, 64), 10, 2)
  with pytest.raises(RuntimeError, match="no backend"):
    gpu_flow.prime(_frames(128, 128), 10, 2)
  assert attempts, "the warm stream was reused for a different shape"


def test_a_stream_already_shut_down_is_not_handed_out_again(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A thread still holding a reference warms a new stream on its next call."""
  attempts = _refuse_to_build(monkeypatch)
  _pretend_warm(object(), (64, 64), 10, 2)
  gpu_flow._engines.clear()
  with pytest.raises(RuntimeError, match="no backend"):
    gpu_flow.prime(_frames(), 10, 2)
  assert attempts, "a stream that was shut down was handed out again"


def _fake_torch(free: float, total: float) -> Any:
  class _Cuda:
    @staticmethod
    def is_available() -> bool:
      return True

    @staticmethod
    def device_count() -> int:
      return 1

    @staticmethod
    def mem_get_info(i: int) -> tuple[int, int]:
      return int(free), int(total)

  class _Torch:
    cuda = _Cuda()

  return _Torch()


def test_claimed_batch_is_sized_from_total_memory(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A busy card still gets the batch its hardware affords."""
  fake = _fake_torch(free=9 * 2**30, total=24 * 2**30)
  monkeypatch.setattr(gpu_flow, "_import_backend", lambda: (fake, None))
  idx, batch = gpu_flow._claim_device(gpu_flow._PAIR_MEMORY)
  assert (idx, batch) == (
    0,
    int((24 * 2**30 - gpu_flow._RESERVE) / gpu_flow._PAIR_MEMORY),
  )


def test_claim_requires_the_minimum_free_memory(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A nearly-full card is not claimable even though its total could host a batch."""
  fake = _fake_torch(free=gpu_flow._MIN_FREE - 2**29, total=24 * 2**30)
  monkeypatch.setattr(gpu_flow, "_import_backend", lambda: (fake, None))
  with pytest.raises(RuntimeError, match="enough free memory"):
    gpu_flow._claim_device(gpu_flow._PAIR_MEMORY)


def test_device_batch_floors_at_zero_below_the_reserve() -> None:
  pair = gpu_flow._PAIR_MEMORY
  assert gpu_flow._device_batch(total=gpu_flow._RESERVE, pair_memory=pair) == 0
  assert gpu_flow._device_batch(total=0.0, pair_memory=pair) == 0


def test_device_batch_caps_at_the_maximum() -> None:
  batch = gpu_flow._device_batch(total=2**40, pair_memory=gpu_flow._PAIR_MEMORY)
  assert batch == gpu_flow._MAX_BATCH


def test_only_the_first_launch_at_each_batch_size_holds_the_warm_lock() -> None:
  class Dis:
    held: list[bool] = []

    def calc(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
      self.held.append(gpu_flow._warm_lock.locked())
      return a

  worker = gpu_flow._Worker.__new__(gpu_flow._Worker)
  worker._fwd_dis = worker._bwd_dis = Dis()
  worker._launched = {("fwd", 4)}
  for which, n in (("fwd", 4), ("fwd", 3), ("fwd", 3), ("bwd", 3)):
    worker._calc(which, np.zeros((n, 1)), np.zeros((n, 1)))
  assert Dis.held == [False, True, False, True]
