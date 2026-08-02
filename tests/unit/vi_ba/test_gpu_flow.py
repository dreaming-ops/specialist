"""``vi_ba.gpu_flow``'s stream bookkeeping, which imports and runs without CUDA."""

from __future__ import annotations

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


def test_a_different_frame_shape_does_not_reuse_the_warm_stream() -> None:
  """The staging buffers fit one frame shape."""
  _pretend_warm(object(), (64, 64), 10, 2)
  with pytest.raises(Exception):  # noqa: B017 - no GPU here; it must not reuse
    gpu_flow.prime(_frames(128, 128), 10, 2)


def test_a_stream_already_shut_down_is_not_handed_out_again() -> None:
  """A thread still holding a reference warms a new stream on its next call."""
  _pretend_warm(object(), (64, 64), 10, 2)
  gpu_flow._engines.clear()
  with pytest.raises(Exception):  # noqa: B017 - no GPU here; it must not reuse
    gpu_flow.prime(_frames(), 10, 2)
