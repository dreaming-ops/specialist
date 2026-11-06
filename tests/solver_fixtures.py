"""The solver's factorization on scipy, for tests without a GPU."""

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


class ScipyFactorization:
  xp, xsp = np, sp

  def __init__(self) -> None:
    self._lu: Any = None

  def factorize(self, S: sp.csr_matrix) -> None:
    self._lu = spl.splu(S.tocsc())

  def apply(self, rhs: Any) -> Any:
    return self._lu.solve(rhs)

  def to_host(self, array: Any) -> np.ndarray:
    return np.asarray(array)

  def close(self) -> None:
    self._lu = None
