"""The lens imports numpy and nothing else."""

import subprocess
import sys

_FORBIDDEN = ("gtsam", "cv2", "pyarrow", "rerun", "scipy", "torch", "av")


def test_importing_the_lens_pulls_in_nothing_but_numpy() -> None:
  probe = (
    "import sys, specialist_camera;"
    f"print([m for m in {_FORBIDDEN!r} if m in sys.modules])"
  )
  found = subprocess.run(
    [sys.executable, "-c", probe], capture_output=True, text=True, check=True
  )
  assert found.stdout.strip() == "[]", found.stdout
