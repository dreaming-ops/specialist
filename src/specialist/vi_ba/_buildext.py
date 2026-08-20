"""Compile ``cpp/mei_factor.cpp`` against the gtsam wheel into ``_mei_factor_cpp``."""

from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

GTSAM_TAG = "4.3a0"
BOOST_TAG = "1.83.0"


def _repo_root(start: Path) -> Path:
  for d in (start, *start.parents):
    if (d / "pyproject.toml").exists():
      return d
  raise SystemExit("could not locate repo root (pyproject.toml)")


def _run(cmd: list[str]) -> None:
  print("  $", " ".join(cmd))
  # A git hook's GIT_INDEX_FILE would aim the gtsam clone at the calling repo's index.
  env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
  subprocess.run(cmd, check=True, env=env)


def _gtsam_paths() -> tuple[Path, Path]:
  """Return ``(gtsam_package_dir, libgtsam_path)`` from the installed wheel."""
  import gtsam

  pkg = Path(gtsam.__file__).resolve().parent
  if sys.platform == "darwin":
    libs = sorted((pkg / ".dylibs").glob("libgtsam.*.dylib"))
  else:
    libs = sorted((pkg.parent / "gtsam.libs").glob("libgtsam*.so*")) or sorted(
      pkg.glob("libgtsam*.so*")
    )
  if not libs:
    raise SystemExit("could not locate libgtsam shared library next to the wheel")
  return pkg, libs[0]


def _ensure_headers(gtsam_src: Path) -> None:
  """Shallow-clone gtsam at the pinned tag for its C++ headers."""
  if (gtsam_src / "gtsam" / "geometry" / "Pose3.h").exists():
    return
  gtsam_src.parent.mkdir(parents=True, exist_ok=True)
  print(f"cloning gtsam {GTSAM_TAG} headers into {gtsam_src} ...")
  _run(
    [
      "git",
      "clone",
      "--depth",
      "1",
      "--branch",
      GTSAM_TAG,
      "--single-branch",
      "https://github.com/borglab/gtsam.git",
      str(gtsam_src),
    ]
  )


def _system_boost() -> Path | None:
  """The prefix whose ``boost/`` headers the compiler will see, if any."""
  if sys.platform == "darwin":
    brew = os.popen("brew --prefix 2>/dev/null").read().strip() or "/opt/homebrew"
    root = Path(brew) / "include"
  else:
    root = Path("/usr/include")
  return root if (root / "boost" / "concept_check.hpp").exists() else None


def _ensure_boost(boost_src: Path) -> None:
  """Fetch the pinned Boost release headers; ``boost/`` is all the build needs."""
  if (boost_src / "boost" / "concept_check.hpp").exists():
    return
  import io
  import tarfile
  import urllib.request

  plain = BOOST_TAG.replace(".", "_")
  url = f"https://archives.boost.io/release/{BOOST_TAG}/source/boost_{plain}.tar.gz"
  print(f"fetching boost {BOOST_TAG} headers into {boost_src} ...")
  boost_src.mkdir(parents=True, exist_ok=True)
  with urllib.request.urlopen(url) as resp:
    data = io.BytesIO(resp.read())
  prefix = f"boost_{plain}/boost/"
  with tarfile.open(fileobj=data, mode="r:gz") as tar:
    members = [m for m in tar.getmembers() if m.name.startswith(prefix)]
    for m in members:
      m.name = m.name[len(f"boost_{plain}/") :]
    tar.extractall(boost_src, members=members, filter="data")


def _detect_gxx_abi_version(gtsam_pkg: Path) -> str | None:
  """Return the ``-fabi-version`` the wheel's pybind11 module embeds, or ``None``."""
  pat = re.compile(rb"pybind11_internals_v\d+_[a-z_]*cxxabi(\d+)")
  for so in gtsam_pkg.glob("*.so"):
    m = pat.search(so.read_bytes())
    if m:
      # __GXX_ABI_VERSION == 1000 + fabi-version (e.g. cxxabi1014 -> 14).
      return str(int(m.group(1)) - 1000)
  print(
    "  [warn] could not read the pybind11 ABI version from the gtsam wheel; "
    "omitting -fabi-version (cross-module type sharing may fail)"
  )
  return None


def _detect_cxx11_abi(libgtsam: Path) -> str:
  """Return the wheel's ``_GLIBCXX_USE_CXX11_ABI`` value, ``"0"`` or ``"1"``."""
  try:
    out = subprocess.run(
      ["nm", "-D", str(libgtsam)], capture_output=True, text=True, check=True
    ).stdout
  except Exception:
    print(
      f"  [warn] nm unavailable; assuming _GLIBCXX_USE_CXX11_ABI=0 ({libgtsam.name})"
    )
    return "0"  # manylinux convention
  for line in out.splitlines():
    if "NoiseModelFactor5print" in line:
      return "1" if "__cxx11" in line else "0"
  print(
    f"  [warn] NoiseModelFactor symbol not found in {libgtsam.name}; "
    "assuming _GLIBCXX_USE_CXX11_ABI=0"
  )
  return "0"


def _includes(gtsam_src: Path, generated: Path, boost_root: Path) -> list[str]:
  import numpy
  import pybind11

  return [
    pybind11.get_include(),
    sysconfig.get_path("include"),
    numpy.get_include(),
    str(gtsam_src),
    str(generated),
    str(gtsam_src / "gtsam" / "3rdparty" / "Eigen"),
    str(gtsam_src / "gtsam" / "3rdparty" / "metis" / "include"),
    str(boost_root),
  ]


def build(force: bool = False) -> Path:
  """Compile the extension into the package; return its path."""
  import pybind11

  if sys.platform == "win32":
    raise SystemExit("Windows is not supported; build on macOS or Linux")
  if int(pybind11.__version__.split(".")[0]) != 2:
    raise SystemExit(
      f"pybind11 {pybind11.__version__} found; the gtsam wheel's ABI needs the 2.x "
      "series pinned in pyproject.toml"
    )
  # A wheel off the pinned headers compiles cleanly and breaks the ABI at runtime.
  gtsam_version = importlib.metadata.version("gtsam")
  if gtsam_version != GTSAM_TAG:
    raise SystemExit(
      f"installed gtsam {gtsam_version} != pinned headers {GTSAM_TAG}; bump "
      f"GTSAM_TAG and re-vendor cpp/gtsam_generated/ from the matching source"
    )
  viba_pkg = Path(__file__).resolve().parent
  repo = _repo_root(viba_pkg)
  src = viba_pkg / "cpp" / "mei_factor.cpp"
  generated = viba_pkg / "cpp" / "gtsam_generated"
  gtsam_src = repo / ".build" / f"gtsam-{GTSAM_TAG}"
  out = viba_pkg / ("_mei_factor_cpp" + str(sysconfig.get_config_var("EXT_SUFFIX")))
  newest_input = max(p.stat().st_mtime for p in (src, *generated.rglob("*.h")))
  if out.exists() and not force and out.stat().st_mtime >= newest_input:
    print(f"{out.name} is up to date")
    return out

  gtsam_pkg, libgtsam = _gtsam_paths()
  _ensure_headers(gtsam_src)
  boost_root = _system_boost()
  if boost_root is None:
    boost_root = repo / ".build" / f"boost-{BOOST_TAG}"
    _ensure_boost(boost_root)
  cxx = os.environ.get("CXX", "c++")
  cmd = [cxx, "-O3", "-shared", "-std=c++17", "-fPIC"]
  if sys.platform == "darwin":
    cmd += ["-stdlib=libc++", "-undefined", "dynamic_lookup"]
  else:
    cmd += [f"-D_GLIBCXX_USE_CXX11_ABI={_detect_cxx11_abi(libgtsam)}"]
    abi_version = _detect_gxx_abi_version(gtsam_pkg)
    if abi_version is not None:
      cmd += [f"-fabi-version={abi_version}"]
    if sysconfig.get_platform().endswith("x86_64"):
      # The wheel frees the factor's Eigen buffers, so both build for its baseline.
      cmd += ["-march=x86-64"]
  for inc in _includes(gtsam_src, generated, boost_root):
    cmd += ["-I", inc]
  cmd += [str(src), "-o", str(out), str(libgtsam)]
  # Linux finds libgtsam by rpath; macOS dyld reuses the one `import gtsam` loaded.
  if sys.platform != "darwin":
    cmd += [
      "-Wl,-rpath,$ORIGIN/../../gtsam.libs",
      "-Wl,-rpath,$ORIGIN/../../gtsam",
      "-Wl,--no-as-needed",
    ]
  print(f"compiling {src.name} -> {out.name}")
  _run(cmd)
  return out


def main() -> None:
  """The ``build-mei-factor`` script: force a rebuild, then import the extension."""
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  build(force=True)
  subprocess.run(
    [
      sys.executable,
      "-c",
      "import gtsam, specialist.vi_ba._mei_factor_cpp; print('import OK')",
    ],
    check=True,
  )


if __name__ == "__main__":
  main()
