"""IMU characterization probe; bias instability needs a still capture of 30 minutes."""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import allantools
import numpy as np
import numpy.typing as npt
import rerun as rr
import tyro

if __package__ in (None, ""):  # `python scripts/x.py` puts scripts/ itself first
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._probe import finalize_probe, verdict
from specialist.loader.parse_insv import InsvData, load_insv
from specialist.vi_ba.imu import ImuSamples, stationary_mask

Float64Array = npt.NDArray[np.float64]


@dataclass
class ImuArgs:
  """IMU characterization of one capture."""

  input_insv: Path
  output_dir: Path = Path("logs")
  stationary_window: float = 1.0
  stationary_gyro_threshold: float = math.radians(0.5)
  allan_max_tau: float = 100.0


@dataclass
class StationarySegment:
  start: float
  end: float
  gyro_sigma: tuple[float, float, float]  # rad/s
  accel_sigma: tuple[float, float, float]
  n_samples: int


@dataclass
class AllanResult:
  axis: str
  taus: list[float]
  adev: list[float]
  noise_density: float  # rad/s/sqrt(Hz)
  bias_instability: float  # rad/s


@dataclass
class ImuSummary:
  source: str
  n_samples: int
  duration: float
  median_dt: float
  rate: float
  rate_jitter: float
  gap_count: int
  gap_max: float
  stationary: StationarySegment | None
  allan: list[AllanResult] = field(default_factory=list)
  verdict: str = "unknown"


def _build_stationary(
  ts: Float64Array,
  gyro: Float64Array,
  accel: Float64Array,
  *,
  window: float,
  threshold: float,
) -> StationarySegment | None:
  """Per-axis noise over the longest run of stationary windows, gyro in rad/s."""
  median_dt = float(np.median(np.diff(ts)))
  win = max(8, int(round(window / max(median_dt, 1e-6))))
  # Gate on the gyro alone; the accel sigma is reported, never thresholded.
  quiet = stationary_mask(
    ImuSamples(t=ts, gyro=gyro, accel=accel),
    window=win,
    gyro_threshold=threshold,
    accel_var_threshold=float("inf"),
  )
  if not quiet.any():
    return None
  # The False padding turns a run touching either end into a normal edge pair.
  padded = np.concatenate(([False], quiet, [False])).astype(np.int8)
  edges = np.flatnonzero(np.diff(padded))
  starts, ends = edges[::2], edges[1::2]
  longest = int(np.argmax(ends - starts))
  if int(ends[longest] - starts[longest]) < 4:
    return None
  i0, i1 = int(starts[longest]) * win, int(ends[longest]) * win
  g_seg = gyro[i0:i1]
  a_seg = accel[i0:i1]
  return StationarySegment(
    start=float(ts[i0]),
    end=float(ts[i1 - 1]),
    gyro_sigma=tuple(map(float, g_seg.std(axis=0))),  # type: ignore[arg-type]
    accel_sigma=tuple(float(s) for s in a_seg.std(axis=0)),  # type: ignore[arg-type]
    n_samples=int(i1 - i0),
  )


def _allan_for_axis(
  signal: Float64Array, rate: float, axis: str, max_tau: float
) -> AllanResult:
  taus_target = np.logspace(np.log10(2.0 / rate), np.log10(max_tau), num=50)
  taus, adev, _, _ = allantools.oadev(
    signal, rate=rate, data_type="freq", taus=taus_target
  )
  taus = np.asarray(taus, dtype=np.float64)
  adev = np.asarray(adev, dtype=np.float64)

  # At slope -1/2, ADEV(tau) = N / sqrt(tau), so N is in rad/s/sqrt(Hz).
  short_mask = taus < 1.0
  noise_density = (
    float(np.median(adev[short_mask] * np.sqrt(taus[short_mask])))
    if short_mask.any()
    else float("nan")
  )

  long_mask = taus > 1.0
  bias_instability = float(np.min(adev[long_mask])) if long_mask.any() else float("nan")

  return AllanResult(
    axis=axis,
    taus=taus.tolist(),
    adev=adev.tolist(),
    noise_density=noise_density,
    bias_instability=bias_instability,
  )


def _log_to_rerun(
  data: InsvData, gyro: Float64Array, allans: list[AllanResult]
) -> None:
  times = rr.TimeColumn("imu_time", duration=data.imu_timestamps)
  triples = np.full(data.imu_timestamps.size, 3)
  for entity, samples in (("gyro", gyro), ("accel", data.accel)):
    rr.log(f"imu/{entity}", rr.SeriesLines(names=["x", "y", "z"]), static=True)
    rr.send_columns(
      f"imu/{entity}",
      indexes=[times],
      columns=rr.Scalars.columns(scalars=samples.reshape(-1)).partition(triples),
    )
  for allan in allans:
    rr.send_columns(
      f"imu/allan/{allan.axis}",
      indexes=[rr.TimeColumn("allan_tau", duration=allan.taus)],
      columns=rr.Scalars.columns(scalars=allan.adev),
    )


def _verdict(summary: ImuSummary) -> str:
  problems: list[str] = []
  if summary.gap_max > 0.01:
    problems.append(f"gap up to {summary.gap_max * 1e3:.1f} ms")
  if summary.rate_jitter > 0.05:
    problems.append(f"rate jitter {summary.rate_jitter:.1%}")
  return verdict(problems)


def main(args: ImuArgs) -> None:
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  data = load_insv(args.input_insv)

  ts = data.imu_timestamps
  gyro = data.gyro
  dts = np.diff(ts)
  median_dt = float(np.median(dts))
  rate = 1.0 / median_dt if median_dt > 0 else float("nan")
  rate_jitter = float(np.std(dts) / median_dt) if median_dt > 0 else float("nan")
  gap_count = int((dts > 5.0 * median_dt).sum())
  gap_max = float(dts.max()) if dts.size else 0.0

  stationary = _build_stationary(
    ts,
    gyro,
    data.accel,
    window=args.stationary_window,
    threshold=args.stationary_gyro_threshold,
  )

  allans: list[AllanResult] = []
  if stationary is not None and stationary.n_samples > int(rate * 4):
    i0 = int(np.searchsorted(ts, stationary.start))
    gyro_seg = gyro[i0 : i0 + stationary.n_samples]
    max_tau = min(args.allan_max_tau, stationary.n_samples / rate / 4.0)
    for ax_idx, axis_name in enumerate(("gyro_x", "gyro_y", "gyro_z")):
      allans.append(_allan_for_axis(gyro_seg[:, ax_idx], rate, axis_name, max_tau))

  summary = ImuSummary(
    source=str(args.input_insv),
    n_samples=int(ts.size),
    duration=float(ts[-1] - ts[0]),
    median_dt=median_dt,
    rate=rate,
    rate_jitter=rate_jitter,
    gap_count=gap_count,
    gap_max=gap_max,
    stationary=stationary,
    allan=allans,
  )
  summary.verdict = _verdict(summary)

  finalize_probe(
    "imu", args.output_dir, summary, lambda: _log_to_rerun(data, gyro, allans)
  )


if __name__ == "__main__":
  main(tyro.cli(ImuArgs))
