"""Convert one session's Motive export into the mocap ground-truth layer.

Usage:
    uv run python scripts/convert_mocap.py take.csv dataset/base/<id>.rrd dataset \
        --estimate dataset/estimate/<id>.rrd
"""

import argparse
import csv
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rerun as rr
from rerun.chunk import RrdReader
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation as Rot
from specialist_dataset import schema
from specialist_dataset.read import KeyframePoses, read_keyframe_poses

from specialist.record.session import SessionRecording

Float64Array = npt.NDArray[np.float64]

MOCAP_TIMELINE = "mocap_time"

# Below these a body's correlation or peak ratio leaves it unidentified in the take.
_SYNC_MIN_CORR = 0.5
_SYNC_MIN_PEAK_RATIO = 1.2
# How far a body's correlation must lead the runner-up camera's.
_MIN_CAMERA_MARGIN = 0.15
# Peaks nearer than this to the winner are the same peak, not a rival.
_SYNC_PEAK_EXCLUSION = 1.0

# Outlier limit: error floor plus this many MADs, the MAD at least tolerance x floor.
_ERROR_OUTLIER_MADS = 8.0
_ERROR_FLOOR_TOLERANCE = 0.05

# Above this a gyro carries vibration the optical system never sees.
_PROFILE_CUTOFF = 5.0
# Thinning the gyro costs nothing: it runs far above the profile's bandwidth.
_PROFILE_DECIMATION = 10
# Refine until the clock is quantized well under a mocap frame.
_OFFSET_RESOLUTION = 5e-5


@dataclass
class MotiveBody:
  """One rigid body's times, positions and XYZW quaternions; untracked rows hold NaN."""

  name: str
  times: Float64Array
  positions: Float64Array
  quaternions: Float64Array


def _metadata_from_row(row: list[str]) -> dict[str, str]:
  return {row[i]: row[i + 1] for i in range(0, len(row) - 1, 2)}


def _cell(value: str) -> float:
  return float(value) if value.strip() else float("nan")


def read_motive_csv(path: Path) -> dict[str, MotiveBody]:
  """Parse a Motive CSV's rigid bodies, NaN-masking frames its marker error flags."""
  with path.open(newline="") as f:
    rows = list(csv.reader(f))
  metadata = _metadata_from_row(rows[0])
  rotation_type = metadata.get("Rotation Type", "")
  if rotation_type != "Quaternion":
    raise ValueError(f"unsupported Rotation Type {rotation_type!r} in {path}")
  units = metadata.get("Length Units", "")
  if units != "Meters":
    raise ValueError(f"unsupported Length Units {units!r} in {path}")

  header_rows = {r[1]: r for r in rows[1:] if len(r) > 1 and r[0] == ""}
  axis_index = next(i for i, r in enumerate(rows) if r and r[0] == "Frame")
  kind_row, axis_row = rows[axis_index - 1], rows[axis_index]
  type_row, name_row = header_rows["Type"], header_rows["Name"]
  data = rows[axis_index + 1 :]

  bodies: dict[str, MotiveBody] = {}
  times = np.array([_cell(r[1]) for r in data])
  width = min(len(name_row), len(type_row), len(kind_row), len(axis_row))
  body_columns = [
    i for i in range(2, width) if type_row[i] == "Rigid Body" and name_row[i]
  ]
  for name in sorted({name_row[i] for i in body_columns}):
    columns = {
      (kind_row[i], axis_row[i]): i for i in body_columns if name_row[i] == name
    }
    try:
      position_cols = [columns["Position", a] for a in "XYZ"]
      quaternion_cols = [columns["Rotation", a] for a in "XYZW"]
    except KeyError as e:
      raise ValueError(f"body {name!r} in {path} lacks column {e}") from e
    positions = np.array([[_cell(r[i]) for i in position_cols] for r in data])
    quaternions = np.array([[_cell(r[i]) for i in quaternion_cols] for r in data])
    error_col = columns.get(("Position", "Error"))
    if error_col is None:
      error_col = columns.get(("Mean Marker Error", ""))
    if error_col is None:
      print(f"mocap body {name}: no marker-error column; dropouts pass unmasked")
    else:
      # A zero mean marker error is a held pose, not a measurement.
      error = np.array([_cell(r[error_col]) for r in data])
      untracked = ~np.isfinite(error) | (error <= 0.0)
      measured = error[~untracked]
      floor = float(np.median(measured)) if measured.size else float("nan")
      spread = (
        float(np.median(np.abs(measured - floor))) if measured.size else float("nan")
      )
      limit = floor + _ERROR_OUTLIER_MADS * max(spread, _ERROR_FLOOR_TOLERANCE * floor)
      outlier = ~untracked & (error > limit)
      print(
        f"mocap body {name}: error floor {1e3 * floor:.3f}mm, masking "
        f"{int(untracked.sum())} untracked and {int(outlier.sum())} above "
        f"{1e3 * limit:.3f}mm, of {len(error)} frames"
      )
      untracked |= outlier
      positions[untracked] = np.nan
      quaternions[untracked] = np.nan
    bodies[name] = MotiveBody(name, times, positions, quaternions)
  if not bodies:
    raise ValueError(f"no rigid bodies found in {path}")
  return bodies


@dataclass
class SyncResult:
  """The time map ``mocap = capture * rate + offset`` and its correlation evidence."""

  corr: float
  offset: float
  rate: float
  peak_corr: float
  second_corr: float
  # How far this camera's correlation led the runner-up camera's.
  margin: float = float("inf")

  @property
  def peak_ratio(self) -> float:
    return self.peak_corr / max(self.second_corr, 1e-9)

  @property
  def ok(self) -> bool:
    return (
      self.corr >= _SYNC_MIN_CORR
      and self.peak_ratio >= _SYNC_MIN_PEAK_RATIO
      and self.margin >= _MIN_CAMERA_MARGIN
    )


@dataclass
class Profile:
  """One instrument's motion as a scalar series: rotation rate or speed."""

  name: str
  times: Float64Array
  values: Float64Array


def _lowpass(times: Float64Array, values: Float64Array, cutoff: float) -> Float64Array:
  """Zero-phase lowpass at the sampling rate the times imply."""
  rate = 1.0 / float(np.median(np.diff(times)))
  sos = butter(2, min(cutoff / (0.5 * rate), 0.99), output="sos")
  filtered: Float64Array = sosfiltfilt(sos, values)
  return filtered


def gyro_profile(base: Path, slot: str) -> Profile | None:
  """The low-passed gyro rate magnitude, blind to IMU mounting; None if absent."""
  times, values = [], []
  content = f"/{schema.imu_signal(slot, 'gyro')}"
  stream = RrdReader(str(base)).store().stream().filter(content=content)
  for chunk in stream.to_chunks():
    batch = chunk.to_record_batch()
    names = batch.schema.names
    column = next((n for n in names if "Scalars" in n), None)
    if column is None or schema.TIMELINE not in names:
      continue
    times.append(
      np.array(
        [
          v.value / schema.NS_PER_SECOND
          for v in batch.column(names.index(schema.TIMELINE))
        ]
      )
    )
    values.append(
      np.array(
        [
          np.asarray(v, dtype=np.float64).reshape(-1)
          for v in batch.column(names.index(column)).to_pylist()
        ]
      )
    )
  if not times:
    return None
  t = np.concatenate(times)
  order = np.argsort(t, kind="stable")
  t = t[order]
  magnitude = np.linalg.norm(np.concatenate(values)[order], axis=1)
  keep = slice(None, None, _PROFILE_DECIMATION)
  return Profile(f"{slot} gyro", t[keep], _lowpass(t, magnitude, _PROFILE_CUTOFF)[keep])


def body_omega_profile(body: MotiveBody) -> Profile | None:
  """The mocap body's rotation rate, differenced frame to frame."""
  measured = np.isfinite(body.quaternions).all(axis=1)
  times, quaternions = body.times[measured], body.quaternions[measured]
  if len(times) < 16:
    return None
  rotations = Rot.from_quat(quaternions)
  swept = (rotations[:-1].inv() * rotations[1:]).magnitude()
  dt = np.diff(times)
  midtimes = 0.5 * (times[1:] + times[:-1])
  return Profile(
    f"{body.name} rotation", midtimes, _lowpass(midtimes, swept / dt, _PROFILE_CUTOFF)
  )


def solved_speed_profile(
  times: Float64Array, positions: Float64Array
) -> Profile | None:
  """The solved trajectory's speed between consecutive samples; None under four."""
  if len(times) < 4:
    return None
  speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
  return Profile("solved speed", 0.5 * (times[1:] + times[:-1]), speed)


def body_speed_profile(body: MotiveBody, window: float) -> Profile | None:
  """The body's speed over ``window`` seconds of tracked frames; None when too few."""
  measured = np.isfinite(body.positions).all(axis=1)
  times, positions = body.times[measured], body.positions[measured]
  if len(times) < 16:
    return None
  k = max(1, round(window / float(np.median(np.diff(times)))))
  if k >= len(times):
    return None
  dt = times[k:] - times[:-k]
  speed = np.linalg.norm(positions[k:] - positions[:-k], axis=1) / dt
  return Profile(f"{body.name} speed", 0.5 * (times[k:] + times[:-k]), speed)


def correlator(capture: Profile, mocap: Profile) -> Callable[[float, float], float]:
  """How well the two profiles line up under ``mocap = capture * rate + offset``."""

  def correlation(offset: float, rate: float) -> float:
    resampled = np.interp(
      capture.times * rate + offset, mocap.times, mocap.values, left=0, right=0
    )
    if np.std(resampled) < 1e-9:
      return -1.0
    return float(np.corrcoef(capture.values, resampled)[0, 1])

  return correlation


def scan_offset(capture: Profile, mocap: Profile) -> SyncResult:
  """Coarse-to-fine offset search at unit rate, also scoring the rival coarse peak."""
  correlation = correlator(capture, mocap)
  coarse_offsets = np.arange(
    mocap.times[0] - capture.times[-1], mocap.times[-1] - capture.times[0], 0.05
  )
  if len(coarse_offsets) < 2:
    return SyncResult(-1.0, 0.0, 1.0, 0.0, 1.0)
  coarse = np.array([correlation(float(o), 1.0) for o in coarse_offsets])
  peak = int(np.argmax(coarse))
  away = np.abs(coarse_offsets - coarse_offsets[peak]) > _SYNC_PEAK_EXCLUSION
  second = float(coarse[away].max()) if away.any() else -1.0
  best = (float(coarse[peak]), float(coarse_offsets[peak]), 1.0)
  span, step = 0.05, 0.005
  while step > _OFFSET_RESOLUTION:
    _, offset, rate = best
    grid = [
      (correlation(offset + do, rate), offset + do, rate)
      for do in np.arange(-span, span + step, step)
    ]
    best = max([best, *grid], key=lambda x: x[0])
    span, step = span / 4, step / 4
  return SyncResult(best[0], best[1], best[2], float(coarse[peak]), second)


def session_clock(paired: dict[str, tuple[str, SyncResult]]) -> SyncResult:
  """Accepted bodies' correlation-weighted mean offset at unit rate; exits if none."""
  accepted = [sync for _, sync in paired.values() if sync.ok]
  if not accepted:
    raise SystemExit(
      "no rigid body identified a capture; the take's clock cannot be recovered"
    )
  weight = sum(sync.corr for sync in accepted)
  return SyncResult(
    corr=weight / len(accepted),
    offset=sum(sync.corr * sync.offset for sync in accepted) / weight,
    rate=1.0,
    peak_corr=max(sync.peak_corr for sync in accepted),
    second_corr=min(sync.second_corr for sync in accepted),
  )


def session_times(times: Float64Array, sync: SyncResult) -> Float64Array:
  """``times`` on the session clock, inverting ``mocap = session * rate + offset``."""
  out: Float64Array = (times - sync.offset) / sync.rate
  return out


def pair_bodies(
  captures: dict[str, Profile], mocaps: dict[str, Profile]
) -> dict[str, tuple[str, SyncResult]]:
  """Each body's best-correlated capture; exits when two accepted bodies claim one."""
  paired: dict[str, tuple[str, SyncResult]] = {}
  for name, mocap in sorted(mocaps.items()):
    scored = [
      (slot, scan_offset(capture, mocap)) for slot, capture in sorted(captures.items())
    ]
    slot, sync = max(scored, key=lambda pair: pair[1].corr)
    runners = ", ".join(f"{s}={r.corr:.3f}" for s, r in scored if s != slot)
    rival = max((r.corr for s, r in scored if s != slot), default=0.0)
    margin = sync.corr - rival
    sync = replace(sync, margin=margin)
    verdict = (
      "ok" if sync.ok else ("TOO CLOSE" if margin < _MIN_CAMERA_MARGIN else "AMBIGUOUS")
    )
    print(
      f"  {name} -> {slot}: corr={sync.corr:.3f} offset={sync.offset:.4f}s "
      f"peak_ratio={sync.peak_ratio:.2f} margin={margin:.3f} verdict={verdict}"
      + (f" (others {runners})" if runners else "")
    )
    paired[name] = (slot, sync)
  claimed = [slot for slot, sync in paired.values() if sync.ok]
  if len(set(claimed)) != len(claimed):
    raise SystemExit(
      "two bodies identified the same capture, so one arm's ground truth would "
      "be written under the other's name; the take's rigid-body naming or its "
      "geometry is not what this session expects"
    )
  return paired


def choose_signal(
  base: Path,
  bodies: dict[str, MotiveBody],
  run: KeyframePoses | None,
) -> tuple[str, dict[str, tuple[str, SyncResult]]]:
  """Pair on rotation rate unless it fails, then on speed, which needs ``run``."""
  print("pairing on rotation rate (gyro):")
  captures = {s: p for s in _slots(base) if (p := gyro_profile(base, s))}
  omegas = {n: p for n, b in bodies.items() if (p := body_omega_profile(b))}
  rotation = pair_bodies(captures, omegas) if captures and omegas else {}
  best = max((s.corr for _, s in rotation.values()), default=-1.0)
  if rotation and all(sync.ok for _, sync in rotation.values()):
    return "rotation", rotation
  if run is None:
    raise SystemExit(
      f"rotation rate correlates at only {best:.3f}; the take's orientation "
      f"cannot carry the sync, so pass --estimate to fall back to its positions"
    )
  print(
    f"rotation rate correlates at only {best:.3f}; the rigid-body definition "
    f"leaves this take's orientation unusable, falling back to speed:"
  )
  speeds = {
    t: p
    for t in run.slots
    if (
      p := solved_speed_profile(run.kf_times_by_slot[t], run.kf_translations_by_slot[t])
    )
  }
  window = float(np.median(np.diff(run.kf_times_by_slot[run.slots[0]])))
  body_speeds = {
    n: p for n, b in bodies.items() if (p := body_speed_profile(b, window))
  }
  return "speed", pair_bodies(speeds, body_speeds)


def _slots(base: Path) -> list[str]:
  """The camera slots the base recording carries."""
  return schema.slots_in(RrdReader(str(base)).store().schema().entity_paths())


def base_tag_ids(estimate_rrd: Path | None) -> dict[str, int]:
  """Each camera slot's base tag id from ``estimate_rrd``; empty when it is None."""
  if estimate_rrd is None:
    return {}
  store = RrdReader(str(estimate_rrd)).store()
  tags: dict[str, int] = {}
  for slot in schema.slots_in(store.schema().entity_paths()):
    content = f"/{schema.base_tag(slot)}"
    for chunk in store.stream().filter(content=content, is_static=True).to_chunks():
      batch = chunk.to_record_batch()
      if "tag_id" in batch.schema.names:
        value = batch.to_pydict()["tag_id"][0]
        tags[slot] = int(value[0] if isinstance(value, list) else value)
  return tags


def write_layer(
  out_path: Path,
  session_id: str,
  bodies: dict[str, MotiveBody],
  paired: dict[str, tuple[str, SyncResult]],
  session: SyncResult,
  signal: str,
  source: str,
  tags: dict[str, int],
) -> None:
  """Write accepted bodies on both clocks, each naming its camera slot and base tag."""
  partial = out_path.parent / (out_path.name + ".partial")
  partial.unlink(missing_ok=True)
  with rr.RecordingStream(
    schema.APPLICATION_ID, recording_id=session_id, send_properties=False
  ) as stream:
    stream.save(str(partial))
    for name in sorted(bodies):
      if name not in paired or not paired[name][1].ok:
        print(f"mocap body {name}: identified no capture; left out of the layer")
        continue
      body = bodies[name]
      slot, sync = paired[name]
      measured = np.isfinite(body.positions).all(axis=1)
      measured &= np.isfinite(body.quaternions).all(axis=1)
      stream.send_columns(
        f"mocap/{name}",
        indexes=[
          rr.TimeColumn(MOCAP_TIMELINE, duration=body.times[measured]),
          rr.TimeColumn(
            schema.TIMELINE, duration=session_times(body.times[measured], session)
          ),
        ],
        columns=rr.Transform3D.columns(
          translation=body.positions[measured], quaternion=body.quaternions[measured]
        ),
      )
      stream.send_property(
        f"mocap_{name}",
        rr.AnyValues(
          slot=slot,
          base_tag_id=tags.get(slot, -1),
          correlation=sync.corr,
          peak_ratio=sync.peak_ratio,
          offset=sync.offset,
          measured_frames=int(measured.sum()),
          total_frames=int(measured.size),
        ),
      )
    stream.send_property(
      "mocap",
      rr.AnyValues(
        source=source,
        signal=signal,
        offset=session.offset,
        rate=session.rate,
      ),
    )
  partial.replace(out_path)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("mocap_csv", type=Path, help="Motive Tracking-Data CSV export")
  ap.add_argument("base_rrd", type=Path, help="the session's base recording")
  ap.add_argument(
    "output_dir", type=Path, help="dataset root; the layer lands in mocap/"
  )
  ap.add_argument(
    "--estimate",
    type=Path,
    default=None,
    help="the session's estimate layer, needed only to fall back to positions "
    "when the take's orientation cannot carry the sync",
  )
  ap.add_argument("--overwrite", action="store_true", help="replace an existing layer")
  args = ap.parse_args()

  session_id = SessionRecording(args.base_rrd).session_id
  out_path = args.output_dir / "mocap" / f"{session_id}.rrd"
  if out_path.exists() and not args.overwrite:
    raise SystemExit(
      f"{out_path} exists; pass --overwrite to replace it (a registered mocap "
      f"layer must also be re-registered, or its byte-range index goes stale)"
    )
  bodies = read_motive_csv(args.mocap_csv)
  run = read_keyframe_poses(args.estimate) if args.estimate is not None else None
  signal, paired = choose_signal(args.base_rrd, bodies, run)
  if not paired:
    raise SystemExit("no rigid body could be matched to a capture")

  session = session_clock(paired)
  spread = max(
    abs(sync.offset - session.offset) for _, sync in paired.values() if sync.ok
  )
  print(
    f"session clock ({signal}): offset={session.offset:.5f}s "
    f"corr={session.corr:.3f} (accepted bodies spread {1e3 * spread:.1f} ms)"
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  write_layer(
    out_path,
    session_id,
    bodies,
    paired,
    session,
    signal,
    args.mocap_csv.name,
    base_tag_ids(args.estimate),
  )
  print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
