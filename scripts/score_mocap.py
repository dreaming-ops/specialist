"""Score a solve against the session's mocap ground-truth layer.

Usage:
    uv run python scripts/score_mocap.py dataset/estimate/<id>.rrd \
        dataset/mocap/<id>.rrd out.npz --tracker 0
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from rerun.chunk import RrdReader
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

if __package__ in (None, ""):  # `python scripts/x.py` puts scripts/ itself first
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from specialist_dataset import schema
from specialist_dataset.read import KeyframePoses, read_keyframe_poses

from scripts.convert_mocap import MotiveBody

Float64Array = npt.NDArray[np.float64]

# A gap wider than this is an occlusion; interpolating across one invents motion.
_MAX_INTERP_GAP = 0.1
# A camera's similarity fit onto the body it rode needs a scale inside this band.
_PAIRING_SCALE = (0.95, 1.05)


def read_mocap_layer(rrd_path: Path) -> dict[str, MotiveBody]:
  """The layer's bodies on the session clock, untracked frames absent; exits if none."""
  times: dict[str, list[Float64Array]] = {}
  positions: dict[str, list[Float64Array]] = {}
  quaternions: dict[str, list[Float64Array]] = {}
  stream = RrdReader(str(rrd_path)).store().stream().filter(content="/mocap/**")
  for chunk in stream.to_chunks():
    name = chunk.entity_path.rsplit("/", 1)[-1]
    batch = chunk.to_record_batch()
    columns = batch.schema.names
    if schema.TIMELINE not in columns or "Transform3D:translation" not in columns:
      continue
    times.setdefault(name, []).append(
      np.array(
        [
          v.value / schema.NS_PER_SECOND
          for v in batch.column(columns.index(schema.TIMELINE))
        ],
        dtype=np.float64,
      )
    )
    positions.setdefault(name, []).append(
      np.array(
        [
          np.asarray(v, dtype=np.float64).reshape(3)
          for v in batch.column(columns.index("Transform3D:translation")).to_pylist()
        ]
      )
    )
    quaternions.setdefault(name, []).append(
      np.array(
        [
          np.asarray(v, dtype=np.float64).reshape(4)
          for v in batch.column(columns.index("Transform3D:quaternion")).to_pylist()
        ]
      )
    )
  bodies = {}
  for name in times:
    order = np.argsort(np.concatenate(times[name]), kind="stable")
    bodies[name] = MotiveBody(
      name,
      np.concatenate(times[name])[order],
      np.concatenate(positions[name])[order],
      np.concatenate(quaternions[name])[order],
    )
  if not bodies:
    raise SystemExit(f"{rrd_path} carries no /mocap bodies; convert it first")
  return bodies


def _umeyama_weighted(
  src: np.ndarray, dst: np.ndarray, weights: np.ndarray, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
  """Weighted Umeyama scale, rotation and translation mapping ``src`` onto ``dst``."""
  weights = weights / weights.sum()
  mean_src = (src * weights[:, None]).sum(0)
  mean_dst = (dst * weights[:, None]).sum(0)
  src_centered = src - mean_src
  dst_centered = dst - mean_dst
  u, singular_values, vt = np.linalg.svd(
    (dst_centered * weights[:, None]).T @ src_centered
  )
  reflection_fix = np.eye(3)
  if np.linalg.det(u @ vt) < 0:
    reflection_fix[-1, -1] = -1
  rotation = u @ reflection_fix @ vt
  scale = 1.0
  if with_scale:
    scale = float(
      np.trace(np.diag(singular_values) @ reflection_fix)
      / (weights[:, None] * src_centered**2).sum()
    )
  return scale, rotation, mean_dst - scale * rotation @ mean_src


def _robust_fit(
  src: np.ndarray, dst: np.ndarray, iters: int = 10, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
  """Huber-IRLS alignment, sim(3) or rigid SE(3)."""
  weights = np.ones(len(src))
  scale, rotation, translation = _umeyama_weighted(src, dst, weights, with_scale)
  for _ in range(iters):
    residual = np.linalg.norm(
      (scale * (rotation @ src.T).T + translation) - dst, axis=1
    )
    huber_k = 1.345 * (float(np.median(residual)) + 1e-9)
    weights = np.where(residual <= huber_k, 1.0, huber_k / (residual + 1e-9))
    scale, rotation, translation = _umeyama_weighted(src, dst, weights, with_scale)
  return scale, rotation, translation, weights


def _resample(
  times: Float64Array, values: Float64Array, query: Float64Array
) -> tuple[Float64Array, npt.NDArray[np.bool_]]:
  """Rows linearly resampled at ``query``, valid only inside coverage and short gaps."""
  finite = np.all(np.isfinite(values), axis=1)
  valid_times = times[finite]
  resampled = np.column_stack(
    [np.interp(query, valid_times, values[finite, j]) for j in range(values.shape[1])]
  )
  inside = (query >= valid_times[0]) & (query <= valid_times[-1])
  right = np.clip(np.searchsorted(valid_times, query), 1, len(valid_times) - 1)
  gap = valid_times[right] - valid_times[right - 1]
  return resampled, inside & (gap <= _MAX_INTERP_GAP)


def solve_lever_arm(
  kf_positions: Float64Array,
  kf_rotations: Float64Array,
  mocap_positions: Float64Array,
  rotation: Float64Array,
  weights: Float64Array,
) -> Float64Array:
  """The body-frame lever arm, fitting scale and translation with ``rotation`` fixed."""
  rotated_positions = (rotation @ kf_positions.T).T
  rotated_bodies = np.einsum("ij,njk->nik", rotation, kf_rotations)
  design = np.zeros((len(kf_positions), 3, 7))
  design[:, :, 0] = rotated_positions
  design[:, :, 1:4] = rotated_bodies
  design[:, :, 4:7] = np.eye(3)
  w = np.sqrt(weights)[:, None, None]
  solution, _, _, _ = np.linalg.lstsq(
    (design * w).reshape(-1, 7),
    (mocap_positions * w[..., 0]).reshape(-1),
    rcond=None,
  )
  return np.asarray(solution[1:4] / solution[0], dtype=np.float64)


@dataclass
class Alignment:
  """A robust fit, rigid or similarity, and its residual statistics."""

  scale: float
  rotation: Float64Array
  translation: Float64Array
  weights: Float64Array
  ate_mm: Float64Array

  @property
  def inliers(self) -> int:
    return int((self.weights > 0.5).sum())


def _align(src: Float64Array, dst: Float64Array, with_scale: bool = True) -> Alignment:
  scale, rotation, translation, weights = _robust_fit(src, dst, with_scale=with_scale)
  residual_mm = (
    np.linalg.norm((scale * (rotation @ src.T).T + translation) - dst, axis=1) * 1000
  )
  return Alignment(scale, rotation, translation, weights, residual_mm)


def _ate_str(a: Alignment) -> str:
  p50, p90, p99 = np.percentile(a.ate_mm, [50, 90, 99])
  return f"ATE mm p50/p90/p99 {p50:.1f}/{p90:.1f}/{p99:.1f} inliers={a.inliers}/{len(a.weights)}"


def _score_cross_arm(
  run: KeyframePoses, bodies: dict[str, MotiveBody], body_names: list[str]
) -> None:
  """Print the inter-arm marker distance error, which needs no shared alignment."""
  markers = []
  for name, body_name in zip(run.slots, body_names, strict=True):
    body = bodies[body_name]
    kf_times = run.kf_times_by_slot[name]
    kf_positions = run.kf_translations_by_slot[name]
    kf_rotations = run.kf_rotations_by_slot[name]
    mocap_positions, covered = _resample(body.times, body.positions, kf_times)
    idx = np.where(covered)[0]
    if len(idx) == 0:
      print(
        f"cross-arm check skipped: no keyframe resolves inside {body_name}'s "
        f"tracked mocap coverage"
      )
      return
    fit = _align(kf_positions[idx], mocap_positions[idx], with_scale=False)
    lever = np.zeros(3)
    for _ in range(2):
      lever = solve_lever_arm(
        kf_positions[idx],
        kf_rotations[idx],
        mocap_positions[idx],
        fit.rotation,
        fit.weights,
      )
      fit = _align(
        kf_positions[idx] + kf_rotations[idx] @ lever,
        mocap_positions[idx],
        with_scale=False,
      )
    marker_run = kf_positions + kf_rotations @ lever
    print(
      f"cross-arm leg {name}/{body_name}: lever=[{1e3 * lever[0]:.1f} "
      f"{1e3 * lever[1]:.1f} {1e3 * lever[2]:.1f}]mm"
    )
    markers.append((kf_times, marker_run, body))

  (t0, m0, b0), (t1, m1, b1) = markers
  lo, hi = max(t0[0], t1[0]), min(t0[-1], t1[-1])
  grid = np.arange(lo, hi, 0.1)
  a0 = np.stack([np.interp(grid, t0, m0[:, i]) for i in range(3)], axis=1)
  a1 = np.stack([np.interp(grid, t1, m1[:, i]) for i in range(3)], axis=1)
  d_run = np.linalg.norm(a0 - a1, axis=1)
  g0, c0 = _resample(b0.times, b0.positions, grid)
  g1, c1 = _resample(b1.times, b1.positions, grid)
  ok = c0 & c1
  d_gt = np.linalg.norm(g0[ok] - g1[ok], axis=1)
  err = np.abs(d_run[ok] - d_gt) * 1000
  p50, p90, p99 = np.percentile(err, [50, 90, 99])
  print(
    f"cross-arm distance error mm p50/p90/p99 {p50:.1f}/{p90:.1f}/{p99:.1f} "
    f"(gt median {np.median(d_gt):.3f} m, {int(ok.sum())} samples)"
  )


def read_body_slots(rrd_path: Path, slots: list[str]) -> dict[str, str]:
  """The slot each body rode; refuses positional ``tracker`` names and unknown slots."""
  recorded: dict[str, str] = {}
  stale: list[str] = []
  stream = RrdReader(str(rrd_path)).store().stream().filter(content="/__properties/**")
  for chunk in stream.to_chunks():
    entity = chunk.entity_path.rsplit("/", 1)[-1]
    if not entity.startswith("mocap_"):
      continue
    batch = chunk.to_record_batch()
    names = batch.schema.names
    body = entity.removeprefix("mocap_")
    field = "slot" if "slot" in names else ("tracker" if "tracker" in names else None)
    if field is None:
      continue
    values = batch.column(names.index(field)).to_pylist()
    if not values or not values[-1]:
      continue
    value = values[-1]
    name = str(value[0] if isinstance(value, list) else value)
    if field == "slot":
      recorded[body] = name
    else:
      stale.append(f"{body}={name}")
  if stale:
    raise SystemExit(
      f"{rrd_path} pairs bodies to positional capture names ({', '.join(stale)}), "
      f"which do not identify a camera; re-run convert_mocap.py to record the "
      f"slot each body rode"
    )
  unknown = {s for s in recorded.values() if s not in slots}
  if unknown:
    raise SystemExit(
      f"{rrd_path} names cameras the estimate has no slot for: {unknown}"
    )
  return recorded


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("run_rrd", type=Path, help="the session's estimate layer")
  ap.add_argument("mocap_rrd", type=Path, help="convert_mocap.py mocap-layer rrd")
  ap.add_argument("out_npz", type=Path, help="overlay .npz (poses, times)")
  ap.add_argument(
    "--body",
    default=None,
    help="rigid-body name to score (default: the one the layer recorded)",
  )
  ap.add_argument(
    "--tracker",
    type=int,
    default=0,
    help="index into the run's camera slots to score (default: the first)",
  )
  ap.add_argument(
    "--cross-arm",
    action="store_true",
    help="score the alignment-free inter-arm distance error instead, against "
    "the bodies the layer records for each tracker",
  )
  ap.add_argument(
    "--lever-arm",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=None,
    help="measured body-frame lever arm in meters; bypasses the estimate",
  )
  args = ap.parse_args()

  run = read_keyframe_poses(args.run_rrd)
  bodies = read_mocap_layer(args.mocap_rrd)
  body_slots = read_body_slots(args.mocap_rrd, run.slots)
  by_slot = {s: b for b, s in body_slots.items()}
  if len(by_slot) != len(body_slots):
    raise SystemExit(
      f"two bodies claim one camera in {args.mocap_rrd}: {body_slots}. One arm "
      f"would be scored against the other's ground truth"
    )
  if args.cross_arm:
    missing = [slot for slot in run.slots if slot not in by_slot]
    if missing:
      raise SystemExit(f"the layer records no body for {', '.join(missing)}")
    _score_cross_arm(run, bodies, [by_slot[s] for s in run.slots])
    return
  name = run.slots[args.tracker]
  print(f"scoring camera {name} (index {args.tracker})")
  kf_times = run.kf_times_by_slot[name]
  kf_positions = run.kf_translations_by_slot[name]
  kf_rotations = run.kf_rotations_by_slot[name]

  if args.body is not None and args.body not in bodies:
    raise SystemExit(f"body {args.body!r} not in {sorted(bodies)}")
  recorded = by_slot.get(name)
  target = args.body or recorded
  if args.body is not None and recorded is not None and args.body != recorded:
    print(
      f"[warn] --body {args.body} contradicts the layer, which pairs {name} "
      f"with {recorded}; scoring what was asked for"
    )
  if target is None:
    raise SystemExit(
      f"the layer records no body for {name}; pass --body, or re-convert so "
      f"the pairing is stored"
    )
  body = bodies[target]
  print(f"ground truth body {target} ({len(body.times)} tracked frames)")

  mocap_positions, covered = _resample(body.times, body.positions, kf_times)
  idx = np.where(covered)[0]
  if len(idx) == 0:
    raise SystemExit(
      f"no keyframe resolves inside {target}'s tracked mocap coverage; nothing to score"
    )

  # A rigid alignment cannot hide a scale error inside the fit.
  baseline = _align(kf_positions[idx], mocap_positions[idx], with_scale=False)
  print(f"SE3 poses={len(idx)} {_ate_str(baseline)}")

  lever_arm = (
    None if args.lever_arm is None else np.asarray(args.lever_arm, dtype=np.float64)
  )
  refined = baseline
  if lever_arm is None:
    lever_arm = np.zeros(3)
    for _ in range(2):
      lever_arm = solve_lever_arm(
        kf_positions[idx],
        kf_rotations[idx],
        mocap_positions[idx],
        refined.rotation,
        refined.weights,
      )
      refined = _align(
        kf_positions[idx] + kf_rotations[idx] @ lever_arm,
        mocap_positions[idx],
        with_scale=False,
      )
  else:
    refined = _align(
      kf_positions[idx] + kf_rotations[idx] @ lever_arm,
      mocap_positions[idx],
      with_scale=False,
    )
  r_mm = lever_arm * 1000
  print(
    f"SE3 lever-arm r=[{r_mm[0]:.1f} {r_mm[1]:.1f} {r_mm[2]:.1f}]mm {_ate_str(refined)}"
  )

  # A similarity fit's scale shows which arm a camera rode, so it gates the run.
  pairing = _align(
    kf_positions[idx] + kf_rotations[idx] @ lever_arm, mocap_positions[idx]
  )
  if not _PAIRING_SCALE[0] <= pairing.scale <= _PAIRING_SCALE[1]:
    raise SystemExit(
      f"{name} is being scored against a body that did not ride it, or the two "
      f"clocks disagree: fitting it onto this body needs a {pairing.scale:.3f} "
      f"size change, against {_PAIRING_SCALE} for an arm that rode the camera. "
      f"The ATE above is not a measure of this run."
    )
  print(f"saved {args.out_npz} poses={len(idx)}")

  # Overlay poses in the run's world; rotations omit the marker-to-gripper rotation.
  slerp = Slerp(body.times, Rot.from_quat(body.quaternions))
  ghost_rot = slerp(np.clip(kf_times[idx], body.times[0], body.times[-1])).as_matrix()
  ghost_pos = (
    refined.rotation.T @ (mocap_positions[idx] - refined.translation).T
  ).T - kf_rotations[idx] @ lever_arm
  poses = np.empty((len(idx), 4, 4))
  for n in range(len(idx)):
    poses[n] = np.eye(4)
    poses[n, :3, :3] = refined.rotation.T @ ghost_rot[n]
    poses[n, :3, 3] = ghost_pos[n]
  np.savez(args.out_npz, poses=poses, times=kf_times[idx], ate_mm=refined.ate_mm)


if __name__ == "__main__":
  main()
