"""The parallax filter and track unprojection of ``vi_ba.tracks``."""

import numpy as np
from specialist_camera import MeiParams, mei_project, mei_unproject

from specialist.vi_ba.tracks import (
  filter_tracks_by_parallax,
  unproject_tracks,
)
from tests.mei_fixtures import wide_mei_params


def _params() -> MeiParams:
  return MeiParams(
    fx=400.0,
    fy=400.0,
    cx=480.0,
    cy=480.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
    xi=1.0,
  )


def _project_one(point_cam: np.ndarray, params: MeiParams) -> np.ndarray:
  return mei_project(point_cam.reshape(1, 3), params)[0]


def test_static_camera_track_kept_only_if_parallax_above_threshold() -> None:
  params = _params()
  body_mounted_pixel = _project_one(np.array([0.05, 0.0, 0.5]), params)
  static_track = {kf: body_mounted_pixel.copy() for kf in range(5)}

  moving_track = {
    kf: _project_one(np.array([0.05 + 0.05 * kf, 0.0, 0.5]), params) for kf in range(5)
  }

  tracks = {1: static_track, 2: moving_track}
  kept, rejected = filter_tracks_by_parallax(
    tracks, params, min_parallax=np.deg2rad(0.5)
  )
  assert 1 not in kept and 1 in rejected, "static-bearing track must be rejected"
  assert 2 in kept and 2 not in rejected, "wide-bearing-sweep track must be kept"


def test_unproject_tracks_batched_slices_match_per_track_scalar() -> None:
  params = _params()
  tracks = {
    7: {
      2: np.array([120.0, 90.0]),
      0: np.array([300.0, 310.0]),
      1: np.array([50.0, 470.0]),
    },
    3: {5: np.array([460.0, 12.0]), 4: np.array([470.0, 495.0])},
    9: {0: np.array([240.0, 240.0])},  # single obs -> below min_obs
  }
  out = {
    tid: (kfs, bearings)
    for tid, kfs, bearings in unproject_tracks(tracks, params, min_obs=2)
  }
  assert set(out) == {7, 3}
  for tid, (kfs, bearings) in out.items():
    assert kfs == sorted(tracks[tid])
    expected = mei_unproject(
      np.asarray([tracks[tid][k] for k in kfs], dtype=np.float64), params
    )
    assert np.array_equal(bearings, expected)


def test_split_tracks_to_window_cuts_long_tracks_into_new_landmarks() -> None:
  """Pieces take fresh ids at the parent's point, and a window of 0 changes nothing."""
  from specialist.vi_ba.tracks import split_tracks_to_window

  pt = np.array([1.0, 2.0, 3.0])
  tracks = {
    3: {k: np.array([k, 0.0]) for k in (0, 5, 40, 45, 90)},
    7: {k: np.array([k, 1.0]) for k in (2, 4)},
  }
  points = {3: pt, 7: pt + 1}
  same_points, same_tracks = split_tracks_to_window(points, tracks, 0)
  assert same_points is points and same_tracks is tracks

  out_points, out_tracks = split_tracks_to_window(points, tracks, 10)
  assert set(out_tracks) == {3, 7, 8, 9}
  assert sorted(out_tracks[3]) == [0, 5]
  assert sorted(out_tracks[8]) == [40, 45]
  assert sorted(out_tracks[9]) == [90]
  assert all(np.array_equal(out_points[t], pt) for t in (3, 8, 9))
  assert np.array_equal(out_points[7], pt + 1)


def test_parallax_filter_is_the_per_track_filter() -> None:
  """Over tracks of many lengths, both keep the same tracks in the same order."""
  rng = np.random.default_rng(29)
  params = wide_mei_params()
  tracks: dict[int, dict[int, np.ndarray]] = {}
  for tid in range(300):
    n = int(rng.choice([1, 2, 3, 5, 8, 13, 40]))
    base = rng.uniform(300.0, 1600.0, 2)
    spread = float(rng.choice([0.02, 0.5, 3.0, 40.0]))
    tracks[tid * 7] = {
      int(k): base + rng.normal(0.0, spread, 2)
      for k in rng.choice(200, n, replace=False)
    }
  cos_min = float(np.cos(0.004))
  want_kept: dict[int, dict[int, np.ndarray]] = {}
  want_rejected = {tid: obs for tid, obs in tracks.items() if len(obs) < 2}
  for tid, _, bearings in unproject_tracks(tracks, params, min_obs=2):
    dots = np.clip(bearings @ bearings.T, -1.0, 1.0)
    (want_kept if float(dots.min()) <= cos_min else want_rejected)[tid] = tracks[tid]
  kept, rejected = filter_tracks_by_parallax(tracks, params, 0.004)
  assert list(kept) == list(want_kept)
  assert list(rejected) == list(want_rejected)
  assert 20 < len(kept) < 280


def test_revisit_candidates_are_the_per_keyframe_nearest_later_keyframes() -> None:
  """Candidates pass the gap, window and distance cap, nearest first."""
  from specialist.vi_ba import tracks as t

  rng = np.random.default_rng(5)
  centers = rng.uniform(0.0, 0.3, (400, 3))
  populated = rng.uniform(size=400) > 0.1
  got = t._revisit_candidates(centers, populated, 120)
  want: list[tuple[int, int]] = []
  for i in np.flatnonzero(populated):
    dist = np.linalg.norm(centers - centers[i], axis=1)
    cand = [
      int(j)
      for j in np.argsort(dist, kind="stable")
      if t._REVISIT_MIN_KF_GAP < j - i <= 120 and dist[j] <= t._REVISIT_MAX_DIST
    ][: t._REVISIT_N_NEIGHBORS]
    want.extend((int(i), j) for j in cand)
  assert got == want


def test_matched_components_join_matches_across_keyframes() -> None:
  from specialist.vi_ba.tracks import _matched_components

  edges = [
    (0, 1, np.array([[0, 2], [1, 3]], dtype=np.int64)),
    (1, 2, np.array([[2, 5]], dtype=np.int64)),
    (2, 3, np.array([[9, 0]], dtype=np.int64)),
  ]
  got = _matched_components(edges, [2, 4, 10, 1])
  assert got == [{0: 0, 1: 2, 2: 5}, {0: 1, 1: 3}, {2: 9, 3: 0}]
