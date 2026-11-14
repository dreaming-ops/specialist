"""Continuous captures: reject ambiguous pairs, preserve scene-change gaps."""

import json
from pathlib import Path

import numpy as np
import pytest
from specialist_dataset.episodes import (
  Episode,
  Marker,
  TapConfig,
  detect,
  manifest,
  propose,
  read_manifest,
  triple_taps,
  validate,
)
from specialist_dataset.state import hold


def signal(hits: list[float]) -> tuple[np.ndarray, np.ndarray]:
  t = np.arange(0, 60, 0.002)
  a = np.zeros((len(t), 3))
  a[:, 2] = 9.81
  for at in hits:
    a[np.argmin(abs(t - at)), 2] = 45
  return t, a


def test_two_episodes_exclude_scene_reset() -> None:
  hits = [at + d for at in (2, 12, 30, 40) for d in (0, 0.3, 0.6)]
  t, a = signal(hits)
  _, b = signal([h + 0.04 for h in hits])
  markers, warnings = detect({"a": t, "b": t}, {"a": a, "b": b})
  assert not warnings
  episodes = propose(markers, (0, 60))
  assert len(episodes) == 2
  assert episodes[0].capture_start == pytest.approx(2.84)
  assert episodes[0].task_end == 12
  assert episodes[0].capture_end < episodes[1].capture_start


@pytest.mark.parametrize("hits", [[], [2], [2, 2.3], [2, 2.3, 2.6, 2.9]])
def test_only_exact_triples_are_markers(hits: list[float]) -> None:
  t, a = signal(hits)
  assert triple_taps(t, a, TapConfig()) == []


def test_rebound_is_debounced() -> None:
  t, a = signal([2, 2.04, 2.3, 2.34, 2.6, 2.64])
  assert triple_taps(t, a, TapConfig()) == [Marker(2, 2.6)]


def test_strongest_impact_wins_over_weak_early_contact() -> None:
  t, a = signal([1.96, 2.26, 2.56])
  for at in [2, 2.3, 2.6]:
    a[np.argmin(abs(t - at)), 2] = 210
  assert triple_taps(t, a, TapConfig()) == [Marker(2, 2.6)]


def test_small_rebounds_outside_debounce_do_not_hide_triple() -> None:
  t, a = signal([2.14, 2.44, 2.74])
  for at in [2, 2.3, 2.6]:
    a[np.argmin(abs(t - at)), 2] = 210
  assert triple_taps(t, a, TapConfig()) == [Marker(2, 2.6)]


def test_weak_contacts_cannot_bridge_separate_strong_impacts() -> None:
  t, a = signal([2.4, 2.8, 3.4, 3.8])
  for at in [2, 3, 4]:
    a[np.argmin(abs(t - at)), 2] = 210
  assert triple_taps(t, a, TapConfig()) == []


def test_missing_camera_triple_is_reported() -> None:
  t, a = signal([2, 2.3, 2.6])
  _, b = signal([2, 2.3])
  markers, warnings = detect({"a": t, "b": t}, {"a": a, "b": b})
  assert not markers and warnings


def test_unclosed_episode_is_rejected() -> None:
  with pytest.raises(ValueError, match="even marker count"):
    propose([Marker(2, 2.6)], (0, 60))


def test_review_and_session_identity_are_required(tmp_path: Path) -> None:
  episodes = [Episode("ep0001", 3, 12, 13, "stack blocks")]
  data = manifest("session", [], episodes, [])
  path = tmp_path / "episodes.json"
  path.write_text(json.dumps(data))
  with pytest.raises(ValueError, match="review"):
    read_manifest(path, "session", (0, 60))
  data["reviewed"] = True
  path.write_text(json.dumps(data))
  assert read_manifest(path, "session", (0, 60)) == episodes
  with pytest.raises(ValueError, match="different session"):
    read_manifest(path, "wrong", (0, 60))


@pytest.mark.parametrize(
  "episode",
  [
    Episode("../bad", 3, 12, 13),
    Episode("a", float("nan"), 12, 13),
    Episode("a", 3, 12, 65),
    Episode("a", 12, 3, 13),
  ],
)
def test_invalid_boundaries_rejected(episode: Episode) -> None:
  with pytest.raises(ValueError):
    validate([episode], (0, 60))


def test_one_second_hold_on_short_split_episode() -> None:
  t = np.arange(30, 35, 0.01)
  speed = np.where(t < 31.2, 0.0, 1.0)
  assert hold(
    t, np.ones(len(t)), speed, (30, 35), min_duration=1, head_fraction=1
  ) == pytest.approx(31.19)
  with pytest.raises(ValueError, match="under"):
    hold(
      t,
      np.ones(len(t)),
      np.where(t < 30.5, 0.0, 1.0),
      (30, 35),
      min_duration=1,
      head_fraction=1,
    )
