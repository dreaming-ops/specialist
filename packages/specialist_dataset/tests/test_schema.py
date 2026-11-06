"""The naming authority: how a camera earns its slot, and what a slot is not."""

import numpy as np
import pytest
from specialist_dataset import schema


def test_the_slot_is_the_serial_rank_not_the_argument_order() -> None:
  serials = ["IAQEB2601V3F6K", "IAQEB2601UVVAJ", "IAQEB2601A0000"]
  assert schema.camera_slots(serials) == ["cam02", "cam01", "cam00"]
  reversed_order = list(reversed(serials))
  assert schema.camera_slots(reversed_order) == ["cam00", "cam01", "cam02"]


def test_a_serial_that_cannot_identify_a_camera_refuses() -> None:
  for serials in ([""], ["A", ""], ["A", "A"], ["", ""]):
    with pytest.raises(ValueError, match="one distinct serial"):
      schema.camera_slots(serials)


def test_a_camera_path_reports_its_slot_and_nothing_else_does() -> None:
  assert schema.slot_of(f"/{schema.video('cam00', 'front')}") == "cam00"
  assert schema.slot_of(schema.state("cam07", "pose")) == "cam07"
  assert schema.slot_of(f"/{schema.anchor(3)}") is None
  assert schema.slot_of("/estimate") is None
  assert schema.slots_in(
    [schema.video("cam01", "rear"), schema.imu("cam00"), "/world/anchor/0"]
  ) == ["cam00", "cam01"]


def test_keyframe_index_buckets_session_frames_by_stride() -> None:
  stride = schema.keyframe_stride(60.0)
  assert stride == 6
  rows = schema.keyframe_index(np.arange(0, 19, dtype=np.int64), stride)
  assert list(rows[:8]) == [0, 1, 1, 1, 1, 1, 1, 2]
  assert rows[stride] == 1 and rows[2 * stride] == 2


def test_a_state_layer_is_named_by_its_configuration() -> None:
  """The directory name carries the configuration; the entity paths inside never do."""
  assert schema.state_layer(12, 100.0) == "state_s12_r100"
  assert schema.state_layer(3, 1000.0) == "state_s3_r1000"
  assert schema.state_image("left") == "state/left/image"
  assert schema.state_pose("right") == "state/right/pose"
  assert schema.slot_of(f"/{schema.state_image('left')}") is None


def test_a_selector_names_the_field_a_query_addresses() -> None:
  assert (
    schema.selector(schema.state("cam00", "pose"), schema.POSE_TRANSLATION)
    == "/camera/cam00/state/pose:Transform3D:translation"
  )
