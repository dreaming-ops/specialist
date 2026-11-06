"""The drop policy for the capture metadata the base recording keeps."""

import numpy as np

from specialist.loader.parse_insv import capture_metadata


def test_superseded_container_and_empty_fields_drop() -> None:
  meta = {
    "serial_number": "IAQEB0000TEST",
    "offset_v3": [1.0, 2.0],  # parsed into its own field
    "offset_v2": [1.0],  # superseded calibration encoding
    "original_offset": [1.0],
    "rolling_shutter_time": 15.7,
    "dimension": {"x": 2880, "y": 2880},  # container-derivable
    "frame_rate": 30,
    "creation_time": 0,  # zero means the clock was never set
    "ip": "",
    "user_options": None,
    "gps_sources": [],
  }
  kept = capture_metadata(meta)
  assert kept == {"serial_number": "IAQEB0000TEST"}


def test_nested_and_array_values_flatten_to_columns() -> None:
  meta = {
    "window_crop_info": {"src_width": 5952, "dst_width": 5760},
    "gyro_cfg_info": {"acc_range": 32, "gyro_range": 2000},
    "photo_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "is_raw_gyro": True,
    "hdr_state": "NotHdr",
    "timelapse_interval": 0,  # zero is a real setting here, not unset
  }
  kept = capture_metadata(meta)
  assert kept["window_crop_info_src_width"] == 5952
  assert kept["gyro_cfg_info_gyro_range"] == 2000
  assert isinstance(kept["photo_rot"], np.ndarray)
  assert kept["photo_rot"].shape == (9,)
  assert kept["is_raw_gyro"] is True
  assert kept["hdr_state"] == "NotHdr"
  assert kept["timelapse_interval"] == 0
