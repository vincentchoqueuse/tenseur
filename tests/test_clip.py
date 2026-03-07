"""Tests for the Clip dataclass."""

import numpy as np
import pytest

from tenseur.core.clip import Clip
from tenseur.helpers.create import simple_tensor


class TestClip:
    def test_creation(self, sample_clip):
        assert sample_clip._track == 1
        assert sample_clip.name == "test"
        assert sample_clip.channel == 0
        assert sample_clip.bar_offset == 0

    def test_defaults(self, single_bar_tensor):
        clip = Clip(tensor=single_bar_tensor)
        assert clip._track == 1
        assert clip.name == ""
        assert clip.channel == 0
        assert clip.bar_offset == 0

    def test_invalid_shape_2d(self):
        bad = np.zeros((4, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="4 dimensions"):
            Clip(tensor=bad)

    def test_invalid_shape_last_dim(self):
        bad = np.zeros((1, 1, 4, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="Last dimension must be 4"):
            Clip(tensor=bad)

    def test_invalid_track_zero(self, single_bar_tensor):
        with pytest.raises(ValueError, match="Track number must be >= 1"):
            Clip(tensor=single_bar_tensor, _track=0)

    def test_invalid_channel_high(self, single_bar_tensor):
        with pytest.raises(ValueError, match="MIDI channel"):
            Clip(tensor=single_bar_tensor, channel=16)

    def test_invalid_channel_negative(self, single_bar_tensor):
        with pytest.raises(ValueError, match="MIDI channel"):
            Clip(tensor=single_bar_tensor, channel=-1)
