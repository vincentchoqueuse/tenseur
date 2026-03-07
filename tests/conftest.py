"""Shared fixtures for tenseur tests."""

import numpy as np
import pytest

from tenseur.core.clip import Clip
from tenseur.helpers.create import simple_tensor


@pytest.fixture
def single_bar_tensor():
    """A (1, 1, 4, 4) tensor — 1 voice, 1 bar, 4 steps."""
    return simple_tensor(4, pitch=60, velocity=0.8, duration=1.0)


@pytest.fixture
def multi_voice_tensor():
    """A (3, 2, 8, 4) tensor — 3 voices, 2 bars, 8 steps."""
    return simple_tensor((3, 2, 8), pitch=64, velocity=0.6, duration=0.5)


@pytest.fixture
def silent_tensor():
    """A tensor with all velocities set to 0 (inactive)."""
    return simple_tensor(4, pitch=60, velocity=0, duration=1.0)


@pytest.fixture
def sample_clip(single_bar_tensor):
    """A Clip wrapping a single-bar tensor."""
    return Clip(tensor=single_bar_tensor, name="test", channel=0)
