"""Tests for tensor creation and transform helpers."""

import numpy as np
import pytest

from tenseur.core.constants import DUR, PITCH, START, VEL
from tenseur.helpers.create import simple_tensor
from tenseur.helpers.transforms import humanize, quantize_pitch, swing


class TestSimpleTensor:
    def test_int_size(self):
        t = simple_tensor(8)
        assert t.shape == (1, 1, 8, 4)
        assert t.dtype == np.float64

    def test_tuple_2d_size(self):
        t = simple_tensor((3, 8))
        assert t.shape == (3, 1, 8, 4)

    def test_tuple_3d_size(self):
        t = simple_tensor((2, 3, 16))
        assert t.shape == (2, 3, 16, 4)

    def test_values(self):
        t = simple_tensor(4, pitch=72, velocity=0.7, duration=0.5)
        assert np.all(t[..., PITCH] == 72)
        assert np.all(t[..., VEL] == 0.7)
        assert np.all(t[..., DUR] == 0.5)

    def test_start_times_single_bar(self):
        t = simple_tensor(4)
        starts = t[0, 0, :, START]
        np.testing.assert_array_equal(starts, [0, 1, 2, 3])

    def test_start_times_multi_bar(self):
        t = simple_tensor((1, 2, 4), steps_per_bar=16)
        # Bar 0: [0,1,2,3], Bar 1: [16,17,18,19]
        np.testing.assert_array_equal(t[0, 0, :, START], [0, 1, 2, 3])
        np.testing.assert_array_equal(t[0, 1, :, START], [16, 17, 18, 19])


class TestSwing:
    def test_swing_shifts_odd_steps(self, single_bar_tensor):
        result = swing(single_bar_tensor, amount=0.5)
        orig_starts = single_bar_tensor[0, 0, :, START]
        new_starts = result[0, 0, :, START]
        # Even steps unchanged
        np.testing.assert_array_equal(new_starts[0::2], orig_starts[0::2])
        # Odd steps shifted
        np.testing.assert_allclose(new_starts[1::2], orig_starts[1::2] + 0.5)

    def test_swing_does_not_mutate(self, single_bar_tensor):
        original = single_bar_tensor.copy()
        swing(single_bar_tensor, amount=0.5)
        np.testing.assert_array_equal(single_bar_tensor, original)


class TestHumanize:
    def test_noise_added(self, single_bar_tensor):
        result = humanize(single_bar_tensor, velocity=0.08, timing=0.1, seed=42)
        # Should differ from original
        assert not np.array_equal(result[..., VEL], single_bar_tensor[..., VEL])
        assert not np.array_equal(result[..., START], single_bar_tensor[..., START])

    def test_velocity_clipped(self, single_bar_tensor):
        result = humanize(single_bar_tensor, velocity=2.0, timing=0, seed=0)
        active = result[..., VEL] > 0
        assert np.all(result[..., VEL][active] > 0)
        assert np.all(result[..., VEL][active] <= 1.0)

    def test_inactive_preserved(self, silent_tensor):
        result = humanize(silent_tensor, velocity=0.08, timing=0.1, seed=42)
        assert np.all(result[..., VEL] == 0)

    def test_does_not_mutate(self, single_bar_tensor):
        original = single_bar_tensor.copy()
        humanize(single_bar_tensor, velocity=0.04, timing=0.05, seed=42)
        np.testing.assert_array_equal(single_bar_tensor, original)


class TestQuantizePitch:
    def test_snap_to_nearest(self):
        t = simple_tensor(4, pitch=61)  # between C4(60) and D4(62)
        result = quantize_pitch(t, [60, 62, 64])
        # 61 is equidistant; ties go to lower (left)
        assert np.all(result[..., PITCH][result[..., VEL] > 0] == 60)

    def test_exact_match(self):
        t = simple_tensor(4, pitch=64)
        result = quantize_pitch(t, [60, 62, 64, 67])
        assert np.all(result[..., PITCH][result[..., VEL] > 0] == 64)

    def test_inactive_unchanged(self, silent_tensor):
        original_pitches = silent_tensor[..., PITCH].copy()
        result = quantize_pitch(silent_tensor, [60, 64, 67])
        np.testing.assert_array_equal(result[..., PITCH], original_pitches)

    def test_does_not_mutate(self, single_bar_tensor):
        original = single_bar_tensor.copy()
        quantize_pitch(single_bar_tensor, [60, 64, 67])
        np.testing.assert_array_equal(single_bar_tensor, original)
