"""Tests for the MIDI rendering backend."""

import os
import tempfile

import mido
import numpy as np
import pytest

from tenseur.backends.midi import MidiBackend
from tenseur.core.clip import Clip
from tenseur.core.constants import DUR, PITCH, START, VEL
from tenseur.helpers.create import simple_tensor


class TestMidiBackend:
    def test_roundtrip(self, sample_clip):
        """Render a clip and read it back, verify notes exist."""
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            path = f.name
        try:
            MidiBackend().render_clips([sample_clip], filename=path)
            mid = mido.MidiFile(path)
            # Should have tempo track + 1 data track
            assert len(mid.tracks) == 2
            # Data track should have note_on messages
            note_ons = [m for m in mid.tracks[1] if m.type == "note_on"]
            assert len(note_ons) > 0
        finally:
            os.unlink(path)

    def test_multi_track(self):
        """Multiple clips on different tracks produce separate MIDI tracks."""
        t1 = simple_tensor(4, pitch=60, velocity=0.8, duration=1.0)
        t2 = simple_tensor(4, pitch=72, velocity=0.6, duration=0.5)
        clip1 = Clip(tensor=t1)
        clip2 = Clip(tensor=t2).track(2)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            path = f.name
        try:
            MidiBackend().render_clips([clip1, clip2], filename=path)
            mid = mido.MidiFile(path)
            # tempo + 2 data tracks
            assert len(mid.tracks) == 3
        finally:
            os.unlink(path)

    def test_velocity_clamping(self):
        """Velocities above 1.0 should be clamped to 127 at render."""
        t = simple_tensor(2, pitch=60, velocity=1.5, duration=1.0)
        clip = Clip(tensor=t)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            path = f.name
        try:
            MidiBackend().render_clips([clip], filename=path)
            mid = mido.MidiFile(path)
            for msg in mid.tracks[1]:
                if msg.type == "note_on":
                    assert msg.velocity <= 127
        finally:
            os.unlink(path)

    def test_empty_clip(self, silent_tensor):
        """A clip with no active notes produces no note events."""
        clip = Clip(tensor=silent_tensor)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            path = f.name
        try:
            MidiBackend().render_clips([clip], filename=path)
            mid = mido.MidiFile(path)
            # Should still have tempo track but data track has no notes
            data_tracks = mid.tracks[1:]
            for track in data_tracks:
                note_ons = [m for m in track if m.type == "note_on"]
                assert len(note_ons) == 0
        finally:
            os.unlink(path)
