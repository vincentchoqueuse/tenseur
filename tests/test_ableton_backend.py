"""Tests for the Ableton Live OSC backend."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tenseur.backends.ableton import AbletonBackend, AbletonOSCClient
from tenseur.core.clip import Clip
from tenseur.core.constants import DUR, PITCH, START, VEL
from tenseur.helpers.create import simple_tensor


class TestAbletonBackend:
    def _make_mock_client(self):
        client = MagicMock(spec=AbletonOSCClient)
        return client

    def test_render_creates_clip(self, sample_clip):
        """Should call create_clip on the Ableton client."""
        client = self._make_mock_client()
        AbletonBackend().render_clips([sample_clip], client=client, index=1)
        client.create_clip.assert_called_once()

    def test_render_sets_notes(self, sample_clip):
        """Should call set_notes with converted note data."""
        client = self._make_mock_client()
        AbletonBackend().render_clips([sample_clip], client=client, index=1)
        client.set_notes.assert_called_once()
        _, _, notes, _ = client.set_notes.call_args[0][0], client.set_notes.call_args[0][1], client.set_notes.call_args[0][2], None
        # sample_clip has 4 active notes
        assert len(notes) == 4

    def test_render_note_format(self, sample_clip):
        """Each note should be [pitch, start_beat, dur_beat, velocity]."""
        client = self._make_mock_client()
        AbletonBackend().render_clips(
            [sample_clip], client=client, index=1,
            steps_per_bar=16, beats_per_bar=4,
        )
        args = client.set_notes.call_args
        notes = args[0][2]
        for note in notes:
            assert len(note) == 4
            pitch, start_beat, dur_beat, vel = note
            assert isinstance(pitch, int)
            assert isinstance(start_beat, float)
            assert isinstance(dur_beat, float)
            assert isinstance(vel, int)
            assert 0 <= pitch <= 127
            assert 0 <= vel <= 127

    def test_render_silent_clip(self, silent_tensor):
        """No notes should be sent for a silent tensor."""
        client = self._make_mock_client()
        clip = Clip(tensor=silent_tensor)
        AbletonBackend().render_clips([clip], client=client, index=1)
        client.set_notes.assert_not_called()

    def test_render_sets_tempo(self, sample_clip):
        """Should set tempo when bpm is provided."""
        client = self._make_mock_client()
        AbletonBackend().render_clips([sample_clip], client=client, bpm=140.0)
        client.set_tempo.assert_called_once_with(140.0)

    def test_render_sets_clip_name(self):
        """Should set clip name when provided."""
        t = simple_tensor(4, pitch=60, velocity=0.8, duration=1.0)
        clip = Clip(tensor=t, name="kick")
        client = self._make_mock_client()
        AbletonBackend().render_clips([clip], client=client, index=1)
        client.set_clip_info.assert_called_once_with(0, 0, name="kick")

    def test_render_multi_track(self):
        """Clips on different tracks should create separate Ableton clips."""
        t1 = simple_tensor(4, pitch=60, velocity=0.8, duration=1.0)
        t2 = simple_tensor(4, pitch=72, velocity=0.6, duration=0.5)
        clip1 = Clip(tensor=t1)
        clip2 = Clip(tensor=t2).track(2)
        client = self._make_mock_client()
        AbletonBackend().render_clips([clip1, clip2], client=client, index=1)
        assert client.create_clip.call_count == 2
        assert client.set_notes.call_count == 2

    def test_step_to_beat_conversion(self):
        """Verify steps are correctly converted to beats."""
        # 16 steps per bar, 4 beats per bar -> 0.25 beats per step
        t = simple_tensor(4, pitch=60, velocity=0.8, duration=2.0)
        clip = Clip(tensor=t)
        client = self._make_mock_client()
        AbletonBackend().render_clips(
            [clip], client=client, steps_per_bar=16, beats_per_bar=4,
        )
        notes = client.set_notes.call_args[0][2]
        # Step 0 -> beat 0, Step 1 -> beat 0.25, etc.
        assert notes[0][1] == 0.0  # start beat
        assert abs(notes[1][1] - 0.25) < 1e-9
        # Duration: 2.0 steps * 0.25 = 0.5 beats
        assert abs(notes[0][2] - 0.5) < 1e-9

    def test_no_create_flag(self, sample_clip):
        """When create=False, should not call create_clip."""
        client = self._make_mock_client()
        AbletonBackend().render_clips([sample_clip], client=client, create=False)
        client.create_clip.assert_not_called()

    def test_returns_client(self, sample_clip):
        """Should return the client instance."""
        client = self._make_mock_client()
        result = AbletonBackend().render_clips([sample_clip], client=client)
        assert result is client
