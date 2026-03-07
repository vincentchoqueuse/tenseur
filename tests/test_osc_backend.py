"""Tests for the OSC rendering backend."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tenseur.backends.osc import OscBackend
from tenseur.core.clip import Clip
from tenseur.core.constants import VEL
from tenseur.helpers.create import simple_tensor


class TestOscBackend:
    @patch("pythonosc.udp_client.SimpleUDPClient")
    def test_message_count(self, mock_client_cls, sample_clip):
        """Should send one message per active note."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        count = OscBackend().render_clips([sample_clip], delay=0)
        # sample_clip has 4 active notes (1 voice, 1 bar, 4 steps)
        assert count == 4
        assert mock_client.send_message.call_count == 4

    @patch("pythonosc.udp_client.SimpleUDPClient")
    def test_message_format(self, mock_client_cls, sample_clip):
        """Each message should have [track, channel, pitch, vel, start, dur]."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        OscBackend().render_clips([sample_clip], delay=0)

        for call in mock_client.send_message.call_args_list:
            address, args = call[0]
            assert address == "/tenseur/note"
            assert len(args) == 6
            track, channel, pitch, vel, start, dur = args
            assert isinstance(track, int)
            assert isinstance(channel, int)
            assert isinstance(pitch, int)
            assert isinstance(vel, int)
            assert isinstance(start, float)
            assert isinstance(dur, float)

    @patch("pythonosc.udp_client.SimpleUDPClient")
    def test_silent_tensor(self, mock_client_cls, silent_tensor):
        """No messages should be sent for a silent tensor."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        clip = Clip(tensor=silent_tensor)
        count = OscBackend().render_clips([clip], delay=0)
        assert count == 0
        assert mock_client.send_message.call_count == 0

    @patch("pythonosc.udp_client.SimpleUDPClient")
    def test_custom_address(self, mock_client_cls, sample_clip):
        """Should use the custom OSC address."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        OscBackend().render_clips(
            [sample_clip], address="/custom/note", delay=0
        )

        for call in mock_client.send_message.call_args_list:
            address, _ = call[0]
            assert address == "/custom/note"
