"""OSC rendering backend using python-osc."""

import time
from typing import Any

import numpy as np

from tenseur.backends.base import Backend
from tenseur.core.constants import DUR, PITCH, START, VEL


class OscBackend(Backend):
    """Sends clip data as OSC messages via UDP."""

    def render_clips(
        self,
        clips: list,
        ip: str = "127.0.0.1",
        port: int = 9000,
        address: str = "/tenseur/note",
        steps_per_bar: int | None = None,
        delay: float = 0.001,
        **kwargs: Any,
    ) -> int:
        """Send clips as OSC messages.

        Each active note sends one message:
            [track, channel, pitch, velocity, absolute_start, duration]

        Args:
            clips: List of Clip instances.
            ip: Target IP address.
            port: Target UDP port.
            address: OSC address prefix.
            steps_per_bar: Steps per bar.
            delay: Delay in seconds between messages.
            **kwargs: Ignored.

        Returns:
            Number of messages sent.
        """
        from pythonosc.udp_client import SimpleUDPClient

        if steps_per_bar is None:
            steps_per_bar = clips[0].tensor.shape[2]

        client = SimpleUDPClient(ip, port)
        count = 0

        for clip in clips:
            tensor = clip.tensor
            track = clip._track
            channel = clip.channel
            bar_off = clip.bar_offset

            V, B, S, _ = tensor.shape
            for v in range(V):
                for b in range(B):
                    for s in range(S):
                        note = tensor[v, b, s]
                        if note[VEL] <= 0:
                            continue
                        vel = int(np.clip(note[VEL] * 127, 1, 127))
                        pitch = int(np.clip(note[PITCH], 0, 127))
                        abs_start = float(note[START] + bar_off * steps_per_bar)
                        dur = float(note[DUR])

                        client.send_message(
                            address,
                            [track, channel, pitch, vel, abs_start, dur],
                        )
                        count += 1
                        if delay > 0:
                            time.sleep(delay)

        return count
