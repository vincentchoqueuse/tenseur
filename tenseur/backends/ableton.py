"""Ableton Live OSC backend — send clips directly into Ableton via AbletonOSC."""

import threading
import time
from typing import Any, Callable, Iterable

import numpy as np

from tenseur.backends.base import Backend
from tenseur.core.constants import DUR, PITCH, START, VEL

# Default AbletonOSC ports
DEFAULT_ABLETON_REMOTE_PORT = 11000
DEFAULT_ABLETON_LOCAL_PORT = 11001
DEFAULT_QUERY_TIMEOUT = 0.150


class AbletonOSCClient:
    """Client for communicating with Ableton Live via the AbletonOSC protocol.

    Manages bidirectional OSC communication: sends commands to Ableton and
    receives responses via a background server thread.

    Args:
        hostname: Remote host (default "127.0.0.1").
        port: Remote port (default 11000).
        client_port: Local port for receiving replies (default 11001).
    """

    def __init__(
        self,
        hostname: str = "127.0.0.1",
        port: int = DEFAULT_ABLETON_REMOTE_PORT,
        client_port: int = DEFAULT_ABLETON_LOCAL_PORT,
    ):
        from pythonosc.dispatcher import Dispatcher
        from pythonosc.osc_server import ThreadingOSCUDPServer
        from pythonosc.udp_client import SimpleUDPClient

        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self._handle_osc_message)

        self.server = ThreadingOSCUDPServer(("0.0.0.0", client_port), dispatcher)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

        self.address_handlers: dict[str, Callable] = {}
        self.client = SimpleUDPClient(hostname, port)
        self.verbose = False

    def _handle_osc_message(self, address: str, *params: Any) -> None:
        if address in self.address_handlers:
            self.address_handlers[address](address, params)
        if self.verbose:
            print(f"Received OSC: {address} {params}")

    def stop(self) -> None:
        """Stop the OSC server and clean up."""
        self.server.shutdown()
        self.server_thread.join()
        self.server = None

    # ── Low-level messaging ──────────────────────────────────────────

    def send_message(self, address: str, params: Iterable = ()) -> None:
        """Send a single OSC message."""
        self.client.send_message(address, params)

    def send_bundle(self, messages: list[tuple[str, tuple]]) -> None:
        """Send multiple OSC messages as a single bundle."""
        from pythonosc.osc_bundle_builder import OscBundleBuilder
        from pythonosc.osc_message_builder import OscMessageBuilder

        bundle_builder = OscBundleBuilder(int(time.time()))
        for address, params in messages:
            msg = OscMessageBuilder(address=address)
            for param in params:
                msg.add_arg(param)
            bundle_builder.add_content(msg.build())
        self.client.send(bundle_builder.build())

    def set_handler(self, address: str, callback: Callable) -> None:
        """Register a callback for an OSC address."""
        self.address_handlers[address] = callback

    def remove_handler(self, address: str) -> None:
        """Remove a handler for an OSC address."""
        del self.address_handlers[address]

    def await_message(self, address: str, timeout: float = DEFAULT_QUERY_TIMEOUT) -> tuple:
        """Wait for a message at a specific address. Returns the params."""
        response_params = None
        event = threading.Event()

        def handler(addr: str, params: tuple) -> None:
            nonlocal response_params
            response_params = params
            event.set()

        self.set_handler(address, handler)
        event.wait(timeout)
        self.remove_handler(address)
        if not event.is_set():
            raise RuntimeError(f"No response received for address: {address}")
        return response_params

    def query(self, address: str, params: tuple = (), timeout: float = DEFAULT_QUERY_TIMEOUT) -> tuple:
        """Send a message and wait for a response at the same address."""
        response_params = None
        event = threading.Event()

        def handler(addr: str, params: tuple) -> None:
            nonlocal response_params
            response_params = params
            event.set()

        self.set_handler(address, handler)
        self.send_message(address, params)
        event.wait(timeout)
        self.remove_handler(address)
        if not event.is_set():
            raise RuntimeError(f"No response received to query: {address}")
        return response_params

    # ── Ableton Live high-level API ──────────────────────────────────

    def set_tempo(self, tempo_bpm: float) -> None:
        """Set the song tempo."""
        self.send_message("/live/song/set/tempo", (tempo_bpm,))

    def set_time_signature(self, numerator: int, denominator: int = 4) -> None:
        """Set the song time signature."""
        self.send_message("/live/song/set/signature_numerator", (numerator,))
        self.send_message("/live/song/set/signature_denominator", (denominator,))

    def create_clip(self, track_index: int, index: int, length_bars: float = 4.0) -> None:
        """Create a new MIDI clip in a slot."""
        self.send_message("/live/clip_slot/create_clip", (track_index, index, length_bars))

    def delete_clip(self, track_index: int, index: int) -> None:
        """Delete a clip from a slot."""
        self.send_message("/live/clip_slot/delete_clip", (track_index, index))

    def set_clip_info(self, track_index: int, index: int, color: int | None = None, name: str | None = None) -> None:
        """Set clip metadata (color, name)."""
        if color is not None:
            self.send_message("/live/clip/set/color", (track_index, index, color))
        if name is not None:
            self.send_message("/live/clip/set/name", (track_index, index, name))

    def set_loop_start(self, track_index: int, index: int, loop_start: float) -> None:
        """Set the loop start point in beats."""
        self.send_message("/live/clip/set/loop_start", (track_index, index, loop_start))

    def set_loop_end(self, track_index: int, index: int, loop_end: float) -> None:
        """Set the loop end point in beats."""
        self.send_message("/live/clip/set/loop_end", (track_index, index, loop_end))

    def get_notes(self, track_index: int, index: int) -> list[list]:
        """Get all notes from a clip. Returns list of [pitch, start, dur, vel, mute]."""
        response = self.query("/live/clip/get/notes", (track_index, index))
        flat = response[2:]
        return [flat[i:i + 5] for i in range(0, len(flat), 5)]

    def add_notes(self, track_index: int, index: int, notes: list[list]) -> None:
        """Add notes to a clip. Each note: [pitch, start, duration, velocity]."""
        flat = []
        for note in notes:
            pitch = max(min(int(note[0]), 127), 0)
            velocity = max(min(int(note[3]), 127), 0)
            flat.extend([pitch, float(note[1]), float(note[2]), velocity, False])
        self.send_message("/live/clip/add/notes", (track_index, index, *flat))

    def clear_notes(
        self,
        track_index: int,
        index: int,
        start_pitch: int = 0,
        pitch_span: int = 128,
        start_time: float = 0.0,
        time_span: float = 16.0,
    ) -> None:
        """Remove notes from a clip within specified ranges."""
        self.send_message(
            "/live/clip/remove/notes",
            (track_index, index, start_pitch, pitch_span, start_time, time_span),
        )

    def set_notes(self, track_index: int, index: int, notes: list[list], time_span: float | None = None) -> None:
        """Replace all notes in a clip (clear then add)."""
        if time_span is None and len(notes) > 0:
            time_span = max(n[1] + n[2] for n in notes) + 1.0
        elif time_span is None:
            time_span = 16.0
        self.clear_notes(track_index, index, time_span=time_span)
        self.add_notes(track_index, index, notes)


class Live:
    """Persistent connection to Ableton Live via OSC.

    Opens the connection once at init; use push(clip) to send clips,
    and stop() when done.

    Args:
        index: Clip-slot number (1-based, default 1).
        bpm: If set, change the song tempo on connect.
        beats_per_bar: Beats per bar in Ableton (default 4).
        steps_per_bar: Steps per bar (auto-detected from first clip if None).
        create: Whether to create new clips in Ableton (default True).
        **kwargs: hostname, port, client_port forwarded to AbletonOSCClient.
    """

    def __init__(
        self,
        index: int = 1,
        bpm: float | None = None,
        beats_per_bar: int = 4,
        steps_per_bar: int | None = None,
        create: bool = True,
        dry: bool = False,
        **kwargs: Any,
    ):
        self.index = index
        self.beats_per_bar = beats_per_bar
        self.steps_per_bar = steps_per_bar
        self.create = create
        self.dry = dry
        if dry:
            self.client = None
            return
        self.client = AbletonOSCClient(**{
            k: v for k, v in kwargs.items()
            if k in ("hostname", "port", "client_port")
        })
        if bpm is not None:
            self.client.set_tempo(bpm)
        if beats_per_bar != 4:
            self.client.set_time_signature(beats_per_bar)

    def push(self, clip: Any) -> None:
        """Send a single clip to Ableton Live. No-op in dry mode."""
        if self.dry:
            return

        from tenseur.backends import _log_render

        _log_render([clip], "ableton", index=self.index)

        tensor = clip.tensor
        track = clip._track - 1
        bar_off = clip.bar_offset
        V, B, S, _ = tensor.shape

        clip_idx_0 = self.index - 1
        steps_per_bar = self.steps_per_bar if self.steps_per_bar is not None else S
        beats_per_step = self.beats_per_bar / steps_per_bar

        if clip.name:
            self.client.set_clip_info(track, clip_idx_0, name=clip.name)

        notes = []
        for v in range(V):
            for b in range(B):
                for s in range(S):
                    note = tensor[v, b, s]
                    if note[VEL] <= 0:
                        continue
                    vel = int(np.clip(note[VEL] * 127, 1, 127))
                    pitch = int(np.clip(note[PITCH], 0, 127))
                    start_step = note[START] + bar_off * steps_per_bar
                    start_beat = max(0.0, start_step * beats_per_step)
                    dur_beat = max(0.001, note[DUR] * beats_per_step)
                    notes.append([pitch, start_beat, dur_beat, vel])

        length_beats = clip.bars * self.beats_per_bar

        if self.create:
            self.client.create_clip(track, clip_idx_0, length_beats)

        self.client.set_loop_start(track, clip_idx_0, 0.0)
        self.client.set_loop_end(track, clip_idx_0, length_beats)

        if notes:
            self.client.set_notes(track, clip_idx_0, notes, time_span=length_beats)

    def stop(self) -> None:
        """Close the OSC connection. No-op in dry mode."""
        if not self.dry:
            self.client.stop()


class AbletonBackend(Backend):
    """Renders Tenseur clips into Ableton Live via AbletonOSC.

    Converts tensor data to Ableton's note format and pushes clips
    into Ableton's session view using the AbletonOSC protocol.
    """

    def render_clips(
        self,
        clips: list,
        client: AbletonOSCClient | None = None,
        index: int = 1,
        bpm: float | None = None,
        steps_per_bar: int | None = None,
        beats_per_bar: int = 4,
        create: bool = True,
        **kwargs: Any,
    ) -> AbletonOSCClient:
        """Push clips into Ableton Live.

        Each Clip is placed on its `.track` at the given `index` slot.
        The tensor's START/DUR values (in steps) are converted to beats.
        Clip length is determined by each clip's `.bars` attribute.

        Args:
            clips: List of Clip instances.
            client: Existing AbletonOSCClient (created if None).
            index: Clip-slot number for all clips (1-based).
            bpm: If set, change the song tempo.
            steps_per_bar: Steps per bar (defaults to S from the first clip's tensor).
            beats_per_bar: Beats per bar in Ableton (default 4).
            create: Whether to create new clips (default True).
            **kwargs: Passed to AbletonOSCClient constructor if creating one.

        Returns:
            The AbletonOSCClient instance (for further interaction).
        """
        own_client = client is None
        if own_client:
            client = AbletonOSCClient(**{
                k: v for k, v in kwargs.items()
                if k in ("hostname", "port", "client_port")
            })

        if bpm is not None:
            client.set_tempo(bpm)
        if beats_per_bar != 4:
            client.set_time_signature(beats_per_bar)

        # Convert 1-based user-facing indices to 0-based for Ableton OSC protocol
        clip_idx_0 = index - 1
        if steps_per_bar is None:
            steps_per_bar = clips[0].tensor.shape[2]
        beats_per_step = beats_per_bar / steps_per_bar

        for clip in clips:
            tensor = clip.tensor
            track = clip._track - 1
            bar_off = clip.bar_offset
            V, B, S, _ = tensor.shape

            if clip.name:
                client.set_clip_info(track, clip_idx_0, name=clip.name)

            # Convert tensor notes to Ableton format [pitch, start_beat, dur_beat, velocity]
            notes = []
            for v in range(V):
                for b in range(B):
                    for s in range(S):
                        note = tensor[v, b, s]
                        if note[VEL] <= 0:
                            continue
                        vel = int(np.clip(note[VEL] * 127, 1, 127))
                        pitch = int(np.clip(note[PITCH], 0, 127))
                        start_step = note[START] + bar_off * steps_per_bar
                        start_beat = max(0.0, start_step * beats_per_step)
                        dur_beat = max(0.001, note[DUR] * beats_per_step)
                        notes.append([pitch, start_beat, dur_beat, vel])

            length_beats = clip.bars * beats_per_bar

            if create:
                client.create_clip(track, clip_idx_0, length_beats)

            client.set_loop_start(track, clip_idx_0, 0.0)
            client.set_loop_end(track, clip_idx_0, length_beats)

            if notes:
                client.set_notes(track, clip_idx_0, notes, time_span=length_beats)

        return client
