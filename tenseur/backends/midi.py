"""MIDI file rendering backend using mido."""

from collections import defaultdict
from typing import Any

import mido
import numpy as np

from tenseur.backends.base import Backend
from tenseur.core.constants import DUR, PITCH, START, VEL


class MidiBackend(Backend):
    """Renders clips to a Type-1 MIDI file."""

    def render_clips(
        self,
        clips: list,
        filename: str = "output.mid",
        bpm: float = 120.0,
        steps_per_bar: int | None = None,
        ticks_per_step: int = 120,
        **kwargs: Any,
    ) -> str:
        """Render clips to a MIDI file.

        Args:
            clips: List of Clip instances.
            filename: Output .mid file path.
            bpm: Tempo in beats per minute.
            steps_per_bar: Steps per bar (defaults to S from the first clip's tensor).
            ticks_per_step: MIDI ticks per step (default 120, giving 480 ticks/beat at 4 steps/beat).
            **kwargs: Ignored.

        Returns:
            The filename that was written.
        """
        if steps_per_bar is None:
            steps_per_bar = clips[0].tensor.shape[2]
        ticks_per_beat = ticks_per_step * (steps_per_bar // 4)
        mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

        # Tempo track
        tempo_track = mido.MidiTrack()
        tempo_track.append(
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0)
        )
        mid.tracks.append(tempo_track)

        # Group clips by track index
        tracks: dict[int, list] = defaultdict(list)
        for clip in clips:
            tracks[clip._track].append(clip)

        for track_idx in sorted(tracks.keys()):
            midi_track = mido.MidiTrack()
            events: list[tuple[int, int, str, int, int]] = []  # (tick, order, type, pitch, vel)

            for clip in tracks[track_idx]:
                tensor = clip.tensor
                channel = clip.channel
                bar_off = clip.bar_offset

                # Flatten across V, B, S
                V, B, S, _ = tensor.shape
                for v in range(V):
                    for b in range(B):
                        for s in range(S):
                            note = tensor[v, b, s]
                            if note[VEL] <= 0:
                                continue
                            vel = int(np.clip(note[VEL] * 127, 1, 127))
                            pitch = int(np.clip(note[PITCH], 0, 127))
                            start_step = note[START] + bar_off * steps_per_bar
                            dur_steps = note[DUR]

                            on_tick = max(0, int(round(start_step * ticks_per_step)))
                            off_tick = max(on_tick, int(round((start_step + dur_steps) * ticks_per_step)))

                            events.append((on_tick, 0, "note_on", pitch, vel))
                            events.append((off_tick, 1, "note_off", pitch, 0))

            # Sort by tick, then note_off before note_on at same tick
            events.sort(key=lambda e: (e[0], e[1]))

            prev_tick = 0
            for tick, _, msg_type, pitch, vel in events:
                delta = tick - prev_tick
                midi_track.append(
                    mido.Message(msg_type, note=pitch, velocity=vel, time=delta)
                )
                prev_tick = tick

            mid.tracks.append(midi_track)

        mid.save(filename)
        return filename
