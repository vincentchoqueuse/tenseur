"""Tensor dimension indices and MIDI/note-name mappings."""

# Tensor dimension indices (axis -1 of the 4D tensor)
PITCH = 0
VEL = 1
START = 2
DUR = 3

DEFAULT_STEPS_PER_BAR = 16

# Programmatic sharps-only canonical mapping
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

NOTE_TO_MIDI: dict[str, int] = {}
MIDI_TO_NOTE: dict[int, str] = {}

for _midi in range(128):
    _octave = (_midi // 12) - 1
    _name = f"{_NOTE_NAMES[_midi % 12]}{_octave}"
    NOTE_TO_MIDI[_name] = _midi
    MIDI_TO_NOTE[_midi] = _name
