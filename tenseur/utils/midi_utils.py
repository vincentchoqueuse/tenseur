"""MIDI/note-name conversion utilities."""

from tenseur.core.constants import MIDI_TO_NOTE, NOTE_TO_MIDI

# Flat-to-sharp enharmonic mapping
_FLAT_TO_SHARP = {
    "Db": "C#",
    "Eb": "D#",
    "Fb": "E",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "Cb": "B",
    "B#": "C",
    "E#": "F",
}


def note_name_to_midi(name: str) -> int:
    """Convert a note name like 'C4', 'c#3', 'Db5' to a MIDI number.

    Handles:
    - Case insensitive note letter (c4 -> C4)
    - Flat names (Db4 -> C#4)
    - Enharmonic edge cases (Cb4 -> B3, B#4 -> C5)

    Raises:
        ValueError: If the note name cannot be parsed.
    """
    if not name or len(name) < 2:
        raise ValueError(f"Invalid note name: {name!r}")

    # Normalize: uppercase first letter
    note_part = name[0].upper()
    rest = name[1:]

    # Extract accidental
    accidental = ""
    if rest and rest[0] in "#b":
        accidental = rest[0]
        rest = rest[1:]

    # Parse octave
    try:
        octave = int(rest)
    except ValueError:
        raise ValueError(f"Invalid note name: {name!r}")

    note = note_part + accidental

    # Handle enharmonic conversions
    if note in _FLAT_TO_SHARP:
        sharp = _FLAT_TO_SHARP[note]
        # Adjust octave for Cb (Cb4 -> B3) and B# (B#4 -> C5)
        if note == "Cb":
            octave -= 1
        elif note == "B#":
            octave += 1
        note = sharp

    canonical = f"{note}{octave}"
    if canonical not in NOTE_TO_MIDI:
        raise ValueError(f"Note out of MIDI range: {name!r} (resolved to {canonical})")
    return NOTE_TO_MIDI[canonical]


def midi_to_note_name(midi: int) -> str:
    """Convert a MIDI number (0-127) to a canonical note name (sharps only).

    Raises:
        ValueError: If the MIDI number is out of range.
    """
    if midi not in MIDI_TO_NOTE:
        raise ValueError(f"MIDI number out of range [0, 127]: {midi}")
    return MIDI_TO_NOTE[midi]


def parse_pitch(pitch: int | str) -> int:
    """Parse a pitch given as MIDI int or note name string.

    Returns:
        MIDI note number as int.

    Raises:
        ValueError: If the pitch cannot be parsed.
        TypeError: If the pitch is neither int nor str.
    """
    if isinstance(pitch, int):
        if not 0 <= pitch <= 127:
            raise ValueError(f"MIDI pitch out of range [0, 127]: {pitch}")
        return pitch
    if isinstance(pitch, str):
        return note_name_to_midi(pitch)
    raise TypeError(f"Pitch must be int or str, got {type(pitch).__name__}")
