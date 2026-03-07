"""Scale and pitch-set generators."""

import numpy as np

from tenseur.core.types import PitchSet


def generate_scale(
    divisions: int | float = 12,
    root: int = 60,
    offset: float = 0.0,
) -> PitchSet:
    """Generate a pitch set from equal division of the octave (EDO).

    Divides the octave into `divisions` equal parts and returns all
    resulting MIDI pitches in the valid range [0, 127].

    Args:
        divisions: Number of equal divisions per octave.
            12 = chromatic (12-TET), 7 = roughly diatonic,
            5 = roughly pentatonic, 19 = microtonal, etc.
        root: Root pitch as MIDI number (default 60 = C4).
        offset: Pitch offset in semitones (for detuning/shifting).

    Returns:
        Sorted list of MIDI pitch values (int), usable with quantize_pitch.

    Examples:
        >>> generate_scale(12)          # chromatic
        >>> generate_scale(7, root=60)  # ~diatonic from C4
        >>> generate_scale(5, offset=-0.15)  # detuned pentatonic
    """
    scale_factor = 12 / divisions
    raw = root + np.rint(offset + scale_factor * np.arange(-127, 127))
    pitches = raw[(raw >= 0) & (raw <= 127)].astype(int)
    return sorted(set(pitches.tolist()))
