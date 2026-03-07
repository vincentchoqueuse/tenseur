"""Tenseur — minimal tensor-based musical composition with NumPy."""

__version__ = "0.1.0"

from tenseur.backends.ableton import AbletonOSCClient, Live
from tenseur.core.clip import Clip
from tenseur.core.constants import DUR, PITCH, START, VEL
from tenseur.helpers.create import drum_tensor, euclidean, scatter, simple_tensor
from tenseur.helpers.scales import generate_scale
from tenseur.helpers.transforms import quantize, quantize_pitch, sine, swing, upsample
from tenseur.utils.midi_utils import midi_to_note_name, note_name_to_midi
from tenseur.utils.random import bernoulli, normal, randn, seed, shuffle

__all__ = [
    "PITCH",
    "VEL",
    "START",
    "DUR",
    "Clip",
    "simple_tensor",
    "drum_tensor",
    "euclidean",
    "scatter",
    "sine",
    "swing",
    "quantize",
    "quantize_pitch",
    "upsample",
    "generate_scale",
    "Live",
    "note_name_to_midi",
    "midi_to_note_name",
    "AbletonOSCClient",
    "seed",
    "randn",
    "shuffle",
]
