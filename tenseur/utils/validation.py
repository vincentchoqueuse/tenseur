"""Validation utilities for tensor shapes and MIDI ranges."""

import warnings

import numpy as np

from tenseur.core.types import Tensor


def validate_tensor_shape(tensor: Tensor) -> None:
    """Validate that a tensor has shape (V, B, S, 4).

    Raises:
        ValueError: If the tensor does not have exactly 4 dimensions
            or if the last dimension is not 4.
    """
    if tensor.ndim != 4:
        raise ValueError(
            f"Tensor must have 4 dimensions (V, B, S, 4), got {tensor.ndim}."
        )
    if tensor.shape[-1] != 4:
        raise ValueError(
            f"Last dimension must be 4 (pitch, vel, start, dur), got {tensor.shape[-1]}."
        )


def validate_pitch_range(tensor: Tensor) -> None:
    """Warn if any active pitch values are outside MIDI range [0, 127]."""
    from tenseur.core.constants import PITCH, VEL

    active = tensor[..., VEL] > 0
    pitches = tensor[..., PITCH][active]
    if pitches.size > 0 and (np.any(pitches < 0) or np.any(pitches > 127)):
        warnings.warn("Some pitch values are outside MIDI range [0, 127].")


def validate_velocity_range(tensor: Tensor) -> None:
    """Warn if any active velocity values are outside normalized range (0, 1]."""
    from tenseur.core.constants import VEL

    active = tensor[..., VEL] > 0
    velocities = tensor[..., VEL][active]
    if velocities.size > 0 and np.any(velocities > 1.0):
        warnings.warn("Some velocity values are above 1.0 (normalized range is 0-1).")


def validate_midi_range(value: int, name: str = "value") -> None:
    """Validate that a value is within MIDI range [0, 127].

    Raises:
        ValueError: If the value is outside [0, 127].
    """
    if not 0 <= value <= 127:
        raise ValueError(f"{name} must be in range [0, 127], got {value}.")
