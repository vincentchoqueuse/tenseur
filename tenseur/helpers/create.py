"""Tensor creation helpers."""

import numpy as np

from tenseur.core.constants import DEFAULT_STEPS_PER_BAR, DUR, PITCH, START, VEL
from tenseur.core.types import Tensor, TensorSize


def simple_tensor(
    size: TensorSize,
    pitch: float = 0.0,
    velocity: float = 0.0,
    duration: float = 1.0,
    steps_per_bar: int | None = None,
) -> Tensor:
    """Create a tensor filled with uniform note values.

    Args:
        size: Shape specification:
            - int S -> (1, 1, S)
            - (V, S) -> (V, 1, S)
            - (V, B, S) -> (V, B, S)
        pitch: MIDI pitch value (0-127).
        velocity: Normalized velocity (0-1). 0 means inactive.
        duration: Note duration in steps.
        steps_per_bar: Steps per bar for start-time computation.
            Defaults to S (the last dimension).

    Returns:
        Tensor of shape (V, B, S, 4) with dtype float64.
    """
    if isinstance(size, int):
        V, B, S = 1, 1, size
    elif len(size) == 2:
        V, S = size
        B = 1
    else:
        V, B, S = size

    if steps_per_bar is None:
        steps_per_bar = S

    tensor = np.zeros((V, B, S, 4), dtype=np.float64)
    tensor[..., PITCH] = pitch
    tensor[..., VEL] = velocity
    tensor[..., DUR] = duration

    # Auto-compute start times via broadcasting:
    # start = bar_index * steps_per_bar + step_index
    bars = np.arange(B, dtype=np.float64)
    steps = np.arange(S, dtype=np.float64)
    start_times = bars.reshape(B, 1) * steps_per_bar + steps.reshape(1, S)
    tensor[..., START] = start_times  # broadcasts over V

    return tensor


def euclidean(pulses: int, steps: int | tuple[int, ...], rotation: int = 0) -> np.ndarray:
    """Generate a Euclidean rhythm pattern.

    Distributes *pulses* hits as evenly as possible across *steps* slots
    by spacing ideal positions linearly and projecting onto the grid.

    Args:
        pulses: Number of active hits.
        steps: Number of steps (int), or a shape tuple.
            - int S -> (S,)
            - (S,) -> (S,)
            - (B, S) -> (B, S) pattern broadcast across rows
            - (V, B, S) -> (V, B, S) pattern broadcast across leading dims
        rotation: Rotate the pattern to the right by this many steps.

    Returns:
        Boolean numpy array of the given shape.
    """
    if isinstance(steps, int):
        shape = (steps,)
    else:
        shape = steps
    S = shape[-1]
    if pulses > S:
        raise ValueError(
            f"pulses ({pulses}) must be <= steps ({S})."
        )
    row = np.zeros(S, dtype=bool)
    if pulses > 0:
        ideal = np.linspace(0, S, pulses, endpoint=False)
        hits = np.round(ideal).astype(int) % S
        row[hits] = True
    if rotation:
        row = np.roll(row, rotation)
    if len(shape) == 1:
        return row
    return np.broadcast_to(row, shape).copy()


def scatter(pulses: int, steps: int | tuple[int, ...]) -> np.ndarray:
    """Place *pulses* hits randomly across *steps* slots.

    Uses the global RNG (controlled via ``seed()``).

    Args:
        pulses: Number of active hits per row.
        steps: Number of steps (int), or a shape tuple.
            - int S -> (S,)
            - (S,) -> (S,)
            - (B, S) -> (B, S) with independent draws per row
            - (V, B, S) -> (V, B, S) with independent draws per row

    Returns:
        Boolean numpy array of the given shape.
    """
    if isinstance(steps, int):
        steps = (steps,)
    S = steps[-1]
    if pulses > S:
        raise ValueError(
            f"pulses ({pulses}) must be <= steps ({S})."
        )
    from tenseur.utils.random import _rng
    leading = steps[:-1]
    if not leading:
        idx = _rng.choice(S, size=pulses, replace=False)
        pattern = np.zeros(S, dtype=bool)
        pattern[idx] = True
        return pattern
    n_rows = int(np.prod(leading))
    pattern = np.zeros((n_rows, S), dtype=bool)
    for i in range(n_rows):
        idx = _rng.choice(S, size=pulses, replace=False)
        pattern[i, idx] = True
    return pattern.reshape(steps)



def drum_tensor(
    size: tuple[int, int, int] | list[int] = (16, 1, 16),
    pitch: float = 4.0,
    velocity: float = 0.0,
    duration: float = 0.5,
    kit: float = 0.0,
    steps_per_bar: int | None = None,
) -> Tensor:
    """Create a drum tensor with one voice per pitch.

    Each voice gets a sequential pitch starting from pitch:
    voice 0 = pitch, voice 1 = pitch+1, etc.

    Args:
        size: (voices, bars, steps).
        pitch: Base MIDI pitch for voice 0 (default 36 = C2/kick).
        velocity: Normalized velocity (0-1). 0 means inactive.
        duration: Note duration in steps.
        steps_per_bar: Steps per bar for start-time computation.
            Defaults to S (the last dimension).

    Returns:
        Tensor of shape (V, B, S, 4).
    """
    V, B, S = size

    if steps_per_bar is None:
        steps_per_bar = S

    tensor = np.zeros((V, B, S, 4), dtype=np.float64)

    # Sequential pitches per voice, offset by kit * V
    for v in range(V):
        tensor[v, ..., PITCH] = pitch + v + kit * V

    tensor[..., VEL] = velocity
    tensor[..., DUR] = duration

    # Start times
    bar_idx = np.arange(B, dtype=np.float64)
    step_idx = np.arange(S, dtype=np.float64)
    start_times = bar_idx.reshape(B, 1) * steps_per_bar + step_idx.reshape(1, S)
    tensor[..., START] = start_times

    return tensor
