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
    size: tuple[int, int, int] | list[int] = (16, 4, 16),
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


def rdrum_tensor(
    size: tuple[int, int, int] | list[int] = (16, 4, 16),
    prob: float = 0.05,
    duration: float = 1.0,
    kit: float = 0.0,
    steps_per_bar: int | None = None,
) -> Tensor:
    """Create a drum tensor with random velocity activation.

    Shortcut for::

        drums = drum_tensor(size, duration=duration, kit=kit)
        drums[..., VEL] = bernoulli(size, prob=prob)

    Args:
        size: (voices, bars, steps).
        prob: Probability of each note being active.
        duration: Note duration in steps.
        kit: Kit offset for drum_tensor.
        steps_per_bar: Steps per bar for start-time computation.

    Returns:
        Tensor of shape (V, B, S, 4) with stochastic velocity.
    """
    from tenseur.utils.random import bernoulli
    tensor = drum_tensor(size, duration=duration, kit=kit, steps_per_bar=steps_per_bar)
    tensor[..., VEL] = bernoulli(size, prob=prob)
    return tensor


def random_tensor(
    size: TensorSize = (1, 16, 16),
    prob: float = 0.1,
    scale: float = 3.0,
    duration: float = 16.0,
    dur_scale: float = 5.0,
    walk: bool = True,
    range: float | None = 7.0,
    root: bool = True,
    steps_per_bar: int | None = None,
) -> Tensor:
    """Create a tensor with random pitch and stochastic activation.

    Generates melodic material from noise. When ``walk=True``, pitches
    follow a Brownian motion (cumulative sum) for smooth contours.
    When ``range`` is set, pitches are wrapped with modulo to stay
    within an interval.

    Args:
        size: Tensor size (V, B, S) or shortcuts accepted by simple_tensor.
        prob: Probability of each note being active (bernoulli).
        scale: Standard deviation of the pitch noise.
        duration: Note duration in steps.
        dur_scale: Standard deviation added to duration (0 = fixed duration).
        walk: If True, apply cumsum for Brownian motion melody.
            If False, use raw Gaussian noise.
        range: If set, wrap pitches with modulo to stay in [0, range).
            Set to None for unbounded pitch.
        root: If True, force the first note of each voice to pitch 0
            (root) so the walk starts from a known point.
        steps_per_bar: Steps per bar for start-time computation.

    Returns:
        Tensor of shape (V, B, S, 4).

    Example::

        # Brownian piano, sparse, wrapped to 7 scale degrees
        noise = random_tensor(prob=0.1, scale=3, walk=True, range=7)
        Clip(noise).linear_pitch(12/7, ROOT).project_pitch(SCALE).render(live)

        # Dense white noise texture, no walk
        texture = random_tensor(prob=0.5, scale=5, walk=False, range=None)
    """
    from tenseur.utils.random import bernoulli, normal

    tensor = simple_tensor(size, duration=duration, steps_per_bar=steps_per_bar)
    V, B, S, _ = tensor.shape

    # Pitch generation
    pitch = scale * normal((V, B * S,))
    if walk:
        pitch = np.cumsum(pitch, axis=-1)
    pitch = pitch.reshape(V, B, S)
    if root:
        pitch -= pitch[:, 0:1, 0:1]
    if range is not None:
        pitch = pitch % range

    tensor[..., PITCH] = pitch

    # Duration variation
    if dur_scale > 0:
        tensor[..., DUR] = duration + dur_scale * normal((V, B, S))

    # Stochastic activation
    tensor[..., VEL] = bernoulli((V, B, S), prob=prob)

    return tensor


def drum_pattern(
    pattern: dict[int, list[int] | np.ndarray],
    bars: int = 4,
    steps: int = 16,
    voices: int | None = None,
    duration: float = 1.0,
    kit: float = 0.0,
    steps_per_bar: int | None = None,
) -> Tensor:
    """Create a drum tensor from a voice→steps mapping.

    Quick drum programming for live coding. Each key is a voice index,
    each value defines which steps are active (list of indices, slice-like
    string, or a boolean/float array).

    Args:
        pattern: {voice: steps} mapping. Values can be:
            - list of ints: step indices, e.g. [0, 4, 8, 12]
            - np.ndarray: velocity array of length ``steps``
              (boolean or float, broadcast over bars)
        bars: Number of bars.
        steps: Steps per bar.
        voices: Total voices (default: max voice index + 1).
        duration: Note duration in steps.
        kit: Kit offset for drum_tensor.
        steps_per_bar: Steps per bar for start-time computation.

    Returns:
        Tensor of shape (V, B, S, 4).

    Example::

        drums = drum_pattern({
            0: [0, 10],            # kick
            2: [4, 12],            # snare
            4: euclidean(7, 16),   # hat
            8: [5, 8, 11, 14],    # shaker
        })
    """
    if voices is None:
        voices = max(pattern.keys()) + 1

    tensor = drum_tensor(
        (voices, bars, steps),
        duration=duration,
        kit=kit,
        steps_per_bar=steps_per_bar,
    )

    for voice, hits in pattern.items():
        if isinstance(hits, list):
            tensor[voice, :, hits, VEL] = 1.0
        elif isinstance(hits, np.ndarray):
            tensor[voice, :, :, VEL] = hits
        else:
            tensor[voice, :, :, VEL] = hits

    return tensor
