"""Pure transformation functions for tensors. Never mutate input."""

import numpy as np

from tenseur.core.constants import DUR, PITCH, START, VEL
from tenseur.utils.random import bernoulli as _bernoulli
from tenseur.core.types import PitchSet, Tensor


def sine(t: np.ndarray, freq: float = 1.0, amp: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Sine wave over time values.

    Args:
        t: Time array (typically start times from a tensor slice).
        freq: Frequency in cycles per step.
        amp: Amplitude.
        offset: DC offset.

    Returns:
        offset + amp * sin(2π * freq * t)
    """
    return offset + amp * np.sin(2 * np.pi * freq * t)


def swing(tensor: Tensor, amount: float = 0.5) -> Tensor:
    """Apply swing by shifting odd-indexed steps' start times.

    Args:
        tensor: Input tensor of shape (V, B, S, 4).
        amount: How much to shift odd steps (in fractional steps).

    Returns:
        New tensor with swing applied.
    """
    out = tensor.copy()
    # Odd-indexed steps along the S axis (axis=2)
    out[:, :, 1::2, START] += amount
    return out



def quantize(values: np.ndarray, allowed: list[int] | list[float] | np.ndarray) -> np.ndarray:
    """Snap every value to the nearest in an allowed set.

    Works on any array shape — use it on a slice of a tensor or standalone.

    Args:
        values: Array of any shape.
        allowed: Sorted list/array of allowed values.

    Returns:
        New array with each value snapped to the nearest allowed value.
    """
    ps = np.array(sorted(allowed), dtype=np.float64)
    if ps.size == 0:
        return values.copy()

    idx = np.searchsorted(ps, values, side="left")
    idx = np.clip(idx, 0, len(ps) - 1)

    idx_left = np.clip(idx - 1, 0, len(ps) - 1)
    dist_right = np.abs(values - ps[idx])
    dist_left = np.abs(values - ps[idx_left])

    best_idx = np.where(dist_left <= dist_right, idx_left, idx)
    return ps[best_idx]


def quantize_pitch(tensor: Tensor, pitch_set: PitchSet) -> Tensor:
    """Snap active pitches to the nearest value in a pitch set.

    Convenience wrapper around quantize() for full (V, B, S, 4) tensors.
    Only affects active notes (vel > 0).

    Args:
        tensor: Input tensor of shape (V, B, S, 4).
        pitch_set: Sorted list of allowed MIDI pitch values.

    Returns:
        New tensor with active pitches snapped to the nearest allowed value.
    """
    out = tensor.copy()
    active = out[..., VEL] > 0
    quantized = quantize(out[..., PITCH], pitch_set)
    out[..., PITCH] = np.where(active, quantized, out[..., PITCH])
    return out


def upsample(tensor: Tensor, factor: int = 2) -> Tensor:
    """Upsample by zero-stuffing along the step axis.

    Inserts (factor - 1) silent steps after each original step.
    A factor of 2 doubles S, a factor of 3 triples it, etc.

    Args:
        tensor: Input tensor of shape (V, B, S, 4).
        factor: Upsampling factor (must be >= 1).

    Returns:
        New tensor of shape (V, B, S * factor, 4) with zeros between original steps.
    """
    _, B, S, _ = tensor.shape
    S_new = S * factor
    out = np.zeros((*tensor.shape[:2], S_new, 4), dtype=tensor.dtype)
    out[:, :, ::factor, :] = tensor
    # Propagate pitch & duration to inserted steps so activating them works
    for k in range(1, factor):
        out[:, :, k::factor, PITCH] = tensor[..., PITCH]
        out[:, :, k::factor, DUR] = tensor[..., DUR]
    # Recalculate start times: bar * spb_new + step
    spb = int(tensor[0, 1, 0, START] - tensor[0, 0, 0, START]) if B > 1 else S
    spb_new = spb * factor
    bars = np.arange(B, dtype=np.float64).reshape(1, B, 1)
    steps = np.arange(S_new, dtype=np.float64).reshape(1, 1, S_new)
    out[..., START] = bars * spb_new + steps
    return out


def crossfade(a: Tensor, b: Tensor, prob: float = 0.5) -> Tensor:
    """Pick from tensor a or b at each step based on probability.

    At each (v, b, s) position, all 4 properties come from either a or b.
    Uses the global RNG for reproducibility.

    Args:
        a: First tensor of shape (V, B, S, 4).
        b: Second tensor of shape (V, B, S, 4) — must match a's shape.
        prob: Probability of picking from b (0 = all a, 1 = all b).

    Returns:
        New tensor mixing a and b step-wise.
    """
    mask = _bernoulli(a.shape[:3], prob=prob)
    return np.where(mask[..., None], b, a)


