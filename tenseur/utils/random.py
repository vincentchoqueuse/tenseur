"""Global RNG for reproducible randomness."""

import numpy as np

_rng: np.random.Generator = np.random.default_rng()


def seed(s: int | None = None) -> None:
    """Reset the global RNG with a new seed."""
    global _rng
    _rng = np.random.default_rng(s)


def randn(*shape: int) -> np.ndarray:
    """Draw from the global RNG. Shortcut for standard_normal."""
    return _rng.standard_normal(shape)


def normal(size: tuple[int, ...] = (16, 4, 16), scale: float = 1.0) -> np.ndarray:
    """Draw normal noise with given shape and scale. Uses the global RNG."""
    return scale * _rng.standard_normal(size)


def bernoulli(size: tuple[int, ...] = (16, 4, 16), prob: float = 0.5) -> np.ndarray:
    """Draw 0/1 values with given probability. Uses the global RNG."""
    return (_rng.random(size) < prob).astype(np.float64)


def shuffle(arr: np.ndarray) -> np.ndarray:
    """Return a shuffled copy of the array (uses the global RNG)."""
    return _rng.permutation(arr)
