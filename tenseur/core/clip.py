"""Clip dataclass — a named tensor with track/channel metadata."""

import math
import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from tenseur.core.constants import DUR, PITCH, START, VEL
from tenseur.core.types import Tensor
from tenseur.utils.validation import validate_tensor_shape


@dataclass
class Clip:
    """A musical clip wrapping a (V, B, S, 4) tensor.

    Attributes:
        tensor: Note data of shape (V, B, S, 4).
        track: Track number for rendering (1-based).
        name: Human-readable label.
        channel: MIDI channel (0-15).
        bar_offset: Bar offset for positioning in a timeline.
        bars: Number of bars for rendering. Computed automatically.
    """

    tensor: Tensor
    _track: int = 1
    name: str = ""
    channel: int = 0
    bar_offset: int = 0
    bars: float | None = None

    def __post_init__(self) -> None:
        validate_tensor_shape(self.tensor)
        if self.bars is None:
            self.bars = self.tensor.shape[1] + self.bar_offset
        if self._track < 1:
            raise ValueError(
                f"Track number must be >= 1, got {self._track}."
            )
        if not 0 <= self.channel <= 15:
            raise ValueError(
                f"MIDI channel must be in range [0, 15], got {self.channel}."
            )

    def track(self, n: int) -> "Clip":
        """Set the track number. Returns self for chaining."""
        self._track = n
        return self

    def linear_pitch(self, scale: float = 1.0, offset: float = 0) -> "Clip":
        """Affine transform on pitch: pitch = pitch * scale + offset.

        Returns self for chaining.
        """
        self.tensor[..., PITCH] *= scale
        self.tensor[..., PITCH] += offset
        return self

    def scale_pitch(self, factor: float) -> "Clip":
        """Multiply all pitches by a factor.

        Returns self for chaining.
        """
        self.tensor[..., PITCH] *= factor
        return self

    def transpose(self, semitones: float) -> "Clip":
        """Shift all pitches by a number of semitones.

        Returns self for chaining.
        """
        self.tensor[..., PITCH] += semitones
        return self

    def project_pitch(self, scale: "PitchSet") -> "Clip":
        """Quantize active pitches to the nearest note in a scale.

        Args:
            scale: Allowed MIDI pitch values.

        Returns:
            self (for chaining).
        """
        from tenseur.helpers.transforms import quantize

        active = self.tensor[..., VEL] > 0
        quantized = quantize(self.tensor[..., PITCH], scale)
        self.tensor[..., PITCH] = np.where(active, quantized, self.tensor[..., PITCH])
        return self

    def humanize(self, velocity: float = 0.04, timing: float = 0.05) -> "Clip":
        """Add humanization noise to velocity and timing.

        Args:
            velocity: Std-dev of noise added to velocity.
            timing: Std-dev of noise added to start times.

        Returns:
            self (for chaining).
        """
        from tenseur.utils.random import _rng
        active = self.tensor[..., VEL] > 0
        if velocity > 0:
            self.tensor[..., VEL] += _rng.normal(0, velocity, size=self.tensor[..., VEL].shape)
            self.tensor[..., VEL] = np.where(active, np.clip(self.tensor[..., VEL], 1e-6, 1.0), 0)
        if timing > 0:
            self.tensor[..., START] += _rng.normal(0, timing, size=self.tensor[..., START].shape) * active
        return self

    def quantize(self, grid: float = 1.0) -> "Clip":
        """Snap start times to the nearest grid point.

        Args:
            grid: Grid resolution in steps (1.0 = integer steps, 0.5 = half steps).

        Returns:
            self (for chaining).
        """
        self.tensor[..., START] = np.round(self.tensor[..., START] / grid) * grid
        return self

    def rgate(self, probability: float = 0.5) -> "Clip":
        """Randomly silence active notes.

        Each active note has a *probability* chance of passing through.
        Uses the global RNG (controlled via ``seed()``).

        Args:
            probability: Chance of keeping each note (0 = silence all, 1 = keep all).

        Returns:
            self (for chaining).
        """
        from tenseur.utils.random import _rng
        active = self.tensor[..., VEL] > 0
        kill = _rng.random(active.shape) > probability
        self.tensor[..., VEL] = np.where(active & kill, 0, self.tensor[..., VEL])
        return self

    def sparsify(self, priority: list[int] | np.ndarray | None = None) -> "Clip":
        """Keep only the highest-priority voice at each step.

        When multiple voices are active at the same (bar, step) position,
        only the one with the highest priority (earliest in the list) is kept.

        Args:
            priority: Voice indices in priority order (first = highest).
                Defaults to [0, 1, 2, ...].

        Returns:
            self (for chaining).
        """
        V, B, S, _ = self.tensor.shape
        if priority is None:
            priority = list(range(V))
        claimed = np.zeros((B, S), dtype=bool)
        for v in priority:
            active = self.tensor[v, :, :, VEL] > 0
            self.tensor[v, :, :, VEL] = np.where(claimed & active, 0, self.tensor[v, :, :, VEL])
            claimed |= active
        return self

    def speed(self, factor: float) -> "Clip":
        """Speed up start times and durations by factor. Returns self.

        Args:
            factor: Speed multiplier (2 = twice as fast, 0.5 = twice as slow).

        Returns:
            self (for chaining).
        """
        self.tensor[..., START] /= factor
        self.tensor[..., DUR] /= factor
        self.bars = self.bars / factor

        return self

    _VIEWS: ClassVar[dict[str, tuple[float, float]]] = {
        "step": (-45, 0),
        "bar": (0, -45),
        "top": (90, -90),
    }

    def show(self, view: float | str = 25, azim: float = -60) -> "Clip":
        """Visualize the tensor as 3D scatter and save to PNG.

        Args:
            view: Preset name ("step", "bar", "top") or elevation angle.
            azim: Azimuth angle in degrees (ignored when view is a string).

        Saves to `track_{track}_{name}.png` (or `track_{track}.png` if unnamed).
        Returns self for chaining.
        """
        import matplotlib.pyplot as plt
        from tenseur.viz.plot import scatter_3d

        if isinstance(view, str):
            if view not in self._VIEWS:
                raise ValueError(
                    f"Unknown view {view!r}. Available: {list(self._VIEWS.keys())}"
                )
            elev, azim = self._VIEWS[view]
        else:
            elev = view

        title = self.name or f"Track {self._track}"
        ax = scatter_3d(self.tensor, title=title)
        ax.view_init(elev=elev, azim=azim)

        render_dir = os.path.join(os.getcwd(), "render")
        os.makedirs(render_dir, exist_ok=True)

        filename = f"track_{self._track}.png"
        filepath = os.path.join(render_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="#1f1f1f")

        plt.close()

        return self

    def render(self, live: "tenseur.backends.ableton.Live") -> "Clip":
        """Push this clip to Ableton Live via a Live connection.

        Args:
            live: An open Live instance.

        Returns:
            self (for chaining).
        """
        live.push(self)
        return self
