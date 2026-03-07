"""Backend registry and render dispatcher."""

from typing import Any

import numpy as np

from tenseur.core.constants import PITCH, VEL

_BACKENDS: dict[str, type] = {}


def _ensure_registered() -> None:
    """Lazily register built-in backends."""
    if not _BACKENDS:
        from tenseur.backends.ableton import AbletonBackend
        from tenseur.backends.midi import MidiBackend
        from tenseur.backends.osc import OscBackend

        _BACKENDS["midi"] = MidiBackend
        _BACKENDS["osc"] = OscBackend
        _BACKENDS["ableton"] = AbletonBackend


def _log_render(clips: list, backend: str, index: int | None = None) -> None:
    """Print quick render stats."""
    idx_str = f" clip {index}" if index is not None else ""
    print(f"\n── render [{backend}]{idx_str} ──")
    total_notes = 0
    for clip in clips:
        t = clip.tensor
        active = t[..., VEL] > 0
        n = int(np.sum(active))
        total_notes += n
        V, B, S, _ = t.shape
        label = clip.name or f"track {clip._track}"
        pitch_str = ""
        if n > 0:
            pitches = t[..., PITCH][active]
            pitch_str = f"  pitch [{int(pitches.min())}–{int(pitches.max())}]"
        print(f"  track {clip._track:<3} {label:<16} ({V}v {B}b {S}s)  {n} notes  {clip.bars} bars{pitch_str}")
    print(f"  total: {total_notes} notes, {len(clips)} clips\n")


def render(clips: list, backend: str = "ableton", **kwargs: Any) -> Any:
    """Render clips using the named backend.

    Args:
        clips: List of Clip instances.
        backend: Backend name ('midi' or 'osc').
        **kwargs: Passed to the backend's render_clips method.

    Returns:
        Whatever the backend returns.

    Raises:
        ValueError: If the backend name is unknown.
    """
    from tenseur.core.clip import Clip

    if isinstance(clips, Clip):
        clips = [clips]
    _ensure_registered()
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Available: {list(_BACKENDS.keys())}"
        )
    _log_render(clips, backend, index=kwargs.get("index"))
    return _BACKENDS[backend]().render_clips(clips, **kwargs)
