"""Visualization functions for tensors (requires matplotlib).

Themed to match VS Code Dark Modern exactly.
"""

from typing import Any

import numpy as np

from tenseur.core.constants import DUR, PITCH, START, VEL
from tenseur.core.types import Tensor

# ── VS Code Dark+ palette ────────────────────────────────────────────
_DARK_PLUS = {
    "bg":          "#1f1f1f",
    "panel_bg":    "#1e1e1e",
    "grid":        "#333333",
    "fg":          "#d4d4d4",
    "fg_dim":      "#858585",
    "accent":      "#007acc",
    "blue":        "#569cd6",
    "cyan":        "#9cdcfe",
    "green":       "#6a9955",
    "light_green": "#b5cea8",
    "yellow":      "#dcdcaa",
    "orange":      "#ce9178",
    "selection":   "#264f78",
}

# Voice colors cycle — pulled from Dark+ token colors
_VOICE_COLORS = [
    _DARK_PLUS["blue"],
    _DARK_PLUS["orange"],
    _DARK_PLUS["light_green"],
    _DARK_PLUS["cyan"],
    _DARK_PLUS["yellow"],
    _DARK_PLUS["green"],
]


def _apply_dark_plus(ax: Any, is_3d: bool = False) -> None:
    """Apply VS Code Dark+ styling to an axes."""
    fig = ax.get_figure()
    fig.patch.set_facecolor(_DARK_PLUS["bg"])
    ax.set_facecolor(_DARK_PLUS["panel_bg"])

    ax.tick_params(colors=_DARK_PLUS["fg_dim"], labelsize=8)
    ax.xaxis.label.set_color(_DARK_PLUS["fg"])
    ax.yaxis.label.set_color(_DARK_PLUS["fg"])
    ax.title.set_color(_DARK_PLUS["fg"])
    ax.title.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_color(_DARK_PLUS["grid"])

    ax.xaxis.set_tick_params(color=_DARK_PLUS["grid"])
    ax.yaxis.set_tick_params(color=_DARK_PLUS["grid"])

    if is_3d:
        ax.zaxis.label.set_color(_DARK_PLUS["fg"])
        ax.zaxis.set_tick_params(colors=_DARK_PLUS["fg_dim"], labelsize=8)
        ax.xaxis.pane.set_facecolor(_DARK_PLUS["panel_bg"])
        ax.yaxis.pane.set_facecolor(_DARK_PLUS["panel_bg"])
        ax.zaxis.pane.set_facecolor(_DARK_PLUS["panel_bg"])
        ax.xaxis.pane.set_edgecolor(_DARK_PLUS["grid"])
        ax.yaxis.pane.set_edgecolor(_DARK_PLUS["grid"])
        ax.zaxis.pane.set_edgecolor(_DARK_PLUS["grid"])
        ax.xaxis._axinfo["grid"]["color"] = _DARK_PLUS["grid"]
        ax.yaxis._axinfo["grid"]["color"] = _DARK_PLUS["grid"]
        ax.zaxis._axinfo["grid"]["color"] = _DARK_PLUS["grid"]


def piano_roll(
    tensor: Tensor,
    voice: int = 0,
    title: str = "Piano Roll",
    ax: Any = None,
) -> Any:
    """Draw a piano-roll view of a single voice with Dark+ theme.

    Horizontal bars at each pitch, colored by velocity.

    Args:
        tensor: Shape (V, B, S, 4).
        voice: Which voice index to display.
        title: Plot title.
        ax: Optional matplotlib Axes to draw on.

    Returns:
        The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))

    # Dark+ velocity colormap: dim grid color → accent blue
    cmap = LinearSegmentedColormap.from_list(
        "dark_plus_vel",
        [_DARK_PLUS["grid"], _DARK_PLUS["accent"], _DARK_PLUS["blue"]],
    )

    data = tensor[voice]  # (B, S, 4)
    B, S, _ = data.shape

    rects = []
    colors = []

    for b in range(B):
        for s in range(S):
            note = data[b, s]
            vel = note[VEL]
            if vel <= 0:
                continue
            pitch = note[PITCH]
            start = note[START]
            dur = note[DUR]
            rects.append(Rectangle((start, pitch - 0.4), dur, 0.8))
            colors.append(vel)

    if rects:
        pc = PatchCollection(rects, cmap=cmap, alpha=0.9)
        pc.set_array(np.array(colors))
        pc.set_clim(0, 1)
        ax.add_collection(pc)
        ax.autoscale_view()
        cbar = plt.colorbar(pc, ax=ax, label="Velocity", pad=0.02)
        cbar.ax.yaxis.set_tick_params(color=_DARK_PLUS["fg_dim"])
        cbar.ax.yaxis.label.set_color(_DARK_PLUS["fg"])
        cbar.outline.set_edgecolor(_DARK_PLUS["grid"])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_DARK_PLUS["fg_dim"])

    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Pitch (MIDI)")
    ax.set_title(title)
    _apply_dark_plus(ax)
    return ax


def scatter_3d(
    tensor: Tensor,
    title: str = "Tensor",
    ax: Any = None,
) -> Any:
    """3D scatter: X=bar, Y=step, Z=pitch. Size=velocity, alpha=duration.

    Args:
        tensor: Shape (V, B, S, 4).
        title: Plot title.
        ax: Optional matplotlib 3D Axes.

    Returns:
        The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

    V, B, S, _ = tensor.shape

    for v in range(V):
        color = _VOICE_COLORS[v % len(_VOICE_COLORS)]
        has_notes = False

        for b in range(B):
            for s in range(S):
                note = tensor[v, b, s]
                if note[VEL] <= 0:
                    continue
                has_notes = True
                pitch = note[PITCH]
                dur = note[DUR]
                vel = min(float(note[VEL]), 1.0)

                # Line along step axis to show duration
                step_start = float(s)
                step_end = float(s + dur)
                alpha = 0.5 + 0.5 * vel
                lw = 1.5 + vel * 4

                ax.plot(
                    [b, b],
                    [step_start, step_end],
                    [pitch, pitch],
                    color=color,
                    alpha=alpha,
                    linewidth=lw,
                    solid_capstyle="round",
                )

        if has_notes:
            ax.plot([], [], [], color=color, linewidth=3, label=f"Voice {v}")

    ax.set_xlabel("Bar")
    ax.set_ylabel("Step")
    ax.set_zlabel("Pitch (MIDI)")
    ax.set_title(title)

    legend = ax.legend(
        loc="upper left",
        fontsize=8,
        facecolor=_DARK_PLUS["panel_bg"],
        edgecolor=_DARK_PLUS["grid"],
        labelcolor=_DARK_PLUS["fg_dim"],
    )

    _apply_dark_plus(ax, is_3d=True)
    return ax
