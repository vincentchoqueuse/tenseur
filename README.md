# Tenseur

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://vincentchoqueuse.github.io/tenseur/)

Tensor-based musical composition with NumPy.

Tenseur is a Python library where music is composed in a **latent space of floats** — there are no notes, no chords, no rhythms until the moment of projection onto a tuning system. Composition is tensor manipulation. NumPy indexing is musical notation.

## Install

```bash
pip install -e ".[viz]"
```

Requires Python >= 3.10, NumPy, and optionally matplotlib for visualization.

## The Tensor

Music is a 4D NumPy array of shape `(V, B, S, 4)`:

| Axis | Meaning    | Description                          |
| ---- | ---------- | ------------------------------------ |
| V    | Voices     | Polyphonic layers                    |
| B    | Bars       | Temporal macro-structure             |
| S    | Steps      | Rhythmic grid resolution             |
| 4    | Properties | `[pitch, velocity, start, duration]` |

## Quick Start

```python
import numpy as np
from tenseur import Clip, Live, seed, drum_tensor, euclidean, simple_tensor, generate_scale

seed(42)
SCALE = generate_scale(7, root=52)
live = Live(index=1)

# Drums: 16 voices, 4 bars, 16 steps
drums = drum_tensor((16, 4, 16), duration=1)
drums[0, :, [0, 11], 1] = 1                    # kick
drums[2, :, [4, 12], 1] = 1                    # snare
drums[4, :, :, 1] = euclidean(7, 16)           # hat
Clip(drums).rgate(0.7).sparsify().track(4).render(live)

# Keys: abstract pitch space projected onto 7-EDO
key = simple_tensor((3, 2, 16))
key[0, :, :, 0] = np.arange(16) % 11 % 7      # contour from modular arithmetic
key[0, :, :, 1] = euclidean(11, 16)            # rhythm from Euclidean geometry
Clip(key).linear_pitch(12/7, 52).project_pitch(SCALE).render(live)

live.stop()
```

## Key Ideas

### Latent Space Composition

Composition happens in a continuous float space. Music appears only at `project_pitch(SCALE)`. The same tensor produces different music under different projections. The composition is scale-agnostic by construction.

```python
generate_scale(12)   # 12-TET chromatic
generate_scale(7)    # diatonic
generate_scale(19)   # microtonal 19-EDO
generate_scale(5)    # pentatonic
```

### Fourier Decomposition as Harmonic Language

Melodic and harmonic progressions are periodic signals in latent space:

```python
w = 2 * np.pi * (1/7) * np.arange(4)
V = 3*np.sin(w + 0.1) - 0.5*np.sin(3*w)       # harmonic trajectory
M = np.array([V, V+3.5+7, V+7])                # chord voicing
synth[..., 0, 0] = M                            # applied via broadcasting
```

Harmonic complexity is controlled by adding higher-order Fourier components. Any progression can be decomposed, manipulated, and reprojected.

### Rhythm as Geometry

```python
euclidean(7, 16)              # Bjorklund: 7 hits in 16 steps
euclidean(9, 16, rotation=2)  # phase-shifted
bernoulli((16, 4, 16), 0.3)  # stochastic activation
np.roll(M, 2, axis=1)        # canon: temporal displacement
```

### Stochastic Primitives

```python
normal((16, 4, 16), scale=0.8)    # Gaussian noise
bernoulli((16, 4, 16), prob=0.3)  # binary activation
np.cumsum(normal((8,), scale=3))  # Brownian motion
```

Reproducibility via `seed(42)` — a script + a seed = a versioned composition.

### Fluent Pipeline

```python
Clip(drums).rgate(0.7).sparsify().quantize(1).humanize().track(4).show().render(live)
```

| Method                        | Description                                             |
| ----------------------------- | ------------------------------------------------------- |
| `rgate(p)`                    | Stochastic gate — each note survives with probability p |
| `sparsify()`                  | Monophonic reduction — highest-priority voice per step  |
| `quantize(grid)`              | Snap start times to grid                                |
| `humanize(vel, timing)`       | Add controlled noise                                    |
| `linear_pitch(scale, offset)` | Affine pitch transform                                  |
| `project_pitch(scale)`        | Quantize onto EDO tuning system                         |
| `show()`                      | 3D visualization                                        |
| `render(live)`                | Push to Ableton via OSC                                 |

## Backends

- **Ableton Live** via OSC (requires [AbletonOSC](https://github.com/ideoforms/AbletonOSC))
- **Dry mode**: `Live(index=1, dry=True)` — no connection, render skipped, visualization still works

## Architecture

```
tenseur/
├── core/         # Tensor type, Clip, constants (PITCH=0, VEL=1, START=2, DUR=3)
├── helpers/      # Creation: simple_tensor, drum_tensor, euclidean, scatter
│                 # Transforms: upsample, swing, quantize, sine
│                 # Scales: generate_scale (EDO)
├── viz/          # 3D matplotlib scatter plots
├── backends/     # Ableton Live (OSC), MIDI
└── utils/        # Global RNG (normal, bernoulli, seed), MIDI utilities
```

## Examples

The `examples/` directory contains complete compositions written for live performance. Each file is a self-contained score.

## Origin

Developed at [ENIB](https://www.enib.fr/) — Ecole Nationale d'Ingenieurs de Brest.
Built by a signal processing researcher who cannot read a score.

_Vincent Choqueuse — Associate Professor, ENIB_
_PhD/HDR Signal Processing — Lab-STICC CNRS UMR 6285_

## License

MIT
