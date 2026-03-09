Concepts
========

From Score to Tensor
--------------------

Traditional music production is **event-based**. The piano roll is a grid
where notes are placed one by one. Copy-paste creates repetition. Manual
editing creates variation. The workflow scales linearly with complexity:
more notes, more clicks.

Tenseur is **structure-based**. A composition is a 4D NumPy array of shape
``(V, B, S, 4)`` — voices, bars, steps, and four properties per note:
``[pitch, velocity, start, duration]``. The tensor is the score, the MIDI,
and the piano roll simultaneously.

The difference is not cosmetic. It changes what operations are natural:

.. list-table::
   :header-rows: 1

   * - Operation
     - Piano roll
     - Tensor
   * - Arpeggio
     - Place notes one by one
     - ``np.arange(16) % 3``
   * - Canon
     - Duplicate, shift, adjust
     - ``np.roll(M, 2, axis=1)``
   * - Harmonic progression
     - Draw each chord
     - ``np.sin(w) + 0.5*np.sin(2*w)``
   * - Stochastic variation
     - Not possible
     - ``bernoulli((16, 4, 16), 0.3)``
   * - Repetition with variation
     - Copy-paste, edit
     - Broadcasting

.. code-block:: python

   # A complete arpegiated synth part in 4 lines
   synth = simple_tensor((3, 4, 16), duration=8)
   synth[..., 0] = M[..., None]                          # harmonic content
   synth[..., 1] = (gate == np.arange(3)[:, None, None]) # arpeggio pattern
   Clip(synth).linear_pitch(12/7, ROOT).project_pitch(SCALE).render(live)

The composer works at the level of structure, not events.

Latent Space Composition
------------------------

The central idea: **music is composed before it exists as music.**

Every tensor is a structure of floats. There are no notes, no chords, no rhythms yet.
The composer navigates a continuous mathematical space.
Music appears only at ``project_pitch(SCALE)`` — not before.

This separates:

- **Pitch logic** — abstract float arithmetic, Fourier decomposition, modular operations
- **Pitch materialization** — projection onto any Equal Division of the Octave (EDO)

The same tensor produces different music under different projections.

.. code-block:: python

   generate_scale(12)   # 12-TET chromatic
   generate_scale(7)    # diatonic
   generate_scale(19)   # microtonal 19-EDO
   generate_scale(5)    # pentatonic

Fourier Decomposition as Harmonic Language
------------------------------------------

Melodic and harmonic progressions are treated as periodic signals.
Any progression can be synthesized as a sum of sinusoids:

.. code-block:: python

   w = 2 * np.pi * (1/7) * np.arange(4)
   V = 3*np.sin(w + 0.1) - 0.5*np.sin(3*w)
   M = np.array([V, V+3.5+7, V+7])
   synth[..., 0, 0] = M

This opens a theory of harmony in latent space:

- A chord progression is a trajectory, not a sequence of events
- Harmonic complexity is controlled by adding higher-order Fourier components
- Interpolation between two progressions produces new harmonies

Rhythm as Geometry
------------------

Rhythmic patterns are generated geometrically, not programmed step by step:

.. code-block:: python

   euclidean(7, 16)              # Bjorklund: 7 hits in 16 steps
   euclidean(9, 16, rotation=2)  # phase-shifted
   bernoulli((16, 4, 16), 0.3)  # stochastic activation
   np.roll(M, 2, axis=1)        # canon: temporal displacement

``np.roll`` on the bar axis produces exact canonic structures — rigorous counterpoint
in one NumPy call.

Arpeggios via Modular Indexing
------------------------------

Arpeggios emerge naturally from modular arithmetic on the step axis.
A gate vector distributes voices across time:

.. code-block:: python

   gate = np.arange(16) % 3  # [0, 1, 2, 0, 1, 2, ...]

   synth = simple_tensor((3, 4, 16), duration=8)
   synth[..., 0] = M[..., None]
   synth[0, :, :, 1] = (gate == 0)
   synth[1, :, :, 1] = (gate == 1)
   synth[2, :, :, 1] = (gate == 2)

Voice 0 plays at steps 0, 3, 6, 9… — voice 1 at 1, 4, 7, 10… — voice 2 at 2, 5, 8, 11…
The chord defined in ``M`` is spread across time as an arpeggio.

Changing the modulus changes the arpeggio speed. Compound modular chains
create polyrhythmic arpeggios:

.. code-block:: python

   gate = np.arange(16) % 5          # cycle of 5
   gate = np.arange(16) % 13 % 11   # nested modulo — irregular groupings

Stochastic Primitives
---------------------

Three stochastic primitives produce fundamentally different musical textures:

.. code-block:: python

   # Flat Gaussian — pure randomness
   V = normal((8,), scale=3)

   # Brownian motion — randomness with memory
   V = np.cumsum(normal((8,), scale=3))

   # Deterministic sinusoidal — pure periodic form
   V = 4*np.sin(w + 0.1) - 0.5*np.sin(2*w)

Reproducibility via global seeded RNG:

.. code-block:: python

   seed(42)

One integer determines an entire stochastic universe.
A script + a seed = a versioned, reproducible composition.

Velocity as Signal
------------------

Dynamic expression is a mathematical function, not manual automation:

.. code-block:: python

   synth[..., VEL] = 0.5 + 0.25*np.sin(2*np.pi*(1/(4*16))*synth[..., START])

The instrument breathes over 4 bars. Human dynamics from signal processing primitives.

Fluent Pipeline
---------------

The ``Clip`` class provides a chainable API:

.. code-block:: python

   Clip(drums).rgate(0.7).sparsify().quantize(1).humanize().track(4).show().render(live)

Each method is a stage in a signal processing chain:

.. list-table::
   :header-rows: 1

   * - Method
     - Description
   * - ``rgate(p)``
     - Stochastic gate: each note survives with probability p
   * - ``sparsify()``
     - Monophonic reduction: highest-priority voice per step
   * - ``quantize(grid)``
     - Snap start times to grid
   * - ``humanize(vel, timing)``
     - Add controlled noise to velocity and timing
   * - ``linear_pitch(scale, offset)``
     - Affine pitch transform
   * - ``project_pitch(scale)``
     - Quantize onto EDO tuning system
   * - ``show()``
     - 3D visualization
   * - ``render(live)``
     - Push to Ableton via OSC
