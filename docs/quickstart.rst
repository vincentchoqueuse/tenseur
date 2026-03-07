Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install -e ".[viz]"

Requires Python >= 3.10, NumPy, and optionally matplotlib for visualization.

The Tensor
----------

Music is a 4D NumPy array of shape ``(V, B, S, 4)``:

.. list-table::
   :header-rows: 1

   * - Axis
     - Meaning
     - Description
   * - V
     - Voices
     - Polyphonic layers
   * - B
     - Bars
     - Temporal macro-structure
   * - S
     - Steps
     - Rhythmic grid resolution
   * - 4
     - Properties
     - ``[pitch, velocity, start, duration]``

Minimal Example
---------------

.. code-block:: python

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
   key[0, :, :, 0] = np.arange(16) % 11 % 7
   key[0, :, :, 1] = euclidean(11, 16)
   Clip(key).linear_pitch(12/7, 52).project_pitch(SCALE).render(live)

   live.stop()

Dry Mode
--------

To work without an Ableton connection (visualization only):

.. code-block:: python

   live = Live(index=1, dry=True)
