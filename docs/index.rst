Tenseur
=======

Tensor-based musical composition with NumPy.

Tenseur is a Python library where music is composed in a **latent space of floats**.
There are no notes, no chords, no rhythms until the moment of projection onto a tuning system.
Composition is tensor manipulation. NumPy indexing is musical notation.

.. code-block:: python

   import numpy as np
   from tenseur import Clip, Live, seed, drum_tensor, euclidean, simple_tensor, generate_scale

   seed(42)
   SCALE = generate_scale(7, root=52)
   live = Live(index=1)

   drums = drum_tensor((16, 4, 16), duration=1)
   drums[0, :, [0, 11], 1] = 1
   drums[2, :, [4, 12], 1] = 1
   drums[4, :, :, 1] = euclidean(7, 16)
   Clip(drums).rgate(0.7).sparsify().track(4).render(live)

   live.stop()

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   concepts
   api
