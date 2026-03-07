API Reference
=============

Clip
----

.. autoclass:: tenseur.core.clip.Clip
   :members:
   :exclude-members: tensor, name, channel, bar_offset, bars, track

Tensor Creation
---------------

.. autofunction:: tenseur.helpers.create.simple_tensor

.. autofunction:: tenseur.helpers.create.drum_tensor

.. autofunction:: tenseur.helpers.create.euclidean

.. autofunction:: tenseur.helpers.create.scatter

Scales
------

.. autofunction:: tenseur.helpers.scales.generate_scale

Transforms
----------

.. autofunction:: tenseur.helpers.transforms.sine

.. autofunction:: tenseur.helpers.transforms.swing

.. autofunction:: tenseur.helpers.transforms.upsample

.. autofunction:: tenseur.helpers.transforms.quantize

.. autofunction:: tenseur.helpers.transforms.quantize_pitch

Random
------

.. autofunction:: tenseur.utils.random.seed

.. autofunction:: tenseur.utils.random.normal

.. autofunction:: tenseur.utils.random.bernoulli

.. autofunction:: tenseur.utils.random.shuffle

Live Backend
------------

.. autoclass:: tenseur.backends.ableton.Live
   :members:
   :undoc-members:
