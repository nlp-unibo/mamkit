Ensemble
========

Ensemble trains a text classifier and an audio classifier together and averages their class distributions.
A learned coefficient sets how much each classifier counts.

Method
------

1. The text classifier averages the token embeddings of ``EncodedText`` with ``TextPool`` and applies its head.
2. The audio classifier encodes the frame embeddings of ``EncodedAudio`` with ``AudioTransformer`` and applies its head.
3. A learned weight ``w`` gives the text coefficient ``c = l + (u - l) (tanh(w) + 1) / 2``, where ``l`` and ``u`` are ``lower_bound`` and ``upper_bound``.
4. The model returns the logarithm of ``c`` times the text distribution plus ``1 - c`` times the audio distribution.

At the start of training, ``w`` is 0, so the coefficient is halfway between the two bounds.

Registered tasks
----------------

The ``ensemble`` tag selects the ``text-audio`` tasks of every corpus and task except MM-USED-fallacy AFD.

API
---

.. autoclass:: mamkit.components.models.Ensemble
   :members:
   :no-index:

.. autoclass:: mamkit.components.encoders.TextPool
   :members:
   :no-index:
