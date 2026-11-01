Audio Transformer
=================

The audio Transformer encodes the frame embeddings of a pretrained audio model with a Transformer encoder.
It is the audio-only counterpart of CSA and Ensemble.

Method
------

1. ``EncodedAudio`` computes frame embeddings with the pretrained audio model.
2. ``AudioTransformer`` adds sinusoidal positional encodings and runs a Transformer encoder that ignores padded frames.
3. A residual connection and a layer normalization follow the encoder, and the output is averaged over the valid frames.

Registered tasks
----------------

The ``transformer-encoder`` tag selects the ``audio-only`` tasks of every corpus and task.

API
---

.. autoclass:: mamkit.components.encoders.AudioTransformer
   :members:
   :no-index:
