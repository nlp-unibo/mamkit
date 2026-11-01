MulTA
=====

MulTA fuses the two modalities with cross-modal attention in both directions.
Text queries attend to the audio frames, and audio queries attend to the text tokens.

Method
------

1. ``EncodedText`` computes the token embeddings of a pretrained text model, and ``EncodedAudio`` computes the frame embeddings.
2. ``MulTA`` adds positional encodings to both sequences.
3. A stack of cross-attention blocks updates the text sequence from the audio, and a second stack updates the audio sequence from the text. Each block ignores the padded positions of the sequence it attends to.
4. Each output is averaged over its valid positions, and the two vectors are concatenated.

Registered tasks
----------------

The ``multa`` tag selects the ``text-audio`` tasks of every corpus and task except MM-USED-fallacy AFD.

API
---

.. autoclass:: mamkit.components.encoders.MulTA
   :members:
   :no-index:
