CSA
===

CSA applies self-attention to one sequence that holds the text tokens followed by the audio frames.
Each token can therefore attend to every frame, and each frame to every token.

Method
------

1. ``EncodedText`` computes the token embeddings of a pretrained text model once per text, and ``EncodedAudio`` computes the frame embeddings.
2. ``CSA`` applies text dropout to the token embeddings and adds positional encodings to the audio frames.
3. A Transformer encoder runs over the concatenated sequence, with the text mask and the audio mask concatenated in the same order.
4. Audio dropout is applied to the encoder output, which is then averaged over the valid positions of both modalities.

Registered tasks
----------------

The ``csa`` tag selects the ``text-audio`` tasks of every corpus and task except MM-USED-fallacy AFD.

API
---

.. autoclass:: mamkit.components.encoders.CSA
   :members:
   :no-index:
