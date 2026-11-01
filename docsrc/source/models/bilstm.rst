BiLSTM
======

The BiLSTM family encodes each modality with a stack of bidirectional LSTMs and concatenates the last hidden states.
It is the baseline of every corpus.

Method
------

1. ``TextBiLSTM`` embeds the token indices of ``VocabularyText``. The embedding starts from GloVe vectors when the featurizer names an ``embedding_model``.
2. ``AudioBiLSTM`` reads the frames of ``MFCCAudio``, or the frame embeddings of ``EncodedAudio``.
3. Each stack skips the padded positions that the input mask marks, and returns the concatenated last hidden states of both directions.
4. In ``text-audio`` mode, ``Concat`` joins the text vector and the audio vector before the head.

Registered tasks
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Tag
     - Input modes
     - Corpora and tasks
   * - ``bilstm``
     - text-only
     - Every corpus and task
   * - ``bilstm-mfcc``
     - audio-only, text-audio
     - Every corpus and task
   * - ``bilstm-transformer``
     - audio-only, text-audio
     - Every corpus and task except MM-USED-fallacy AFD

API
---

.. autoclass:: mamkit.components.encoders.TextBiLSTM
   :members:
   :no-index:

.. autoclass:: mamkit.components.encoders.AudioBiLSTM
   :members:
   :no-index:

.. autoclass:: mamkit.components.encoders.Concat
   :members:
   :no-index:
