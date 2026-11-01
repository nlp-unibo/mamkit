Transformer and MM-Transformer
==============================

The text Transformer trains a classification head over a pretrained text model, BERT or RoBERTa.
MM-Transformer adds an audio BiLSTM and concatenates the two vectors.
The MAMKit paper calls the multimodal form MM-BERT or MM-RoBERTa, after its text model.

Method
------

1. ``TokenizedText`` tokenizes each text with the tokenizer of the model card.
2. ``TextTransformer`` runs the pretrained model and averages its last hidden states over the valid tokens. The pretrained weights stay frozen unless ``is_transformer_trainable`` is set.
3. In ``text-audio`` mode, ``AudioBiLSTM`` encodes the frame embeddings of ``EncodedAudio``, and ``Concat`` joins the two vectors.

The text model is a tag of the task, ``bert`` or ``roberta``.
The tokenizer and the encoder of a task always name the same card.

Registered tasks
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Tag
     - Input modes
     - Corpora and tasks
   * - ``transformer``
     - text-only
     - Every corpus and task
   * - ``mmtransformer``
     - text-audio
     - Every corpus and task

API
---

.. autoclass:: mamkit.components.encoders.TextTransformer
   :members:
   :no-index:
