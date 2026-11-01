Features
========

A featurizer turns the inputs of one modality into model inputs.
A task holds at most one featurizer per modality, and applies the same featurizer to side ``a`` and side ``b`` of a pair.

1. ``fit`` reads the training inputs and their context. Only ``VocabularyText`` fits anything.
2. ``transform`` precomputes features once per dataset.
3. ``collate`` pads a batch and returns ``inputs`` and ``input_mask``.
4. ``model_inputs`` returns what the model needs from the fitted featurizer, such as the vocabulary size.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Featurizer
     - Modality
     - Model inputs
   * - ``VocabularyText``
     - Text
     - Token indices from a vocabulary fitted on the training texts, optionally with GloVe vectors
   * - ``TokenizedText``
     - Text
     - Token ids and attention masks from a pretrained tokenizer
   * - ``EncodedText``
     - Text
     - Token embeddings from a pretrained text model
   * - ``MFCCAudio``
     - Audio
     - MFCCs and spectral features of each clip
   * - ``EncodedAudio``
     - Audio
     - Frame embeddings from a pretrained audio model, optionally downsampled or averaged over frames

``EncodedText`` and ``EncodedAudio`` load their pretrained model in ``transform`` and release it in ``clear``, so the model does not stay in memory during training.

API
---

.. automodule:: mamkit.components.features
   :members:
