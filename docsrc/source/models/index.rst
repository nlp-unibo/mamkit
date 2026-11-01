Models
======

MAMKit ships the models of the MAMKit paper as six families, each on its own page.
Every model is a ``Classifier``, an encoder followed by a head of linear layers, except ``Ensemble``, which mixes two classifiers.
A family name is also a tag of every task that uses it, so ``csa`` selects the CSA tasks of every corpus.

.. list-table::
   :header-rows: 1
   :widths: 18 22 20 22 18

   * - Model
     - Text encoding
     - Audio encoding
     - Fusion
     - Tags
   * - :doc:`bilstm`
     - GloVe and BiLSTM
     - MFCCs, or a pretrained audio model, and BiLSTM
     - Concatenation of the two vectors
     - ``bilstm``, ``bilstm-mfcc``, ``bilstm-transformer``
   * - :doc:`transformer`
     - BERT or RoBERTa
     - Pretrained audio model and BiLSTM
     - Concatenation of the two vectors
     - ``transformer``, ``mmtransformer``
   * - :doc:`audio-transformer`
     - None
     - Pretrained audio model and Transformer encoder
     - None
     - ``transformer-encoder``
   * - :doc:`csa`
     - BERT token embeddings
     - Pretrained audio model
     - Self-attention over the concatenated sequences
     - ``csa``
   * - :doc:`ensemble`
     - BERT token embeddings
     - Pretrained audio model and Transformer encoder
     - Learned average of the two class distributions
     - ``ensemble``
   * - :doc:`multa`
     - BERT token embeddings
     - Pretrained audio model
     - Cross-modal attention in both directions
     - ``multa``

The pretrained audio model is wav2vec 2.0 by default, and WavLM for the two MM-USED-fallacy AFD tasks that read pretrained audio features.
The HuBERT and WavLM variants of a task carry an extra tag, such as ``audio.model_card=facebook/hubert-base-ls960``.

A pair model applies one encoder to side ``a`` and to side ``b``, and its head reads the two vectors concatenated.
The M-Arg tasks register every model in this form.

.. toctree::
   :maxdepth: 1

   bilstm
   transformer
   audio-transformer
   csa
   ensemble
   multa

Encoders
--------

Every encoder reads a batch of samples and returns one vector per sample, whose width is its ``output_dim``.
A sample maps each modality to its fields: ``inputs`` holds the token indices, the token embeddings, or the audio frames, and ``input_mask`` marks the valid positions.

.. automodule:: mamkit.components.encoders
   :members:

.. automodule:: mamkit.components.models
   :members:
