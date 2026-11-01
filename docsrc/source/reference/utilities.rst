Utilities
=========

The components call these layers and helpers.
They are documented for a component of your own that needs them.

1. ``mamkit.modules.rnn`` holds ``LSTMStack``, the BiLSTM stack of the BiLSTM encoders.
2. ``mamkit.modules.transformer`` holds the positional encoding and the attention blocks of CSA and MulTA.
3. ``mamkit.utility.processing`` reads audio and computes MFCC and pretrained features.
4. ``mamkit.utility.text`` holds the tokenizer, the vocabulary, and the GloVe reader.
5. ``mamkit.utility.data`` downloads files and debate recordings.

API
---

.. automodule:: mamkit.modules.rnn
   :members:

.. automodule:: mamkit.modules.transformer
   :members:

.. automodule:: mamkit.utility.processing
   :members:

.. automodule:: mamkit.utility.text
   :members:

.. automodule:: mamkit.utility.data
   :members:
