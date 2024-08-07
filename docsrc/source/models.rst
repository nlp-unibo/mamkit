.. _models:

Models
*********************************************
Currently available models can manage text and audio modalities.
For each model, we summarize its text and audio encoding modules and its fusion strategy.
Fusions strategies are divided into three categories:

- *Concat*: Concatenation of the text and audio embeddings.
- *Avg*: Average of the text and audio embeddings.
- *Cross*: Crossmodal Attention between the text and audio embeddings.



.. list-table:: **Models Information**
   :header-rows: 1
   :widths: 25 30 30 15

   * - **Model**
     - **Text Encoding**
     - **Audio Encoding**
     - **Fusion**
   * - **BiLSTM**
     - GloVe + BiLSTM
     - (Wav2Vec2 or MFCCs) + BiLSTM
     - Concat-Late
   * - **MM-BERT**
     - BERT
     - (Wav2Vec2 or HuBERT or WavLM) + BiLSTM
     - Concat-Late
   * - **MM-RoBERTa**
     - RoBERTa
     - (Wav2Vec2 or HuBERT or WavLM) + BiLSTM
     - Concat-Late
   * - **CSA**
     - BERT
     - (Wav2Vec2 or HuBERT or WavLM) + Transformer
     - Concat-Early
   * - **Ensemble**
     - BERT
     - (Wav2Vec2 or HuBERT or WavLM) + Transformer
     - Avg-Late
   * - **Mul-TA**
     - BERT
     - (Wav2Vec2 or HuBERT or WavLM) + Transformer
     - Cross
