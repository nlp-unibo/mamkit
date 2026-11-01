# Codebase schema

This file records call relations and module dependencies. It covers the modules that changed since the schema was added, and it grows as other modules change.

## Text utilities

`mamkit.utility.text` depends only on the standard library and torch.

1. `basic_english` lowercases a text, separates punctuation, and splits the text on whitespace.
2. `build_vocabulary` calls a tokenizer on every text and returns a `Vocabulary`.
3. `Vocabulary` maps tokens to indices and maps unknown tokens to `<unk>`.
4. `glove_vectors` downloads and extracts a GloVe file into a cache and returns one vector per token.

`mamkit.data.processing.VocabBuilder.fit` and `PairVocabBuilder.fit` call `build_vocabulary`. When an embedding model is set, they also call `glove_vectors`.

`mamkit.data.collators.TextCollator` and `PairTextCollator` call a `Vocabulary` through `mamkit.utility.collators.encode_text_torch`.

`mamkit.configs.text` and `mamkit.configs.text_audio` pass `basic_english` as the tokenizer of every vocabulary configuration.

## Audio utilities

`mamkit.utility.processing.load` reads an audio file through soundfile.
`parse_audio_nn` calls `load` and resamples the audio with `torchaudio.functional.resample`.
`encode_audio_nn` calls `parse_audio_nn`.
`parse_audio_mfcc` reads audio through librosa.

## Encoders and models

`mamkit.components.encoders` depends on torch, transformers, `mamkit.modules.rnn`, and `mamkit.modules.transformer`.

1. Every encoder reads one sample, which maps `text` and `audio` to their fields, and returns one vector of width `output_dim`.
2. `TextBiLSTM` and `AudioBiLSTM` call `LSTMStack`. When the sample holds an `input_mask`, they pass its lengths, and `LSTMStack` packs the padded sequences.
3. `TextTransformer` calls a pretrained model from transformers. `TextPool` averages precomputed token embeddings.
4. `AudioTransformer` calls `torch.nn.TransformerEncoder` and `PositionalEncoding`.
5. `CSA` calls `CustomEncoder` and `PositionalEncoding`. `MulTA` calls `MulTA_CrossAttentionBlock` and `PositionalEncoding`.
6. `Concat` calls each of its encoders on the same sample.
7. `TextTransformer`, `TextPool`, `AudioTransformer`, `CSA`, and `MulTA` call `masked_mean`.

`mamkit.components.models` depends only on torch.

1. `Classifier` calls its encoder once, or once per side for a pair batch, and then its head.
2. `Ensemble` calls a text classifier and an audio classifier and mixes their class distributions.

The model classes in `mamkit.models` stay until the legacy configurations that use them are removed.
