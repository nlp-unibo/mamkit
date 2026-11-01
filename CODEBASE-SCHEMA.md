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
