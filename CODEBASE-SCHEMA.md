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

## Datasets and featurizers

`mamkit.components.datasets` depends only on torch. `MAMDataset` holds one list per modality, or one such dictionary per side `a` and `b`, with a label per sample.

`mamkit.components.features` depends on `mamkit.components.datasets`, `mamkit.utility.processing`, `mamkit.utility.text`, and transformers.

1. `VocabularyText` calls `build_vocabulary` and `basic_english`, and `glove_vectors` when it has an embedding model.
2. `TokenizedText` calls a pretrained tokenizer. `EncodedText` calls `encode_text_nn`.
3. `MFCCAudio` calls `encode_audio_mfcc`. `EncodedAudio` calls `encode_audio_nn`.
4. Every `collate` method calls `pad`, except the one of `TokenizedText`.
5. The functions `fit`, `transform`, and `collate` call one featurizer per modality on each side of a `MAMDataset`.
6. `VocabularyText.model_inputs` returns the arguments `vocab_size` and `embedding_matrix` of `TextBiLSTM`.

The processors and collators in `mamkit.data` stay until the legacy configurations that use them are removed.

## Loaders and split strategies

`mamkit.components.loaders` depends on `mamkit.components.datasets`, `mamkit.utility.data`, pandas, pydub, and nltk.

1. Each loader downloads its corpus in `__init__` when its folder is missing. `MMUSED` and `MMUSEDFallacy` call `build_clips` only when their input mode reads audio.
2. `build_clips` calls `youtube_download`, `trimmed_recording`, and `export_clip`.
3. `Loader.dataset` reads the column names from `columns` and returns a `MAMDataset` with the modalities of the input mode.

`mamkit.components.splits` depends on `mamkit.components.datasets` and `mamkit.utility.data`. Each strategy reads `loader.data`, calls `loader.dataset` on row subsets, and returns a list of `Split`. `FoldFiles` calls `download` when its file is missing from the loader folder.

`mamkit.configurations.splits` registers one configuration per published split. Its keys are declared in `mamkit.configurations.keys`. `Registry.build` on the `mamkit` package folder loads them.

The loaders in `mamkit.data.datasets` stay until the legacy configurations that use them are removed.

## Task and Lightning model

`mamkit.components.tasks` depends on `mamkit.components.features`, `mamkit.components.lightning`, Lightning, and `cinnamon.registry`.

1. `Task.run` builds the loader and the split strategy with `Registry.from_key`, seeds Lightning, and calls `Task.train` once per seed and split.
2. `Task.train` builds fresh featurizers with `Registry.from_keys` and calls `features.fit`, `features.transform`, and `features.collate`.
3. `Task.train` builds the model with `Registry.from_key`. Then it passes the featurizer `model_inputs` to `load_vocabulary`, which `TextBiLSTM` defines.
4. `Task.train` wraps the model in `MAMKitLightningModel` and calls `Trainer.fit` and `Trainer.test`.

`MAMKitLightningModel` calls `import_path` to resolve its loss and optimizer. `Classifier`, `Ensemble`, and `Concat` call `Registry.from_keys` on their child components. A configuration can therefore name a child by its key.

`mamkit.configurations.tasks` registers `TOY_TASK`, which reads the `Toy` loader through `COLUMN_SPLIT`.
