# Codebase schema

This file records call relations and module dependencies of the `mamkit` package.

## Text utilities

`mamkit.utility.text` depends only on the standard library and torch.

1. `basic_english` lowercases a text, separates punctuation, and splits the text on whitespace.
2. `build_vocabulary` calls a tokenizer on every text and returns a `Vocabulary`.
3. `Vocabulary` maps tokens to indices and maps unknown tokens to `<unk>`.
4. `glove_vectors` calls `download` for the GloVe archive, extracts one vector file into a cache, and returns one vector per token.

## Audio utilities

`mamkit.utility.processing.load` reads an audio file through soundfile.
`parse_audio_nn` calls `load` and resamples the audio with `torchaudio.functional.resample`.
`encode_audio_nn` calls `parse_audio_nn`.
`parse_audio_mfcc` reads audio through librosa.
`encode_audio_mfcc` calls `parse_audio_mfcc` and stores the result in a cache keyed by path.
`encode_text_nn` calls a pretrained tokenizer and model.

## Downloads and metrics

`mamkit.utility.data.download` fetches a URL with httpx into a `.part` file and renames it when complete. `youtube_download` fetches the audio of each debate video with yt_dlp and skips the debates already downloaded.
`mamkit.utility.metrics.ClassSubsetMulticlassF1Score` computes the multiclass F1 score and averages it over a subset of classes.

## Encoders and models

`mamkit.components.encoders` depends on torch, transformers, `mamkit.modules.rnn`, and `mamkit.modules.transformer`.

1. Every encoder reads one sample, which maps `text` and `audio` to their fields, and returns one vector of width `output_dim`.
2. `TextBiLSTM` and `AudioBiLSTM` call `LSTMStack`. When the sample holds an `input_mask`, they pass its lengths, and `LSTMStack` packs the padded sequences.
3. `TextTransformer` calls a pretrained model from transformers. `TextPool` averages precomputed token embeddings.
4. `AudioTransformer` calls `torch.nn.TransformerEncoder` and `PositionalEncoding`.
5. `CSA` calls `CustomEncoder` and `PositionalEncoding`. `MulTA` calls `MulTACrossAttentionBlock` and `PositionalEncoding`.
6. `Concat` calls each of its encoders on the same sample.
7. `TextTransformer`, `TextPool`, `AudioTransformer`, `CSA`, and `MulTA` call `masked_mean`.

`mamkit.components.models` depends only on torch.

1. `Classifier` calls its encoder once, or once per side for a pair batch, and then its head.
2. `Ensemble` calls a text classifier and an audio classifier and mixes their class distributions.

## Datasets and featurizers

`mamkit.components.datasets` depends only on torch. `MAMDataset` holds one list per modality, or one such dictionary per side `a` and `b`, with a label per sample.

`mamkit.components.features` depends on `mamkit.components.datasets`, `mamkit.utility.processing`, `mamkit.utility.text`, and transformers.

1. `VocabularyText` calls `build_vocabulary` and `basic_english`, and `glove_vectors` when it has an embedding model.
2. `TokenizedText` calls a pretrained tokenizer. `VocabularyText.collate` and `TokenizedText.collate` call `check_text`.
3. `EncodedText`, `MFCCAudio`, and `EncodedAudio` extend `CachedFeaturizer`. Its `transform` calls `encode` only for inputs whose `input_key` is not cached.
4. `EncodedText.encode` calls `encode_text_nn`, `MFCCAudio.encode` calls `encode_audio_mfcc`, and `EncodedAudio.encode` calls `encode_audio_nn`. These functions call `check_text` or `check_audio`, which raise on an empty transcript or a missing clip.
5. Every `collate` method calls `pad`, except the one of `TokenizedText`.
6. The functions `fit`, `transform`, and `collate` call one featurizer per modality on each side of a `MAMDataset`.
7. `VocabularyText.model_inputs` returns the arguments `vocab_size` and `embedding_matrix` of `TextBiLSTM`.

## Loaders and split strategies

`mamkit.components.loaders` depends on `mamkit.components.datasets`, `mamkit.utility.data`, pandas, pydub, and nltk.

1. Each loader downloads its corpus in `__init__` through `build_once`, which builds a `.part` folder and renames it when the build completes. `MMUSED` and `MMUSEDFallacy` build their clip folder the same way, only when their input mode reads audio. `MMUSED`, `MMUSEDFallacy`, and `MArg` call `extract_zip`, which downloads one zip file from Zenodo and extracts it.
2. `build_clips` calls `youtube_download`, `trimmed_recording`, and `export_clip`.
3. `MArg.build` calls `build_chunks` when the feature table is missing, then `pairs`. `build_chunks` calls `split_transcripts`, which calls `split_sentences`, and then cuts one clip per timestamped sentence. `build_chunks` and `build` write their tables through `write_csv`, which renames a `.part` file when complete. `split_sentences` downloads the NLTK sentence model when it is missing. `MMUSEDFallacy` calls `window_start` to cut the context of both tasks.
4. `Loader.dataset` reads the column names from `columns` and returns a `MAMDataset` with the modalities of the input mode.

`mamkit.components.splits` depends on `mamkit.components.datasets` and `mamkit.utility.data`. Each strategy reads `loader.data`, calls `loader.dataset` on row subsets, and returns a list of `Split`. `FoldFiles` calls `download` when its file is missing from the loader folder.

`mamkit.configurations.splits` registers one configuration per published split. Its keys are declared in `mamkit.configurations.keys`. `Registry.build` on the `mamkit` package folder loads them.

## Task and Lightning model

`mamkit.components.tasks` depends on `mamkit.components.features`, `mamkit.components.lightning`, Lightning, and `cinnamon.registry`.

1. `Task.run` builds the loader and the split strategy with `Registry.from_key`, seeds Lightning, and calls `Task.train` once per seed and split.
2. `Task.run` builds one featurizer per modality with `Registry.from_key` and passes them to every `Task.train` call. `Task.train` calls `features.fit`, `features.transform`, and `features.collate`.
3. `Task.train` builds the model with `Registry.from_key`. Then it passes the featurizer `model_inputs` to `load_vocabulary`, which `TextBiLSTM` defines.
4. `Task.train` wraps the model in `MAMKitLightningModel` and calls `Trainer.fit` and `Trainer.test`.

`MAMKitLightningModel` calls `import_path` to resolve its loss and optimizer. `Classifier`, `Ensemble`, and `Concat` call `Registry.from_keys` on their child components. A configuration can therefore name a child by its key.

`mamkit.configurations.tasks` registers `TOY_TASK`, which reads the `Toy` loader through `COLUMN_SPLIT`.

## Experiment registrations

`mamkit.configurations.experiments` holds one module per corpus and one for the metrics. Each module registers the loaders, featurizers, encoders, models, and tasks of the published experiments of its corpus. The keys are declared in `mamkit.configurations.keys`.

1. A task registration names its loader, split strategy, text and audio featurizers, model, and metrics by key.
2. An `EncodedAudio` registration lists the alternative audio cards as variants of `model_card`. The registry then derives one task key per card, tagged `audio.model_card=<card>`.
3. A text card has its own registrations, because the tokenizer and the encoder must name the same card.

`tests/test_registrations.py` reads every task and its children back from the registry and compares them with `tests/golden/configs.json`.
