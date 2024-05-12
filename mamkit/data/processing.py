from pathlib import Path
from typing import Optional, Iterable

import librosa
import numpy as np
import resampy
import torch as th
from skimage.measure import block_reduce
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import pretrained_aliases, build_vocab_from_iterator, vocab, GloVe
from transformers import AutoTokenizer, AutoModel, AutoProcessor


class UnimodalCollator:
    def __init__(
            self,
            features_collator,
            label_collator
    ):
        self.features_collator = features_collator
        self.label_collator = label_collator

    def __call__(
            self,
            batch
    ):
        features_raw, labels = zip(*batch)
        if self.features_collator is None:
            features_collated = features_raw
        else:
            features_collated = self.features_collator(features_raw)

        if self.label_collator is None:
            labels_collated = labels
        else:
            labels_collated = self.label_collator(labels)

        return features_collated, labels_collated


class MultimodalCollator:
    def __init__(
            self,
            text_collator,
            audio_collator,
            label_collator
    ):
        self.text_collator = text_collator
        self.audio_collator = audio_collator
        self.label_collator = label_collator

    def __call__(self, batch):
        text_raw, audio_raw, labels = zip(*batch)
        if self.text_collator is None:
            text_collated = text_raw
        else:
            text_collated = self.text_collator(text_raw)

        if self.audio_collator is None:
            audio_collated = audio_raw
        else:
            audio_collated = self.audio_collator(audio_raw)

        if self.label_collator is None:
            labels_collated = labels
        else:
            labels_collated = self.label_collator(labels)

        return (*text_collated, *audio_collated), labels_collated


class VocabBuilder:

    def __init__(
            self,
            tokenizer,
            embedding_dim: int,
            embedding_model: Optional[str] = None,
            tokenization_args=None
    ):
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.embedding_model = pretrained_aliases[embedding_model]() if embedding_model is not None else None
        self.vocab = None
        self.tokenization_args = tokenization_args if tokenization_args is not None else {}
        self.embedding_matrix = None

    def __call__(
            self,
            texts
    ):
        self.vocab = build_vocab_from_iterator(
            iterator=iter([self.tokenizer(text) for text in texts]),
            specials=['<pad>', '<unk>'],
            special_first=True,
            **self.tokenization_args
        )
        self.vocab.set_default_index(self.vocab['<unk>'])

        if self.embedding_model is not None:
            self.embedding_matrix = self.embedding_model.get_vecs_by_tokens(self.vocab.get_itos())


class TextCollator:

    def __init__(
            self,
            tokenizer
    ):
        self.tokenizer = tokenizer

    def __call__(
            self,
            texts
    ):
        texts = [th.tensor(vocab(self.tokenizer(text))) for text in texts]
        return pad_sequence(texts, padding_value=0, batch_first=True)


class TextTransformerCollator:
    def __init__(
            self,
            model_card,
            tokenizer_args=None,
            model_args=None,
    ):
        self.model_card = model_card
        self.tokenizer_args = tokenizer_args if tokenizer_args is not None else {}
        self.model_args = model_args if model_args is not None else {}

        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_card)
        self.model = AutoModel.from_pretrained(model_card).to(self.device)

    def __call__(
            self,
            text
    ):
        tokenized = self.tokenizer(text,
                                   padding=True,
                                   truncation=True,
                                   return_tensors='pt',
                                   **self.tokenizer_args).to(self.device)
        text_features = self.model(**tokenized, **self.model_args).last_hidden_state
        text_attentions = tokenized.attention_mask
        return text_features, text_attentions


class MFCCCollator:

    def __init__(
            self,
            n_mfcc: int,
            pooling_sizes: Optional[Iterable[int]] = None,
            remove_energy: bool = True,
            normalize: bool = True
    ):
        self.n_mfcc = n_mfcc
        self.pooling_sizes = pooling_sizes
        self.remove_energy = remove_energy
        self.normalize = normalize

    def parse_audio(
            self,
            audio_file
    ):
        audio, sampling_rate = librosa.load(audio_file)
        mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=self.n_mfcc)[2:]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sampling_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sampling_rate)
        chroma_ft = librosa.feature.chroma_stft(y=audio, sr=sampling_rate)

        # [frames, n_mfcc]
        features = np.concatenate(
            (spectral_centroids, spectral_bandwidth, spectral_rolloff, spectral_contrast, chroma_ft, mfccs),
            axis=0).transpose()

        if self.remove_energy:
            features = features[:, 1:]

        if self.pooling_sizes is not None:
            for pooling_size in self.pooling_sizes:
                features = block_reduce(features, (pooling_size, 1), np.mean)

        if self.normalize:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            features = np.divide((features - mean[np.newaxis, :]), std[np.newaxis, :])

        return features

    def __call__(
            self,
            audio_files: Iterable[Path]
    ):
        features = []
        for audio_file in audio_files:
            assert audio_file.is_file(), f'Could not find file {audio_file}'
            audio_features = self.parse_audio(audio_file=audio_file)
            audio_features = th.tensor(audio_features, dtype=th.float32)
            features.append(audio_features)

        return th.tensor(features)


class AudioTransformerCollator:

    def __init__(
            self,
            model_card,
            sampling_rate,
            aggregate: bool = False,
            processor_args=None,
            model_args=None
    ):
        self.model_card = model_card
        self.sampling_rate = sampling_rate
        self.aggregate = aggregate
        self.processor_args = processor_args if processor_args is not None else {}
        self.model_args = model_args if model_args is not None else {}

        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        self.processor = AutoProcessor.from_pretrained(model_card).to(self.device)
        self.model = AutoModel.from_pretrained(model_card).to(self.device)

    def __call__(
            self,
            audio_files: Iterable[Path]
    ):
        audio_raw = []
        for audio_file in audio_files:
            assert audio_file.is_file()
            audio, sampling_rate = librosa.load(audio_file, sr=None)
            audio = resampy.resample(audio, sampling_rate, self.sampling_rate)
            audio_raw.append(audio)

        features = self.processor(audio_raw,
                                  sampling_rate=self.sampling_rate,
                                  padding=True,
                                  truncation=True,
                                  return_tensors='pt',
                                  return_attention_mask=True,
                                  **self.processor_args)
        attention_mask = features.attention_mask
        features = features.input_values[0]
        features = self.model(features[None, :], **self.model_args).last_hidden_state

        if self.aggregate:
            features = np.mean(features.detach().cpu().numpy().squeeze(axis=0), axis=0)

        return features, attention_mask
