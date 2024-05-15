import logging
import pickle
from pathlib import Path
from typing import Optional, Iterable, Dict

import librosa
import numpy as np
import torch as th
from skimage.measure import block_reduce
from torchaudio.backend.soundfile_backend import load
from torchaudio.functional import resample
from torchtext.vocab import pretrained_aliases, build_vocab_from_iterator
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from mamkit.data.datasets import UnimodalDataset, MultimodalDataset, MAMDataset


class Processor:

    def fit(
            self,
            train_data: MAMDataset
    ):
        pass

    def clear(
            self
    ):
        pass

    def reset(
            self
    ):
        pass

    def __call__(
            self,
            data: MAMDataset
    ):
        return data


class ProcessorComponent:

    def fit(
            self,
            *args,
            **kwargs
    ):
        pass

    def clear(
            self
    ):
        pass

    def reset(
            self
    ):
        pass

    def __call__(
            self,
            *args,
            **kwargs
    ):
        return args, kwargs


class UnimodalProcessor(Processor):
    def __init__(
            self,
            features_processor=None,
            label_processor=None,
    ):
        self.features_processor = features_processor
        self.label_processor = label_processor

    def fit(
            self,
            train_data: UnimodalDataset
    ):
        if self.features_processor is not None:
            self.features_processor.fit(train_data.inputs)

        if self.label_processor is not None:
            self.label_processor.fit(train_data.labels)

    def __call__(
            self,
            data: UnimodalDataset
    ):
        if self.features_processor is not None:
            data.inputs = self.features_processor(data.inputs)

        if self.label_processor is not None:
            data.labels = self.label_processor(data.labels)

        return data

    def clear(
            self
    ):
        if self.features_processor is not None:
            self.features_processor.clear()

        if self.label_processor is not None:
            self.label_processor.clear()


class MultimodalProcessor(Processor):
    def __init__(
            self,
            text_processor=None,
            audio_processor=None,
            label_processor=None
    ):
        self.text_processor = text_processor
        self.audio_processor = audio_processor
        self.label_processor = label_processor

    def fit(
            self,
            train_data: MultimodalDataset
    ):
        if self.text_processor is not None:
            self.text_processor.fit(train_data.texts)

        if self.audio_processor is not None:
            self.audio_processor.fit(train_data.audio)

        if self.label_processor is not None:
            self.label_processor.fit(train_data.labels)

    def __call__(
            self,
            data: MultimodalDataset
    ):
        if self.text_processor is not None:
            data.texts = self.text_processor(data.texts)

        if self.audio_processor is not None:
            data.audio = self.audio_processor(data.audio)

        if self.label_processor is not None:
            data.labels = self.label_processor(data.labels)

        return data

    def clear(
            self
    ):
        if self.text_processor is not None:
            self.text_processor.clear()

        if self.audio_processor is not None:
            self.audio_processor.clear()

        if self.label_processor is not None:
            self.label_processor.clear()


class VocabBuilder(ProcessorComponent):

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

    def fit(
            self,
            texts
    ):
        logging.info('Building vocabulary...')
        self.vocab = build_vocab_from_iterator(
            iterator=iter([self.tokenizer(text) for text in texts]),
            specials=['<pad>', '<unk>'],
            special_first=True,
            **self.tokenization_args
        )
        self.vocab.set_default_index(self.vocab['<unk>'])

        if self.embedding_model is not None:
            self.embedding_matrix = self.embedding_model.get_vecs_by_tokens(self.vocab.get_itos())

    def __call__(
            self,
            texts
    ):
        return texts

    def clear(
            self
    ):
        self.embedding_model = None

    def reset(
            self
    ):
        self.embedding_matrix = None


class MFCCExtractor(ProcessorComponent):

    def __init__(
            self,
            mfccs: int,
            pooling_sizes: Optional[Iterable[int]] = None,
            remove_energy: bool = True,
            normalize: bool = True,
            serialization_path: Path = None
    ):
        self.mfccs = mfccs
        self.pooling_sizes = pooling_sizes
        self.remove_energy = remove_energy
        self.normalize = normalize
        self.serialization_path = serialization_path if serialization_path is not None else Path('mfccs.pkl')

    def parse_audio(
            self,
            audio_file
    ):
        audio, sampling_rate = librosa.load(audio_file)
        mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=self.mfccs)[2:]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sampling_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sampling_rate)
        chroma_ft = librosa.feature.chroma_stft(y=audio, sr=sampling_rate)

        # [frames, mfccs]
        features = np.concatenate(
            (spectral_centroids, spectral_bandwidth, spectral_rolloff, spectral_contrast, chroma_ft, mfccs),
            axis=0).transpose()

        if self.remove_energy:
            features = features[:, 1:]

        if self.pooling_sizes is not None:
            for pooling_size in self.pooling_sizes:
                features = block_reduce(features, (pooling_size, 1), np.mean)

        if self.normalize and features.shape[0] > 1:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            features = np.divide((features - mean[np.newaxis, :]), std[np.newaxis, :])

        return features

    def __call__(
            self,
            audio_files: Iterable[Path]
    ):
        preloaded_mfccs: Dict = {}
        preloaded_length = 0
        if self.serialization_path.exists():
            with self.serialization_path.open('rb') as f:
                preloaded_mfccs: Dict = pickle.load(f)
                preloaded_length = len(preloaded_mfccs)

        features = []
        for audio_file in tqdm(audio_files, desc='Extracting MFCCs'):
            assert audio_file.is_file(), f'Could not find file {audio_file}'

            if preloaded_mfccs[audio_file.as_posix()]:
                audio_features = preloaded_mfccs[audio_file.as_posix()]
            else:
                audio_features = self.parse_audio(audio_file=audio_file)
                preloaded_mfccs[audio_file.as_posix()] = audio_features

            features.append(audio_features)

        if len(preloaded_mfccs) != preloaded_length:
            with self.serialization_path.open('wb') as f:
                pickle.dump(preloaded_mfccs, f)

        return features


class TextTransformer(ProcessorComponent):

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
        self.model = None
        self.tokenizer = None

    def _init_models(
            self
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        self.model = AutoModel.from_pretrained(self.model_card).to(self.device)

    def __call__(
            self,
            texts
    ):
        if self.model is None:
            self._init_models()

        tokenized = self.tokenizer(texts,
                                   padding=True,
                                   truncation=True,
                                   return_tensors='pt',
                                   **self.tokenizer_args).to(self.device)
        text_features = self.model(**tokenized, **self.model_args).last_hidden_state
        return text_features.detach().cpu().numpy()

    def clear(
            self
    ):
        self.tokenizer = None
        self.model = None
        th.cuda.empty_cache()


class AudioTransformer(ProcessorComponent):

    def __init__(
            self,
            model_card,
            sampling_rate,
            downsampling_factor=None,
            aggregate: bool = False,
            processor_args=None,
            model_args=None
    ):
        self.model_card = model_card
        self.sampling_rate = sampling_rate
        self.downsampling_factor = downsampling_factor
        self.aggregate = aggregate
        self.processor_args = processor_args if processor_args is not None else {}
        self.model_args = model_args if model_args is not None else {}
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.processor = None
        self.model = None

    def _init_models(
            self
    ):
        self.processor = AutoProcessor.from_pretrained(self.model_card)
        self.model = AutoModel.from_pretrained(self.model_card).to(self.device)

    def __call__(
            self,
            audio_files: Iterable[Path]
    ):
        if self.model is None:
            self._init_models()

        parsed_audio = []
        for audio_file in tqdm(audio_files, desc='Extracting Audio Features...'):
            if not audio_file.is_file():
                raise RuntimeError(f'Could not read file {audio_file}')
            audio, sampling_rate = load(audio_file)
            if sampling_rate != self.sampling_rate:
                audio = resample(audio, sampling_rate, self.sampling_rate)
                audio = th.mean(audio, dim=0, keepdim=True)

            with th.inference_mode():
                features = self.processor(audio,
                                          sampling_rate=self.sampling_rate,
                                          padding=True,
                                          return_tensors='pt',
                                          **self.processor_args)
                features = features.input_values[0].to(self.device)
                features = self.model(features, **self.model_args).last_hidden_state[0].unsqueeze(0)

                if self.downsampling_factor is not None:
                    features = th.nn.functional.interpolate(features.permute(0, 2, 1),
                                                            scale_factor=self.downsampling_factor,
                                                            mode='linear')
                    features = features.permute(0, 2, 1)
                features = features[0].detach().cpu().numpy()

            if self.aggregate:
                features = np.mean(features.squeeze(axis=0), axis=0, keepdims=True)

            parsed_audio.append(features)

        return parsed_audio

    def clear(
            self
    ):
        self.model = None
        self.processor = None
        th.cuda.empty_cache()
