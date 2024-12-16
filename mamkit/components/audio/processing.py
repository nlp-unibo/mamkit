import pickle
from pathlib import Path
from typing import Optional, Iterable, Dict, List

import librosa
import numpy as np
import resampy
import torch as th
from skimage.measure import block_reduce
from torch.nn.utils.rnn import pad_sequence
from torchaudio.backend.soundfile_backend import load
from torchaudio.functional import resample
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoFeatureExtractor

from mamkit.components.processing import ProcessorComponent

__all__ = [
    'MFCCExtractor',
    'PairMFCCExtractor',
    'AudioTransformer',
    'PairAudioTransformer',
    'AudioTransformerExtractor',
    'PairAudioTransformerExtractor'
]


class MFCCExtractor(ProcessorComponent):

    def __init__(
            self,
            mfccs: int,
            sampling_rate=16000,
            pooling_sizes: Optional[Iterable[int]] = None,
            remove_energy: bool = True,
            normalize: bool = True,
            serialization_path: Path = None
    ):
        self.mfccs = mfccs
        self.sampling_rate = sampling_rate
        self.pooling_sizes = pooling_sizes
        self.remove_energy = remove_energy
        self.normalize = normalize
        self.serialization_path = serialization_path if serialization_path is not None else Path('mfccs.pkl')

    def parse_audio(
            self,
            audio_file
    ):
        audio, sampling_rate = librosa.load(audio_file)
        if sampling_rate != self.sampling_rate:
            audio = resampy.resample(audio, sampling_rate, self.sampling_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=self.mfccs)[2:]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sampling_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sampling_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sampling_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sampling_rate)
        chroma_ft = librosa.feature.chroma_stft(y=audio, sr=self.sampling_rate)

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
            audio_file = Path(audio_file) if type(audio_file) != Path else audio_file
            assert audio_file.is_file(), f'Could not find file {audio_file}'

            if audio_file.as_posix() in preloaded_mfccs:
                audio_features = preloaded_mfccs[audio_file.as_posix()]
            else:
                audio_features = self.parse_audio(audio_file=audio_file)
                preloaded_mfccs[audio_file.as_posix()] = audio_features

            features.append(audio_features)

        if len(preloaded_mfccs) != preloaded_length:
            with self.serialization_path.open('wb') as f:
                pickle.dump(preloaded_mfccs, f)

        return features


class PairMFCCExtractor(ProcessorComponent):

    def __init__(
            self,
            mfccs: int,
            sampling_rate=16000,
            pooling_sizes: Optional[Iterable[int]] = None,
            remove_energy: bool = True,
            normalize: bool = True,
            serialization_path: Path = None
    ):
        self.mfccs = mfccs
        self.sampling_rate = sampling_rate
        self.pooling_sizes = pooling_sizes
        self.remove_energy = remove_energy
        self.normalize = normalize
        self.serialization_path = serialization_path if serialization_path is not None else Path('mfccs.pkl')

    def parse_audio(
            self,
            audio_file
    ):
        audio, sampling_rate = librosa.load(audio_file)
        if sampling_rate != self.sampling_rate:
            audio = resampy.resample(audio, sampling_rate, self.sampling_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=self.mfccs)[2:]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sampling_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sampling_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sampling_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sampling_rate)
        chroma_ft = librosa.feature.chroma_stft(y=audio, sr=self.sampling_rate)

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
            a_audio_files: Iterable[Path],
            b_audio_files: Iterable[Path]
    ):
        preloaded_mfccs: Dict = {}
        preloaded_length = 0
        if self.serialization_path.exists():
            with self.serialization_path.open('rb') as f:
                preloaded_mfccs: Dict = pickle.load(f)
                preloaded_length = len(preloaded_mfccs)

        a_features, b_features = [], []
        for a_audio_file, b_audio_file in tqdm(zip(a_audio_files, b_audio_files), desc='Extracting MFCCs'):
            a_audio_file = Path(a_audio_file) if type(a_audio_file) != Path else a_audio_file
            b_audio_file = Path(b_audio_file) if type(b_audio_file) != Path else b_audio_file
            assert a_audio_file.is_file(), f'Could not find file {a_audio_file}'
            assert b_audio_file.is_file(), f'Could not find file {b_audio_file}'

            if a_audio_file.as_posix() in preloaded_mfccs:
                a_audio_features = preloaded_mfccs[a_audio_file.as_posix()]
            else:
                a_audio_features = self.parse_audio(audio_file=a_audio_file)
                preloaded_mfccs[a_audio_file.as_posix()] = a_audio_features

            a_features.append(a_audio_features)

            if b_audio_file.as_posix() in preloaded_mfccs:
                b_audio_features = preloaded_mfccs[b_audio_file.as_posix()]
            else:
                b_audio_features = self.parse_audio(audio_file=b_audio_file)
                preloaded_mfccs[b_audio_file.as_posix()] = b_audio_features

            b_features.append(b_audio_features)

        if len(preloaded_mfccs) != preloaded_length:
            with self.serialization_path.open('wb') as f:
                pickle.dump(preloaded_mfccs, f)

        return a_features, b_features


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
            audio_files: Iterable[str]
    ):
        if self.model is None:
            self._init_models()

        parsed_audio = []
        for audio_file in tqdm(audio_files, desc='Extracting Audio Features...'):
            if not Path(audio_file).is_file():
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
                model_features = self.model(features, **self.model_args).last_hidden_state[0].unsqueeze(0)

                if self.downsampling_factor is not None:
                    try:
                        features = th.nn.functional.interpolate(model_features.permute(0, 2, 1),
                                                                scale_factor=self.downsampling_factor,
                                                                mode='linear')
                        features = features.permute(0, 2, 1)
                    except RuntimeError:
                        features = model_features
                else:
                    features = model_features

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


class AudioTransformerExtractor(ProcessorComponent):

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
        self.processor = AutoFeatureExtractor.from_pretrained(self.model_card)
        self.model = AutoModel.from_pretrained(self.model_card).to(self.device)

    def __call__(
            self,
            audio_files: Iterable[str]
    ):
        if self.model is None:
            self._init_models()

        parsed_audio = []
        for audio_file in tqdm(audio_files, desc='Extracting Audio Features...'):
            if not Path(audio_file).is_file():
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
                model_features = self.model(features, **self.model_args).last_hidden_state[0].unsqueeze(0)

                if self.downsampling_factor is not None:
                    try:
                        features = th.nn.functional.interpolate(model_features.permute(0, 2, 1),
                                                                scale_factor=self.downsampling_factor,
                                                                mode='linear')
                        features = features.permute(0, 2, 1)
                    except RuntimeError:
                        features = model_features
                else:
                    features = model_features

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


class PairAudioTransformer(ProcessorComponent):

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
            a_audio_files: List[Path],
            b_audio_files: List[Path]
    ):
        if self.model is None:
            self._init_models()

        a_parsed_audio, b_parsed_audio = [], []
        for a_audio_file, b_audio_file in tqdm(zip(a_audio_files, b_audio_files), total=len(a_audio_files),
                                               desc='Extracting Audio Features...'):
            if not Path(a_audio_file).is_file():
                raise RuntimeError(f'Could not read file {a_audio_file}')
            a_audio, a_sampling_rate = load(a_audio_file)
            if a_sampling_rate != self.sampling_rate:
                a_audio = resample(a_audio, a_sampling_rate, self.sampling_rate)
            a_audio = th.mean(a_audio, dim=0)

            if not Path(b_audio_file).is_file():
                raise RuntimeError(f'Could not read file {b_audio_file}')
            b_audio, b_sampling_rate = load(b_audio_file)
            if b_sampling_rate != self.sampling_rate:
                b_audio = resample(b_audio, b_sampling_rate, self.sampling_rate)
            b_audio = th.mean(b_audio, dim=0)

            pair_audio = pad_sequence([a_audio, b_audio], batch_first=True, padding_value=0.0)
            with th.inference_mode():
                features = self.processor(pair_audio,
                                          sampling_rate=self.sampling_rate,
                                          padding=True,
                                          return_tensors='pt',
                                          **self.processor_args)
                features = features.input_values[0].to(self.device)
                model_features = self.model(features, **self.model_args).last_hidden_state

                if self.downsampling_factor is not None:
                    try:
                        features = th.nn.functional.interpolate(model_features.permute(0, 2, 1),
                                                                scale_factor=self.downsampling_factor,
                                                                mode='linear')
                        features = features.permute(0, 2, 1)
                    except RuntimeError:
                        features = model_features
                else:
                    features = model_features

                features = features.detach().cpu().numpy()

            if self.aggregate:
                features = np.mean(features, axis=1, keepdims=True)

            a_parsed_audio.append(features[0])
            b_parsed_audio.append(features[1])

        return a_parsed_audio, b_parsed_audio

    def clear(
            self
    ):
        self.model = None
        self.processor = None
        th.cuda.empty_cache()


class PairAudioTransformerExtractor(ProcessorComponent):

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
        self.processor = AutoFeatureExtractor.from_pretrained(self.model_card)
        self.model = AutoModel.from_pretrained(self.model_card).to(self.device)

    def __call__(
            self,
            a_audio_files: List[Path],
            b_audio_files: List[Path]
    ):
        if self.model is None:
            self._init_models()

        a_parsed_audio, b_parsed_audio = [], []
        for a_audio_file, b_audio_file in tqdm(zip(a_audio_files, b_audio_files), total=len(a_audio_files),
                                               desc='Extracting Audio Features...'):
            if not Path(a_audio_file).is_file():
                raise RuntimeError(f'Could not read file {a_audio_file}')
            a_audio, a_sampling_rate = load(a_audio_file)
            if a_sampling_rate != self.sampling_rate:
                a_audio = resample(a_audio, a_sampling_rate, self.sampling_rate)
            a_audio = th.mean(a_audio, dim=0)

            if not Path(b_audio_file).is_file():
                raise RuntimeError(f'Could not read file {b_audio_file}')
            b_audio, b_sampling_rate = load(b_audio_file)
            if b_sampling_rate != self.sampling_rate:
                b_audio = resample(b_audio, b_sampling_rate, self.sampling_rate)
            b_audio = th.mean(b_audio, dim=0)

            pair_audio = pad_sequence([a_audio, b_audio], batch_first=True, padding_value=0.0)
            with th.inference_mode():
                features = self.processor(pair_audio,
                                          sampling_rate=self.sampling_rate,
                                          padding=True,
                                          return_tensors='pt',
                                          **self.processor_args)
                features = features.input_values[0].to(self.device)
                model_features = self.model(features, **self.model_args).last_hidden_state

                if self.downsampling_factor is not None:
                    try:
                        features = th.nn.functional.interpolate(model_features.permute(0, 2, 1),
                                                                scale_factor=self.downsampling_factor,
                                                                mode='linear')
                        features = features.permute(0, 2, 1)
                    except RuntimeError:
                        features = model_features
                else:
                    features = model_features

                features = features.detach().cpu().numpy()

            if self.aggregate:
                features = np.mean(features, axis=1, keepdims=True)

            a_parsed_audio.append(features[0])
            b_parsed_audio.append(features[1])

        return a_parsed_audio, b_parsed_audio

    def clear(
            self
    ):
        self.model = None
        self.processor = None
        th.cuda.empty_cache()
