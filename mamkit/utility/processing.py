import librosa
import numpy as np
from skimage.measure import block_reduce
import resampy
from typing import Optional, Iterable, Union, List, Dict
from pathlib import Path
from torchaudio import load
import torch as th
from torchaudio.functional import resample


def parse_audio_mfcc(
        audio_input: Union[Path, List[Path]],
        mfccs: int,
        sampling_rate: int = 16000,
        pooling_sizes: Optional[Iterable[int]] = None,
        remove_energy: bool = True,
        normalize: bool = True,
):
    if isinstance(audio_input, Path):
        audio, audio_sampling_rate = librosa.load(audio_input)
    else:
        _, audio_sampling_rate = librosa.load(audio_input[0])
        audio = np.concatenate([librosa.load(item)[0] for item in audio_input])

    if audio_sampling_rate != sampling_rate:
        audio = resampy.resample(audio, audio_sampling_rate, sampling_rate)
    mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=mfccs)[2:]
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sampling_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sampling_rate)
    chroma_ft = librosa.feature.chroma_stft(y=audio, sr=sampling_rate)

    # [frames, mfccs]
    features = np.concatenate(
        (spectral_centroids, spectral_bandwidth, spectral_rolloff, spectral_contrast, chroma_ft, mfccs),
        axis=0).transpose()

    if remove_energy:
        features = features[:, 1:]

    if pooling_sizes is not None:
        for pooling_size in pooling_sizes:
            features = block_reduce(features, (pooling_size, 1), np.mean)

    if normalize and features.shape[0] > 1:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        features = np.divide((features - mean[np.newaxis, :]), std[np.newaxis, :])

    return features


def encode_audio_mfcc(
        audio_input: Union[Path, List[Path]],
        mfccs: int,
        sampling_rate: int = 16000,
        pooling_sizes: Optional[Iterable[int]] = None,
        remove_energy: bool = True,
        normalize: bool = True,
        preloaded_mfccs: Dict = {}
):
    if type(audio_input) == list and not len(audio_input):
        return np.array([0.0])

    if audio_input is None:
        return np.array([0.0])

    if isinstance(audio_input, Path) and not audio_input.exists():
        return np.array([0.0])

    input_hash = hash(audio_input.as_posix() if isinstance(audio_input, Path)
                      else '--'.join(sorted([item.as_posix() for item in audio_input])))
    if input_hash in preloaded_mfccs:
        return preloaded_mfccs[input_hash]

    audio_features = parse_audio_mfcc(audio_input=audio_input,
                                      mfccs=mfccs,
                                      normalize=normalize,
                                      pooling_sizes=pooling_sizes,
                                      remove_energy=remove_energy,
                                      sampling_rate=sampling_rate)
    preloaded_mfccs[input_hash] = audio_features

    return audio_features


def encode_audio_and_context_mfcc(
        audio_input: Union[Path, List[Path]],
        mfccs: int,
        audio_context: List[Path] = None,
        sampling_rate: int = 16000,
        pooling_sizes: Optional[Iterable[int]] = None,
        remove_energy: bool = True,
        normalize: bool = True,
        preloaded_mfccs: Dict = {}
):
    audio_features = encode_audio_mfcc(audio_input=audio_input,
                                       preloaded_mfccs=preloaded_mfccs,
                                       mfccs=mfccs,
                                       normalize=normalize,
                                       pooling_sizes=pooling_sizes,
                                       remove_energy=remove_energy,
                                       sampling_rate=sampling_rate)

    context_audio_features = None
    if audio_context is not None:
        context_audio_features = encode_audio_mfcc(audio_input=audio_context,
                                                   preloaded_mfccs=preloaded_mfccs,
                                                   mfccs=mfccs,
                                                   normalize=normalize,
                                                   pooling_sizes=pooling_sizes,
                                                   remove_energy=remove_energy,
                                                   sampling_rate=sampling_rate)

    return audio_features, context_audio_features


def parse_audio_nn(
        audio_input: Union[Path, List[Path]],
        sampling_rate: int = 16000,
):
    if isinstance(audio_input, Path):
        audio, audio_sampling_rate = load(audio_input)
    else:
        _, audio_sampling_rate = load(audio_input[0])
        audio = th.concatenate([load(item)[0] for item in audio_input], dim=-1)

    if sampling_rate != audio_sampling_rate:
        audio = resample(audio, sampling_rate, sampling_rate)

    audio = th.mean(audio, dim=0, keepdim=True)
    return audio


def encode_audio_nn(
        audio_input: Union[Path, List[Path]],
        processor,
        model,
        device,
        processor_args: Dict = {},
        model_args: Dict = {},
        sampling_rate: int = 16000,
        downsampling_factor=None,
        aggregate: bool = False,
):
    if audio_input is None:
        return th.tensor([0.0], dtype=th.float32)

    if isinstance(audio_input, Path) and not audio_input.exists():
        return th.tensor([0.0], dtype=th.float32)

    if type(audio_input) == list and not len(audio_input):
        return th.tensor([0.0], dtype=th.float32)

    audio = parse_audio_nn(audio_input=audio_input,
                           sampling_rate=sampling_rate)

    with th.inference_mode():
        features = processor(audio,
                             sampling_rate=sampling_rate,
                             padding=True,
                             return_tensors='pt',
                             **processor_args)
        features = features.input_values[0].to(device)
        model_features = model(features, **model_args).last_hidden_state[0].unsqueeze(0)

        if downsampling_factor is not None:
            try:
                features = th.nn.functional.interpolate(model_features.permute(0, 2, 1),
                                                        scale_factor=downsampling_factor,
                                                        mode='linear')
                features = features.permute(0, 2, 1)
            except RuntimeError:
                features = model_features
        else:
            features = model_features

        features = features[0].detach().cpu().numpy()

    if aggregate:
        features = np.mean(features.squeeze(axis=0), axis=0, keepdims=True)

    return features


def encode_audio_and_context_nn(
        audio_input: Union[Path, List[Path]],
        processor,
        model,
        device,
        audio_context: List[Path] = None,
        processor_args: Dict = {},
        model_args: Dict = {},
        sampling_rate: int = 16000,
        downsampling_factor=None,
        aggregate: bool = False,
):
    audio_features = encode_audio_nn(audio_input=audio_input,
                                     model=model,
                                     processor=processor,
                                     processor_args=processor_args,
                                     model_args=model_args,
                                     device=device,
                                     sampling_rate=sampling_rate,
                                     downsampling_factor=downsampling_factor,
                                     aggregate=aggregate)

    context_audio_features = None
    if audio_context is not None:
        context_audio_features = encode_audio_nn(audio_input=audio_context,
                                                 model=model,
                                                 processor=processor,
                                                 processor_args=processor_args,
                                                 model_args=model_args,
                                                 device=device,
                                                 sampling_rate=sampling_rate,
                                                 downsampling_factor=downsampling_factor,
                                                 aggregate=aggregate)

    return audio_features, context_audio_features


def encode_text_nn(
        text: str,
        tokenizer,
        model,
        device,
        tokenizer_args: Dict = None,
        model_args: Dict = None
):
    if text is None or not len(text):
        return th.tensor([0.0], dtype=th.float32)

    tokenized = tokenizer([text],
                          padding=True,
                          return_tensors='pt',
                          **tokenizer_args).to(device)
    model_output = model(**tokenized, **model_args)
    text_emb = model_output.last_hidden_state * tokenized.attention_mask[:, :, None]
    text_emb = text_emb.detach().cpu().numpy()[0]

    return text_emb


def encode_text_and_context_nn(
        text: str,
        tokenizer,
        model,
        device,
        context: str = None,
        tokenizer_args: Dict = None,
        model_args: Dict = None
):
    text_emb = encode_text_nn(text=text,
                              model=model,
                              tokenizer=tokenizer,
                              model_args=model_args,
                              tokenizer_args=tokenizer_args,
                              device=device)

    context_emb = None
    if context is not None:
        context_emb = encode_text_nn(text=context,
                                     model=model,
                                     tokenizer=tokenizer,
                                     model_args=model_args,
                                     tokenizer_args=tokenizer_args,
                                     device=device)

    return text_emb, context_emb
