from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import librosa
import numpy as np
import resampy
import soundfile
import torch as th
from skimage.measure import block_reduce
from torchaudio.functional import resample


def load(path: Path):
    """Reads an audio file as a float tensor of shape (channels, frames)."""
    audio, sampling_rate = soundfile.read(path, dtype="float32", always_2d=True)
    return th.from_numpy(audio.T), sampling_rate


def check_audio(audio_input: Union[Path, List[Path]]):
    """Raises ``ValueError`` unless ``audio_input`` names existing audio files."""
    paths = [audio_input] if isinstance(audio_input, Path) else audio_input
    if not paths:
        raise ValueError(f"No audio clip for this sample: {audio_input!r}")
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise ValueError(f"Missing audio clips: {missing}")


def check_text(text: str):
    """Raises ``ValueError`` when ``text`` is missing or blank."""
    if text is None or not text.strip():
        raise ValueError(f"Every sample needs a non-empty transcript, not {text!r}")


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
        (
            spectral_centroids,
            spectral_bandwidth,
            spectral_rolloff,
            spectral_contrast,
            chroma_ft,
            mfccs,
        ),
        axis=0,
    ).transpose()

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
):
    check_audio(audio_input)
    return parse_audio_mfcc(
        audio_input=audio_input,
        mfccs=mfccs,
        normalize=normalize,
        pooling_sizes=pooling_sizes,
        remove_energy=remove_energy,
        sampling_rate=sampling_rate,
    )


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
        audio = resample(audio, audio_sampling_rate, sampling_rate)

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
    check_audio(audio_input)
    audio = parse_audio_nn(audio_input=audio_input, sampling_rate=sampling_rate)

    with th.inference_mode():
        features = processor(
            audio,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
            **processor_args,
        )
        features = features.input_values[0].to(device)
        model_features = model(features, **model_args).last_hidden_state[0].unsqueeze(0)

        if downsampling_factor is not None:
            try:
                features = th.nn.functional.interpolate(
                    model_features.permute(0, 2, 1),
                    scale_factor=downsampling_factor,
                    mode="linear",
                )
                features = features.permute(0, 2, 1)
            except RuntimeError:
                features = model_features
        else:
            features = model_features

        features = features[0].detach().cpu().numpy()

    if aggregate:
        features = np.mean(features, axis=0, keepdims=True)

    return features


def encode_text_nn(
    text: str,
    tokenizer,
    model,
    device,
    tokenizer_args: Dict = None,
    model_args: Dict = None,
):
    check_text(text)
    tokenized = tokenizer(
        [text], padding=True, return_tensors="pt", **tokenizer_args
    ).to(device)
    model_output = model(**tokenized, **model_args)
    text_emb = model_output.last_hidden_state * tokenized.attention_mask[:, :, None]
    text_emb = text_emb.detach().cpu().numpy()[0]

    return text_emb
