"""Featurizers turn the inputs of one modality into model inputs.

A featurizer replaces a processor and its collator. ``fit`` reads the training
inputs, ``transform`` precomputes features once per dataset, ``collate`` pads a
batch, and ``model_inputs`` returns what the model needs from the fitted
featurizer. ``fit``, ``transform``, and ``collate`` apply featurizers to a
``MAMDataset``, one featurizer per modality, with the same featurizer for
side ``a`` and side ``b``.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch as th
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor, AutoTokenizer

from mamkit.components.datasets import MAMDataset, sides
from mamkit.utility.processing import encode_audio_mfcc, encode_audio_nn, encode_text_nn
from mamkit.utility.text import basic_english, build_vocabulary, glove_vectors

DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")


def pad(tensors: List[th.Tensor]) -> Dict[str, th.Tensor]:
    """Pads tensors along their first axis with zeros and masks the padding."""
    lengths = th.tensor([len(tensor) for tensor in tensors])
    padded = pad_sequence(tensors, batch_first=True)
    mask = th.arange(padded.shape[1])[None, :] < lengths[:, None]
    return {"inputs": padded, "input_mask": mask.to(th.float32)}


class Featurizer:
    """A featurizer that fits nothing and passes its inputs through."""

    def fit(self, inputs: List, context: Optional[List] = None):
        pass

    def transform(self, inputs: List) -> List:
        return inputs

    def collate(self, inputs: List) -> Dict[str, th.Tensor]:
        """Pads feature matrices of shape [T, d]; a vector counts as one frame."""
        tensors = [th.as_tensor(np.asarray(x), dtype=th.float32) for x in inputs]
        return pad([tensor.reshape(-1, tensor.shape[-1]) for tensor in tensors])

    def model_inputs(self) -> Dict:
        return {}

    def clear(self):
        pass


class VocabularyText(Featurizer):
    """Token indices from a vocabulary fitted on the training texts.

    The vocabulary also counts the context texts. With ``embedding_model``,
    such as ``glove.6B.200d``, the model receives the matching GloVe vectors.
    """

    def __init__(self, embedding_model: Optional[str] = None, min_frequency: int = 1):
        self.embedding_model = embedding_model
        self.min_frequency = min_frequency
        self.vocabulary = None
        self.embedding_matrix = None

    def fit(self, inputs, context=None):
        logging.info("Building vocabulary...")
        texts = list(inputs) + (list(context) if context is not None else [])
        self.vocabulary = build_vocabulary(texts, basic_english, self.min_frequency)
        if self.embedding_model is not None:
            self.embedding_matrix = glove_vectors(
                self.embedding_model, self.vocabulary.get_itos()
            )

    def collate(self, inputs):
        # A missing or empty text is one padding token, as in the old TextCollator.
        ids = [self.vocabulary(basic_english(text or "")) or [0] for text in inputs]
        return pad([th.tensor(item) for item in ids])

    def model_inputs(self):
        return {
            "vocab_size": len(self.vocabulary),
            "embedding_matrix": self.embedding_matrix,
        }


class TokenizedText(Featurizer):
    """Token ids and attention masks from a pretrained tokenizer."""

    def __init__(self, model_card: str, tokenizer_args: Optional[Dict] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_card)
        self.tokenizer_args = tokenizer_args or {}

    def collate(self, inputs):
        texts = [text if text is not None else "" for text in inputs]
        tokens = self.tokenizer(
            texts, padding=True, return_tensors="pt", **self.tokenizer_args
        )
        return {"inputs": tokens["input_ids"], "input_mask": tokens["attention_mask"]}


class EncodedText(Featurizer):
    """Token embeddings from a pretrained model, computed once per dataset."""

    def __init__(
        self,
        model_card: str,
        tokenizer_args: Optional[Dict] = None,
        model_args: Optional[Dict] = None,
    ):
        self.model_card = model_card
        self.tokenizer_args = tokenizer_args or {}
        self.model_args = model_args or {}
        self.tokenizer = None
        self.model = None

    def transform(self, inputs):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
            self.model = AutoModel.from_pretrained(self.model_card).to(DEVICE)
        with th.inference_mode():
            return [
                encode_text_nn(
                    text=text,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    device=DEVICE,
                    tokenizer_args=self.tokenizer_args,
                    model_args=self.model_args,
                )
                for text in tqdm(inputs, desc="Encoding text")
            ]

    def clear(self):
        self.tokenizer = None
        self.model = None
        th.cuda.empty_cache()


class MFCCAudio(Featurizer):
    """MFCCs and spectral features of each clip.

    With ``cache_path``, computed features are stored in that file and reused
    by later runs.
    """

    def __init__(
        self,
        mfccs: int,
        sampling_rate: int = 16000,
        pooling_sizes: Optional[Iterable[int]] = None,
        remove_energy: bool = True,
        normalize: bool = True,
        cache_path: Optional[Path] = None,
    ):
        self.mfccs = mfccs
        self.sampling_rate = sampling_rate
        self.pooling_sizes = pooling_sizes
        self.remove_energy = remove_energy
        self.normalize = normalize
        self.cache_path = None if cache_path is None else Path(cache_path)
        self.cache = {}
        if self.cache_path is not None and self.cache_path.exists():
            with self.cache_path.open("rb") as file:
                self.cache = pickle.load(file)

    def transform(self, inputs):
        size = len(self.cache)
        features = [
            encode_audio_mfcc(
                audio_input=audio,
                mfccs=self.mfccs,
                sampling_rate=self.sampling_rate,
                pooling_sizes=self.pooling_sizes,
                remove_energy=self.remove_energy,
                normalize=self.normalize,
                preloaded_mfccs=self.cache,
            )
            for audio in tqdm(inputs, desc="Extracting MFCCs")
        ]
        if self.cache_path is not None and len(self.cache) != size:
            with self.cache_path.open("wb") as file:
                pickle.dump(self.cache, file)
        return features


class EncodedAudio(Featurizer):
    """Frame embeddings from a pretrained audio model, computed once per dataset.

    ``feature_extractor`` loads the model's feature extractor instead of its
    processor. ``aggregate`` averages the frames of each clip.
    """

    def __init__(
        self,
        model_card: str,
        sampling_rate: int,
        downsampling_factor: Optional[float] = None,
        aggregate: bool = False,
        feature_extractor: bool = False,
        processor_args: Optional[Dict] = None,
        model_args: Optional[Dict] = None,
    ):
        self.model_card = model_card
        self.sampling_rate = sampling_rate
        self.downsampling_factor = downsampling_factor
        self.aggregate = aggregate
        self.feature_extractor = feature_extractor
        self.processor_args = processor_args or {}
        self.model_args = model_args or {}
        self.processor = None
        self.model = None

    def transform(self, inputs):
        if self.model is None:
            loader = AutoFeatureExtractor if self.feature_extractor else AutoProcessor
            self.processor = loader.from_pretrained(self.model_card)
            self.model = AutoModel.from_pretrained(self.model_card).to(DEVICE)
        return [
            encode_audio_nn(
                audio_input=audio,
                processor=self.processor,
                model=self.model,
                device=DEVICE,
                processor_args=self.processor_args,
                model_args=self.model_args,
                sampling_rate=self.sampling_rate,
                downsampling_factor=self.downsampling_factor,
                aggregate=self.aggregate,
            )
            for audio in tqdm(inputs, desc="Encoding audio")
        ]

    def clear(self):
        self.processor = None
        self.model = None
        th.cuda.empty_cache()


def fit(dataset: MAMDataset, featurizers: Dict[str, Featurizer]):
    """Fits each featurizer on its modality, over both sides of a pair."""
    for modality, featurizer in featurizers.items():
        inputs = [x for side in sides(dataset.inputs) for x in side[modality]]
        context = None
        if dataset.context is not None:
            context = [
                x for side in sides(dataset.context) for x in side.get(modality, [])
            ]
        featurizer.fit(inputs, context or None)


def transform(dataset: MAMDataset, featurizers: Dict[str, Featurizer]) -> MAMDataset:
    """A dataset whose inputs hold the features of each featurizer."""

    def side(inputs: Dict) -> Dict:
        return {
            modality: featurizers[modality].transform(values)
            if modality in featurizers
            else values
            for modality, values in inputs.items()
        }

    inputs = dataset.inputs
    inputs = (
        {"a": side(inputs["a"]), "b": side(inputs["b"])}
        if "a" in inputs
        else side(inputs)
    )
    return MAMDataset(inputs, dataset.labels, dataset.context)


def collate(batch: List, featurizers: Dict[str, Featurizer]):
    """Collates ``(sample, label)`` pairs into a batch for the models."""
    samples, labels = zip(*batch)

    def side(values: List[Dict]) -> Dict:
        return {
            modality: featurizer.collate([value[modality] for value in values])
            for modality, featurizer in featurizers.items()
        }

    if "a" in samples[0]:
        inputs = {
            "a": side([sample["a"] for sample in samples]),
            "b": side([sample["b"] for sample in samples]),
        }
    else:
        inputs = side(samples)
    return inputs, th.tensor(labels)
