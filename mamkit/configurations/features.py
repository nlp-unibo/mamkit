"""Featurizer registrations."""

from typing import Any, Dict, List, Optional

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_class

from mamkit.configurations.keys import VOCABULARY_TEXT


@register_class(
    key=VOCABULARY_TEXT, component="mamkit.components.features.VocabularyText"
)
class VocabularyTextConfig(Configuration):
    """A vocabulary fitted on the training texts, without pretrained vectors."""

    embedding_model: Optional[str] = Param(
        None, description="GloVe vectors such as glove.6B.200d, if any"
    )
    min_frequency: int = Param(1, ge=1)


class TokenizedTextConfig(Configuration):
    model_card: str = Param(description="Hugging Face card of the tokenizer")
    tokenizer_args: Optional[Dict[str, Any]] = Param(None)


class EncodedTextConfig(Configuration):
    model_card: str = Param(description="Hugging Face card of the text model")
    tokenizer_args: Optional[Dict[str, Any]] = Param(None)
    model_args: Optional[Dict[str, Any]] = Param(None)


class MFCCAudioConfig(Configuration):
    mfccs: int = Param(ge=1)
    sampling_rate: int = Param(16000, ge=1)
    pooling_sizes: Optional[List[int]] = Param(None)
    remove_energy: bool = Param(True)
    normalize: bool = Param(True)
    cache_path: Optional[str] = Param(None, description="File that caches features")


class EncodedAudioConfig(Configuration):
    model_card: str = Param(description="Hugging Face card of the audio model")
    sampling_rate: int = Param(16000, ge=1)
    downsampling_factor: Optional[float] = Param(None)
    aggregate: bool = Param(False, description="Average the frames of each clip")
    processor_args: Optional[Dict[str, Any]] = Param(None)
    model_args: Optional[Dict[str, Any]] = Param(None)
