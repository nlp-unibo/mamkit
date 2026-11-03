"""Encoder and model registrations."""

from typing import List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from mamkit.configurations.keys import TOY_CLASSIFIER, TOY_TEXT_BILSTM


@register_class(key=TOY_TEXT_BILSTM, component="mamkit.components.encoders.TextBiLSTM")
class TextBiLSTMConfig(Configuration):
    """A BiLSTM stack over token embeddings."""

    embedding_dim: int = Param(16, ge=1)
    lstm_weights: List[int] = Param([8], description="Hidden size of each layer")
    dropout_rate: float = Param(0.0, ge=0.0, le=1.0)


class TextTransformerConfig(Configuration):
    model_card: str = Param(description="Hugging Face card of the text model")
    dropout_rate: float = Param(0.0, ge=0.0, le=1.0)
    is_transformer_trainable: bool = Param(False)


class TextPoolConfig(Configuration):
    embedding_dim: int = Param(ge=1)
    dropout_rate: float = Param(0.0, ge=0.0, le=1.0)


class AudioBiLSTMConfig(Configuration):
    embedding_dim: int = Param(ge=1)
    lstm_weights: List[int] = Param(description="Hidden size of each layer")
    dropout_rate: float = Param(0.0, ge=0.0, le=1.0)


class AudioTransformerConfig(Configuration):
    embedding_dim: int = Param(ge=1)
    n_heads: int = Param(ge=1)
    ffn_dim: int = Param(ge=1)
    n_layers: int = Param(1, ge=1)
    dropout_rate: float = Param(0.0, ge=0.0, le=1.0)
    encoder_dropout_rate: float = Param(0.1, ge=0.0, le=1.0)
    positional_dropout_rate: float = Param(0.1, ge=0.0, le=1.0)


class CSAConfig(Configuration):
    embedding_dim: int = Param(ge=1)
    ffn_dim: int = Param(ge=1)
    n_heads: int = Param(ge=1)
    n_layers: int = Param(1, ge=1)
    encoder_dropout_rate: float = Param(0.1, ge=0.0, le=1.0)
    text_dropout_rate: float = Param(0.1, ge=0.0, le=1.0)
    audio_dropout_rate: float = Param(0.1, ge=0.0, le=1.0)
    positional_dropout_rate: float = Param(0.1, ge=0.0, le=1.0)


class MulTAConfig(Configuration):
    embedding_dim: int = Param(ge=1)
    ffn_dim: int = Param(ge=1)
    n_blocks: int = Param(ge=1)
    n_heads: int = Param(4, ge=1, description="Attention heads of each block")
    text_dropout_rate: float = Param(0.1, ge=0.0, le=1.0)
    audio_dropout_rate: float = Param(0.1, ge=0.0, le=1.0)
    positional_dropout_rate: float = Param(0.1, ge=0.0, le=1.0)
    ffn_dropout_rate: float = Param(0.1, ge=0.0, le=1.0)


class ConcatConfig(Configuration):
    encoders: List[RegistrationKey] = Param(description="Encoders of one sample")


class EnsembleConfig(Configuration):
    text: RegistrationKey = Param(description="Text classifier")
    audio: RegistrationKey = Param(description="Audio classifier")
    lower_bound: float = Param(0.3, ge=0.0, le=1.0)
    upper_bound: float = Param(0.7, ge=0.0, le=1.0)


@register_class(key=TOY_CLASSIFIER, component="mamkit.components.models.Classifier")
class ClassifierConfig(Configuration):
    """An encoder followed by linear layers separated by ReLU."""

    encoder: RegistrationKey = Param(TOY_TEXT_BILSTM)
    head: List[int] = Param(
        [2], description="Width of each linear layer; the last is the class count"
    )
    pair: bool = Param(False, description="Encode sides a and b and concatenate")
