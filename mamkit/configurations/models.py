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


@register_class(key=TOY_CLASSIFIER, component="mamkit.components.models.Classifier")
class ClassifierConfig(Configuration):
    """An encoder followed by linear layers separated by ReLU."""

    encoder: RegistrationKey = Param(TOY_TEXT_BILSTM)
    head: List[int] = Param(
        [2], description="Width of each linear layer; the last is the class count"
    )
    pair: bool = Param(False, description="Encode sides a and b and concatenate")
