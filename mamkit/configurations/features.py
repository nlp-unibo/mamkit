"""Featurizer registrations."""

from typing import Optional

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
