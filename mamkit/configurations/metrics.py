"""Metric registrations.

Every F1 registration names its averaging, because the torchmetrics default
for multiclass F1 is micro averaging, which equals accuracy (see #4).
"""

from typing import List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_class

from mamkit.configurations.keys import TOY_F1


class F1Config(Configuration):
    task: str = Param("multiclass", description="binary or multiclass")
    num_classes: int = Param(2, ge=2)
    average: str = Param(description="micro, macro, or weighted")


@register_class(key=TOY_F1, component="torchmetrics.F1Score")
class ToyF1Config(F1Config):
    """Macro F1 over the two toy classes."""

    average: str = Param("macro")


class BinaryF1Config(Configuration):
    """F1 of the positive class."""


class ClassSubsetF1Config(Configuration):
    """Macro F1 over ``class_subset`` only."""

    num_classes: int = Param(ge=2)
    class_subset: List[int] = Param()
