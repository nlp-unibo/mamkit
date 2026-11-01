"""Metric registrations of the published experiments.

Generated from the legacy configurations in ``mamkit.configs`` and the demos;
``tests/test_registrations.py`` checks them against ``tests/golden/configs.json``.
"""

from cinnamon.registry import register_method

from mamkit.configurations.keys import (
    MACRO_F1_2_CLASSES,
    MACRO_F1_6_CLASSES,
    MARG_F1,
    UKDEBATES_F1,
)
from mamkit.configurations.metrics import BinaryF1Config, ClassSubsetF1Config, F1Config


class MetricRegistrations:
    @classmethod
    @register_method(
        key=MARG_F1, component="mamkit.utility.metrics.ClassSubsetMulticlassF1Score"
    )
    def marg_f1(cls):
        config = ClassSubsetF1Config(
            num_classes=3,
            class_subset=[1, 2],
        )
        return config

    @classmethod
    @register_method(key=MACRO_F1_2_CLASSES, component="torchmetrics.F1Score")
    def macro_f1_2_classes(cls):
        config = F1Config(
            task="multiclass",
            num_classes=2,
            average="macro",
        )
        return config

    @classmethod
    @register_method(key=MACRO_F1_6_CLASSES, component="torchmetrics.F1Score")
    def macro_f1_6_classes(cls):
        config = F1Config(
            task="multiclass",
            num_classes=6,
            average="macro",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_F1, component="torchmetrics.classification.BinaryF1Score"
    )
    def ukdebates_f1(cls):
        config = BinaryF1Config()
        return config
