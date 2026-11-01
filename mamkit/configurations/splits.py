"""Split strategy registrations, one per published split of each corpus."""

from typing import List, Optional

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_class, register_method

from mamkit.configurations.keys import (
    ALL_TRAIN_SPLIT,
    MARG_085_MANCINI_2022_SPLIT,
    MMUSED_FALLACY_MANCINI_2024_SPLIT,
    MMUSED_FALLACY_MM_ARGFALLACY_2025_SPLIT,
    MMUSED_SPLIT,
    UKDEBATES_MANCINI_2022_SPLIT,
)

FOLDS_URL = (
    "https://raw.githubusercontent.com/lt-nlp-lab-unibo/multimodal-am/main/"
    "deasy-speech/prebuilt_folds/"
)
#: The MM-USED-fallacy dialogues of the MM-ArgFallacy 2025 shared task test set.
SHARED_TASK_DIALOGUES = ["47_2024", "48_2024"]


@register_class(key=ALL_TRAIN_SPLIT, component="mamkit.components.splits.AllTrain")
class AllTrainConfig(Configuration):
    """Trains on every row, without validation or test sets."""


@register_class(key=MMUSED_SPLIT, component="mamkit.components.splits.ColumnSplit")
class ColumnSplitConfig(Configuration):
    """The train, validation, and test sets that MM-USED ships."""

    column: str = Param("split", description="Column naming the set of each row")
    train: str = Param("TRAIN")
    val: str = Param("VALIDATION")
    test: str = Param("TEST")


class FoldFilesConfig(Configuration):
    """Folds of Mancini et al. (2022), published as JSON files."""

    url: str = Param(description="URL of the JSON file of fold row positions")

    @classmethod
    @register_method(
        key=UKDEBATES_MANCINI_2022_SPLIT,
        component="mamkit.components.splits.FoldFiles",
    )
    def ukdebates(cls):
        return cls(url=FOLDS_URL + "aaai2016_all_folds.json")

    # The repository publishes M-Arg folds for confidence 0.85 only.
    @classmethod
    @register_method(
        key=MARG_085_MANCINI_2022_SPLIT,
        component="mamkit.components.splits.FoldFiles",
    )
    def marg_085(cls):
        return cls(url=FOLDS_URL + "m_arg_folds_0.85.json")


@register_class(
    key=MMUSED_FALLACY_MANCINI_2024_SPLIT,
    component="mamkit.components.splits.LeaveOneDialogueOut",
)
class LeaveOneDialogueOutConfig(Configuration):
    """Leave-one-dialogue-out folds of Mancini et al. (2024)."""

    excluded_dialogues: List[str] = Param(SHARED_TASK_DIALOGUES)
    validation_dialogues: int = Param(4, ge=1)
    seed: Optional[int] = Param(
        None, description="Seed of the validation draw; None uses the global seed"
    )


@register_class(
    key=MMUSED_FALLACY_MM_ARGFALLACY_2025_SPLIT,
    component="mamkit.components.splits.DialogueHoldout",
)
class DialogueHoldoutConfig(Configuration):
    """The train and test sets of the MM-ArgFallacy 2025 shared task."""

    test_dialogues: List[str] = Param(SHARED_TASK_DIALOGUES)
