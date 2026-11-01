"""Task registrations."""

from typing import Any, Dict, List, Optional

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from mamkit.configurations.keys import (
    COLUMN_SPLIT,
    TOY_CLASSIFIER,
    TOY_F1,
    TOY_LOADER,
    TOY_TASK,
    VOCABULARY_TEXT,
)


class TaskConfig(Configuration):
    """Fields every task shares."""

    loader: RegistrationKey = Param(description="The corpus")
    split: RegistrationKey = Param(description="The split strategy")
    featurizers: Dict[str, RegistrationKey] = Param(
        description="One featurizer per modality, keyed text or audio"
    )
    model: RegistrationKey = Param()
    metrics: Dict[str, RegistrationKey] = Param()
    seeds: List[int] = Param()
    loss: str = Param("torch.nn.CrossEntropyLoss", description="Import path")
    class_weights: Optional[List[float]] = Param(None)
    optimizer: str = Param("torch.optim.Adam", description="Import path")
    optimizer_args: Dict[str, Any] = Param({})
    batch_size: int = Param(8, ge=1)
    patience: int = Param(5, ge=1, description="Early stopping patience in epochs")
    trainer_args: Dict[str, Any] = Param({})
    save_path: Optional[str] = Param(None)


@register_class(
    key=TOY_TASK, component="mamkit.components.tasks.Task", run_method="run"
)
class ToyTaskConfig(TaskConfig):
    """A text BiLSTM on the toy corpus, for smoke tests."""

    loader: RegistrationKey = Param(TOY_LOADER)
    split: RegistrationKey = Param(COLUMN_SPLIT)
    featurizers: Dict[str, RegistrationKey] = Param({"text": VOCABULARY_TEXT})
    model: RegistrationKey = Param(TOY_CLASSIFIER)
    metrics: Dict[str, RegistrationKey] = Param({"f1": TOY_F1})
    seeds: List[int] = Param([42])
    optimizer_args: Dict[str, Any] = Param({"lr": 0.02})
    trainer_args: Dict[str, Any] = Param(
        {"accelerator": "cpu", "max_epochs": 10, "logger": False}
    )
