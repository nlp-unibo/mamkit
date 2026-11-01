"""Split strategies: each turns the rows of a loader into train, validation,
and test datasets.

A strategy is called with a loader from ``mamkit.components.loaders`` and
returns a list of ``Split``, one per fold. A missing validation or test set is
``None``.
"""

import json
from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np

from mamkit.components.datasets import MAMDataset
from mamkit.utility.data import download


class Split(NamedTuple):
    train: MAMDataset
    val: Optional[MAMDataset]
    test: Optional[MAMDataset]


class AllTrain:
    """One split that trains on every row, without validation or test sets."""

    def __call__(self, loader) -> List[Split]:
        return [Split(loader.dataset(loader.data), None, None)]


class ColumnSplit:
    """One split read from a column that names the set of each row."""

    def __init__(
        self,
        column: str = "split",
        train: str = "TRAIN",
        val: str = "VALIDATION",
        test: str = "TEST",
    ):
        self.column = column
        self.names = (train, val, test)

    def __call__(self, loader) -> List[Split]:
        data = loader.data
        return [
            Split(
                *(
                    loader.dataset(data[data[self.column] == name])
                    for name in self.names
                )
            )
        ]


class FoldFiles:
    """Folds read from a JSON file of row positions, downloaded from ``url``.

    The file maps fold names ending in ``_<number>`` to ``train``,
    ``validation``, and ``test`` lists. Folds follow their number. The file is
    stored in the loader folder under the name it has in ``url``.
    """

    def __init__(self, url: str):
        self.url = url

    def __call__(self, loader) -> List[Split]:
        path = Path(loader.data_path).joinpath(self.url.rsplit("/", 1)[-1])
        if not path.is_file():
            download(url=self.url, file_path=path)
        folds = json.loads(path.read_text())
        data = loader.data
        return [
            Split(
                *(
                    loader.dataset(data.iloc[fold[name]])
                    for name in ("train", "validation", "test")
                )
            )
            for _, fold in sorted(
                folds.items(), key=lambda item: int(item[0].split("_")[-1])
            )
        ]


class LeaveOneDialogueOut:
    """One fold per dialogue, which is the test set of that fold.

    Each fold draws ``validation_dialogues`` of the remaining dialogues as its
    validation set. ``excluded_dialogues`` never form a test fold, but they
    stay in the training data. Without ``seed``, the draw uses the global
    numpy generator, which the caller seeds.
    """

    def __init__(
        self,
        excluded_dialogues: List[str],
        validation_dialogues: int = 4,
        seed: Optional[int] = None,
    ):
        self.excluded_dialogues = excluded_dialogues
        self.validation_dialogues = validation_dialogues
        self.seed = seed

    def __call__(self, loader) -> List[Split]:
        data = loader.data
        generator = np.random if self.seed is None else np.random.RandomState(self.seed)
        splits = []
        for dialogue in sorted(set(data.dialogue_id) - set(self.excluded_dialogues)):
            train = data[data.dialogue_id != dialogue]
            val_dialogues = generator.choice(
                sorted(set(train.dialogue_id)),
                size=self.validation_dialogues,
                replace=False,
            )
            val = train.dialogue_id.isin(val_dialogues)
            splits.append(
                Split(
                    loader.dataset(train[~val]),
                    loader.dataset(train[val]),
                    loader.dataset(data[data.dialogue_id == dialogue]),
                )
            )
        return splits


class DialogueHoldout:
    """One split that tests on ``test_dialogues`` and trains on the others."""

    def __init__(self, test_dialogues: List[str]):
        self.test_dialogues = test_dialogues

    def __call__(self, loader) -> List[Split]:
        data = loader.data
        test = data.dialogue_id.isin(self.test_dialogues)
        return [Split(loader.dataset(data[~test]), None, loader.dataset(data[test]))]
