"""Datasets whose samples hold one side or two sides named ``a`` and ``b``."""

from typing import Dict, List, Optional

from torch.utils.data import Dataset


def sides(tree: Dict) -> List[Dict]:
    """The sides of a sample tree: ``a`` and ``b`` for a pair, else the tree."""
    return [tree["a"], tree["b"]] if "a" in tree else [tree]


def item(tree: Dict, index: int) -> Dict:
    """The element ``index`` of every list in a nested dictionary of lists."""
    return {
        key: item(value, index) if isinstance(value, dict) else value[index]
        for key, value in tree.items()
    }


class MAMDataset(Dataset):
    """Columns of inputs per modality, with one label per sample.

    ``inputs`` maps each modality to a list, such as
    ``{"text": [...], "audio": [...]}``. A pair dataset maps ``a`` and ``b`` to
    such dictionaries. ``context`` has the same layout and holds the context of
    each input, if any. Only the vocabulary of ``VocabularyText`` reads it.
    """

    def __init__(self, inputs: Dict, labels: List, context: Optional[Dict] = None):
        self.inputs = inputs
        self.labels = list(labels)
        self.context = context

    def __getitem__(self, index: int):
        return item(self.inputs, index), self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)
