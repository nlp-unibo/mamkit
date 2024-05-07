import tarfile
from pathlib import Path
from urllib import request

from torch.utils.data import Dataset


# TODO: complete
# TODO: integrate EM's script for building the corpus
class UKDebate(Dataset):

    def __init__(
            self,
            task_name: str
    ):
        assert task_name in ['asd', 'cd']

    def __getitem__(
            self,
            idx
    ):
        return self.texts[idx], self.audio[idx], self.labels[idx]

    def __len__(
            self
    ):
        return len(self.texts)
