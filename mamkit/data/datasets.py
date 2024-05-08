import abc
import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from mamkit.utils import download


@dataclass
class DataInfo:
    train: Dataset
    val: Optional[Dataset]
    test: Optional[Dataset]


class Loader(abc.ABC):

    def __init__(
            self,
            task_name: str,
            input_mode: str
    ):
        self.task_name = task_name
        self.input_mode = input_mode
        self.texts = None
        self.audio = None
        self.labels = None

    @abc.abstractmethod
    def get_splits(
            self,
    ) -> DataInfo:
        pass


class UnimodalDataset(Dataset):

    def __init__(
            self,
            inputs,
            labels
    ):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(
            self,
            idx
    ):
        return self.inputs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class MultimodalDataset(Dataset):

    def __init__(
            self,
            texts,
            audio,
            labels
    ):
        self.texts = texts
        self.audio = audio
        self.labels = labels

    def __getitem__(
            self,
            idx
    ):
        return self.texts[idx], self.audio[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class UKDebate(Loader):

    def __init__(
            self,
            speaker: Optional[str],
            **kwargs
    ):
        super().__init__(**kwargs)

        assert self.task_name in ['asd', 'cd']
        assert (speaker is not None and speaker.casefold() in ['cameron', 'miliband', 'clegg']) or speaker is None

        self.speaker = speaker
        self.download_url = 'http://argumentationmining.disi.unibo.it/dataset_aaai2016.tgz'
        self.folder_name = 'UKDebate'
        self.filepath = Path(f'../data/{self.folder_name}').resolve()
        self.audio_path = self.filepath.joinpath('audio')

        if not self.filepath.exists():
            self.load_data()

        if speaker is not None:
            self.texts, self.audio, self.labels = self.parse_annotations(speaker=speaker)
        else:
            self.texts, self.audio, self.labels = self.parse_all_annotations()

    def parse_all_annotations(
            self
    ):
        texts, audio, labels = [], [], []
        for speaker in ['Miliband', 'Clegg', 'Cameron']:
            sp_texts, sp_audio, sp_labels = self.parse_annotations(speaker=speaker)
            texts.extend(sp_texts)
            audio.extend(sp_audio)
            labels.extend(sp_labels)

        return texts, audio, labels

    def parse_annotations(
            self,
            speaker
    ):
        speaker_path = self.filepath.joinpath('dataset', f'{speaker.capitalize()}.txt')
        with speaker_path.open('r') as txt:
            texts = txt.readlines()

        annotations_path = self.filepath.joinpath('dataset', f'{speaker.capitalize()}Labels.txt')
        with annotations_path.open('r') as txt:
            labels = txt.readlines()
            labels = [1 if label == 'C' else 0 for label in labels]

        audio = [self.audio_path.joinpath(speaker.capitalize(), f'{idx}.wav') for idx in range(len(texts))]

        return texts, audio, labels

    def load_data(
            self
    ):
        logging.getLogger(__name__).info('Downloading UKDebate dataset...')
        archive_path = self.filepath.parent.joinpath('ukdebate.tar.gz')
        download(url=self.download_url,
                 file_path=archive_path)
        logging.getLogger(__name__).info('Download completed...')

        logging.getLogger(__name__).info('Extracting data...')
        self.filepath.mkdir(parents=True)
        with tarfile.open(archive_path) as loaded_tar:
            loaded_tar.extractall(self.filepath)

        logging.getLogger(__name__).info('Extraction completed...')

        if archive_path.is_file():
            archive_path.unlink()

    def get_splits(
            self,
    ) -> DataInfo:
        if self.input_mode == 'text-only':
            return DataInfo(train=UnimodalDataset(inputs=self.texts, labels=self.labels),
                            val=None,
                            test=None)
        if self.input_mode == 'audio-only':
            return DataInfo(train=UnimodalDataset(inputs=self.audio, labels=self.labels),
                            val=None,
                            test=None)

        return DataInfo(train=MultimodalDataset(texts=self.texts, audio=self.audio, labels=self.labels),
                        val=None,
                        test=None)


# TODO: complete
# TODO: integrate EM's script for building the corpus
class MMUSED(Loader):
    pass


# TODO: complete
class MMUSEDFallacy(Loader):
    pass


# TODO: complete
class MArg(Loader):
    pass
