import abc
import logging
import tarfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List, Callable

import pandas as pd
import simplejson as sj
from torch.utils.data import Dataset

from mamkit.utils import download


class InputMode(Enum):
    TEXT_ONLY = 'text-only'
    AUDIO_ONLY = 'audio-only'
    TEXT_AUDIO = 'text-audio'


@dataclass
class SplitInfo:
    train: Dataset
    val: Optional[Dataset]
    test: Optional[Dataset]


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


class Loader(abc.ABC):

    def __init__(
            self,
            task_name: str,
            input_mode: InputMode
    ):
        self.task_name = task_name
        self.input_mode = input_mode

        self.texts = None
        self.audio = None
        self.labels = None
        self.splits = None

        self.splitters = {
            'default': self.get_default_splits
        }

    def add_splits(
            self,
            method: Callable[[pd.DataFrame], List[SplitInfo]],
            key: str
    ):
        self.splitters[key] = method

    def build_info_from_splits(
            self,
            train_df,
            val_df,
            test_df
    ) -> SplitInfo:
        if self.input_mode == InputMode.TEXT_ONLY:
            return SplitInfo(train=UnimodalDataset(inputs=train_df.texts.values, labels=train_df.labels.values),
                             val=UnimodalDataset(inputs=val_df.texts.values, labels=val_df.labels.values),
                             test=UnimodalDataset(inputs=test_df.texts.values, labels=test_df.labels.values))
        if self.input_mode == InputMode.AUDIO_ONLY:
            return SplitInfo(train=UnimodalDataset(inputs=train_df.audio.values, labels=train_df.labels.values),
                             val=UnimodalDataset(inputs=val_df.audio.values, labels=val_df.labels.values),
                             test=UnimodalDataset(inputs=test_df.audio.values, labels=test_df.labels.values))

        return SplitInfo(train=MultimodalDataset(texts=train_df.texts.values, audio=train_df.audio.values, labels=train_df.labels.values),
                         val=MultimodalDataset(texts=val_df.texts.values, audio=val_df.audio.values, labels=val_df.labels.values),
                         test=MultimodalDataset(texts=test_df.texts.values, audio=test_df.audio.values, labels=test_df.labels.values))

    def get_default_splits(
            self,
            data: pd.DataFrame
    ) -> List[SplitInfo]:
        train_df = data[data.split == 'train']
        val_df = data[data.split == 'val']
        test_df = data[data.split == 'test']

        return [self.build_info_from_splits(train_df=train_df, val_df=val_df, test_df=test_df)]

    def get_splits(
            self,
            key: str = 'default'
    ) -> List[SplitInfo]:
        return self.splitters[key](self.data)

    @property
    def data(
            self
    ) -> pd.DataFrame:
        return pd.DataFrame.from_dict({'texts': self.texts,
                                       'audio': self.audio,
                                       'labels': self.labels,
                                       'split': self.splits})


class UKDebate(Loader):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        assert self.task_name in ['asd', 'cd']

        self.download_url = 'http://argumentationmining.disi.unibo.it/dataset_aaai2016.tgz'
        self.folder_name = 'UKDebate'
        self.filepath = Path(Path.cwd().parent, 'data', self.folder_name).resolve()
        self.audio_path = self.filepath.joinpath('audio')

        if not self.filepath.exists():
            self.load()

        self.texts, self.audio, self.labels, self.splits = self.parse_all_annotations()

        self.add_splits(method=self.get_mancini_2022_splits,
                        key='mancini-et-al-2022')

    def parse_all_annotations(
            self
    ):
        texts, audio, labels, splits = [], [], [], []
        for speaker in ['Miliband', 'Clegg', 'Cameron']:
            sp_texts, sp_audio, sp_labels, sp_splits = self.parse_speaker_annotations(speaker=speaker)
            texts.extend(sp_texts)
            audio.extend(sp_audio)
            labels.extend(sp_labels)
            splits.extend(sp_splits)

        return texts, audio, labels, splits

    def parse_speaker_annotations(
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

        return texts, audio, labels, ['train'] * len(texts)

    def load(
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

    def get_mancini_2022_splits(
            self,
            data: pd.DataFrame
    ) -> List[SplitInfo]:
        folds_path = Path(Path.cwd().parent, 'data', 'UKDebate', 'mancini_2022_folds.json').resolve()
        if not folds_path.is_file():
            download(
                url='https://raw.githubusercontent.com/lt-nlp-lab-unibo/multimodal-am/main/deasy-speech/prebuilt_folds/aaai2016_all_folds.json',
                file_path=folds_path)

        with folds_path.open('r') as json_file:
            folds_data = sj.load(json_file)
            folds_data = sorted(folds_data.items(), key=lambda item: int(item[0].split('_')[-1]))

        split_info = []
        for _, fold in folds_data:
            train_df = data.iloc[fold['train']]
            val_df = data.iloc[fold['validation']]
            test_df = data.iloc[fold['test']]

            fold_info = self.build_info_from_splits(train_df=train_df, val_df=val_df, test_df=test_df)
            split_info.append(fold_info)

        return split_info


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
