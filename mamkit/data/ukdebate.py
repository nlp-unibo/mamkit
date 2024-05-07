import logging
import tarfile
from pathlib import Path
from typing import Optional

from mamkit.data.core import Loader, UnimodalDataset, MultimodalDataset, DataInfo
from mamkit.utils import download


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
            DataInfo(train=UnimodalDataset(inputs=self.texts, labels=self.labels),
                     val=None,
                     test=None)
        if self.input_mode == 'audio-only':
            DataInfo(train=UnimodalDataset(inputs=self.audio, labels=self.labels),
                     val=None,
                     test=None)

        return DataInfo(train=MultimodalDataset(texts=self.texts, audio=self.audio, labels=self.labels),
                        val=None,
                        test=None)
