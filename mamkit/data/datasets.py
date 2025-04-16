import abc
import logging
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from distutils.dir_util import copy_tree
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional, List, Callable, Union, Dict, Iterable
from django.utils.functional import cached_property

import numpy as np
import pandas as pd
import simplejson as sj
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
from torch.utils.data import Dataset
from tqdm import tqdm
from itertools import chain

from mamkit.utility.data import download, youtube_download


class InputMode(Enum):
    """
    Enum class for the input modes of the dataset.

    TEXT_ONLY: only text data
    AUDIO_ONLY: only audio data
    TEXT_AUDIO: both text and audio data

    """
    TEXT_ONLY = 'text-only'
    AUDIO_ONLY = 'audio-only'
    TEXT_AUDIO = 'text-audio'


class UnimodalDataset(Dataset):
    """
    Dataset class for unimodal data.
    """

    def __init__(
            self,
            inputs,
            labels,
            context=None
    ):
        """
        Args:
            inputs: list of inputs
            labels: list of labels
            context: list of contexts, if any

        """
        self.inputs = inputs
        self.labels = labels
        self.context = context

    def __getitem__(
            self,
            idx
    ):
        """
        Get item method.

        Args:
            idx: index of the item to retrieve

        Returns:
            tuple: input, label

        """
        return self.inputs[idx], \
            self.labels[idx], \
            self.context[idx] if self.context is not None else None

    def __len__(self):
        return len(self.labels)


class PairUnimodalDataset(Dataset):

    def __init__(
            self,
            a_inputs,
            b_inputs,
            labels,
            a_context=None,
            b_context=None
    ):
        """
        Args:
            a_inputs: list of inputs corresponding to input A
            b_inputs: list of inputs corresponding to input B
            a_context: list of contexts, if any, corresponding to input A
            b_context: list of contexts, if any, corresponding to input B
            labels: list of labels

        """
        self.a_inputs = a_inputs
        self.b_inputs = b_inputs
        self.labels = labels
        self.a_context = a_context
        self.b_context = b_context

    def __getitem__(
            self,
            idx
    ):
        return self.a_inputs[idx], \
            self.b_inputs[idx], \
            self.labels[idx], \
            self.a_context[idx] if self.a_context is not None else None, \
            self.b_context[idx] if self.b_context is not None else None

    def __len__(self):
        return len(self.labels)


class MultimodalDataset(Dataset):

    def __init__(
            self,
            texts,
            audio,
            labels,
            text_context=None,
            audio_context=None
    ):
        """
        Args:
            texts: list of input texts
            audio: list of input audio filepaths
            labels: list of labels
            text_context: list of text contexts, if any
            audio_context: list of audio context, if any
        """
        self.texts = texts
        self.audio = audio
        self.labels = labels
        self.text_context = text_context
        self.audio_context = audio_context

    def __getitem__(
            self,
            idx
    ):
        return self.texts[idx], \
            self.audio[idx], \
            self.labels[idx], \
            self.text_context[idx] if self.text_context is not None else None, \
            self.audio_context[idx] if self.audio_context is not None else None

    def __len__(self):
        return len(self.labels)


class PairMultimodalDataset(Dataset):

    def __init__(
            self,
            a_texts,
            b_texts,
            a_audio,
            b_audio,
            labels,
            a_text_context=None,
            a_audio_context=None,
            b_text_context=None,
            b_audio_context=None
    ):
        """
        Args:
            a_texts: list of texts corresponding to input A
            b_texts: list of texts corresponding to input B
            a_audio: list of audio filepaths corresponding to input A
            b_audio: list of audio filepaths corresponding to input B
            labels: list of labels
            a_text_context: list of text contexts corresponding to input A
            a_audio_context: list of audio contexts corresponding to input A
            b_text_context: list of text contexts corresponding to input B
            b_audio_context: list of audio contexts corresponding to input B
        """
        self.a_texts = a_texts
        self.b_texts = b_texts
        self.a_audio = a_audio
        self.b_audio = b_audio
        self.labels = labels
        self.a_text_context = a_text_context
        self.a_audio_context = a_audio_context
        self.b_text_context = b_text_context
        self.b_audio_context = b_audio_context

    def __getitem__(
            self,
            idx
    ):
        return self.a_texts[idx], \
            self.b_texts[idx], \
            self.a_audio[idx], \
            self.b_audio[idx], \
            self.labels[idx], \
            self.a_text_context[idx] if self.a_text_context is not None else None, \
            self.a_audio_context[idx] if self.a_audio_context is not None else None, \
            self.b_text_context[idx] if self.b_text_context is not None else None, \
            self.b_audio_context[idx] if self.b_audio_context is not None else None

    def __len__(self):
        return len(self.labels)


MAMDataset = Union[UnimodalDataset, PairUnimodalDataset, MultimodalDataset, PairMultimodalDataset]


@dataclass
class SplitInfo:
    train: MAMDataset
    val: Optional[MAMDataset]
    test: Optional[MAMDataset]


class Loader(abc.ABC):
    """
    Base dataset interface
    """

    def __init__(
            self,
            task_name: str,
            input_mode: InputMode,
            base_data_path: Optional[Path] = None
    ):
        """
        Args:
            task_name: name of supported task
            input_mode: supported input mode
            base_data_path: base path where data will be stored
        """

        self.task_name = task_name
        self.input_mode = input_mode
        self.base_data_path = Path.cwd() if base_data_path is None else base_data_path

        self.splitters = {
            'default': partial(self.get_default_splits, as_iterator=True)
        }
        self.data_retriever: Dict[InputMode, Callable[[pd.DataFrame], MAMDataset]] = {
            InputMode.TEXT_ONLY: self._get_text_data,
            InputMode.AUDIO_ONLY: self._get_audio_data,
            InputMode.TEXT_AUDIO: self._get_text_audio_data
        }

    def add_splits(
            self,
            method: Callable[[], List[SplitInfo]],
            key: str
    ):
        """
        Registers the input callable method via given key as a data splitter.
        """

        if not hasattr(self, method.__name__):
            setattr(self, method.__name__, partial(method, self=self))
            self.splitters[key] = getattr(self, method.__name__)
        else:
            self.splitters[key] = method

    @abc.abstractmethod
    def _get_text_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        pass

    @abc.abstractmethod
    def _get_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        pass

    @abc.abstractmethod
    def _get_text_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        pass

    def build_info_from_splits(
            self,
            train_df,
            val_df,
            test_df
    ) -> SplitInfo:
        train_data = self.data_retriever[self.input_mode](df=train_df)
        val_data = self.data_retriever[self.input_mode](df=val_df)
        test_data = self.data_retriever[self.input_mode](df=test_df)

        return SplitInfo(train=train_data,
                         val=val_data,
                         test=test_data)

    @abc.abstractmethod
    def get_default_splits(
            self,
            as_iterator: bool = False
    ) -> Union[List[SplitInfo], SplitInfo]:
        pass

    def get_splits(
            self,
            key: str = 'default'
    ) -> List[SplitInfo]:
        return self.splitters[key]()

    @property
    @abc.abstractmethod
    def data(
            self
    ) -> pd.DataFrame:
        pass


class UKDebates(Loader):
    """
    Official loader for UKDebates dataset.
    From: Argument Mining from Speech: Detecting Claims in Political Debates
    Url: https://cdn.aaai.org/ojs/10384/10384-13-13912-1-2-20201228.pdf
    """

    def __init__(
            self,
            sample_rate: int = 16000,
            **kwargs
    ):
        super().__init__(**kwargs)

        assert self.task_name == 'asd'

        self.folder_name = 'UKDebates'
        self.sample_rate = sample_rate

        self.archive_url = 'http://argumentationmining.disi.unibo.it/dataset_aaai2016.tgz'
        self.data_path = Path(self.base_data_path, self.folder_name).resolve()
        self.audio_path = self.data_path.joinpath('dataset', 'audio')

        self.load()

        self.texts, self.audio, self.labels = self.parse_all_annotations()

        self.add_splits(method=self.get_mancini_2022_splits,
                        key='mancini-et-al-2022')

    def parse_all_annotations(
            self
    ):
        texts, audio, labels = [], [], []
        for speaker in ['Miliband', 'Clegg', 'Cameron']:
            sp_texts, sp_audio, sp_labels = self.parse_speaker_annotations(speaker=speaker)
            texts.extend(sp_texts)
            audio.extend(sp_audio)
            labels.extend(sp_labels)

        return texts, audio, labels

    def parse_speaker_annotations(
            self,
            speaker
    ):
        speaker_path = self.data_path.joinpath('dataset', f'{speaker.capitalize()}.txt')
        with speaker_path.open('r') as txt:
            texts = txt.readlines()
        texts = [' '.join(text.split(' ')[1:]) for text in texts]

        annotations_path = self.data_path.joinpath('dataset', f'{speaker.capitalize()}Labels.txt')
        with annotations_path.open('r') as txt:
            labels = txt.readlines()
            labels = [1 if label.strip() == 'C' else 0 for label in labels]

        audio = [self.audio_path.joinpath(speaker.capitalize(), f'{idx + 1}.wav') for idx in range(len(texts))]

        return texts, audio, labels

    def load(
            self
    ):
        if not self.data_path.exists():
            logging.info(f'Downloading {self.folder_name} data...')
            self.data_path.mkdir(parents=True, exist_ok=True)
            archive_path = self.data_path.parent.joinpath(f'{self.folder_name}.tar.gz')
            download(url=self.archive_url, file_path=archive_path)
            logging.info('Download completed...')

            logging.info('Extracting data...')
            with tarfile.open(archive_path) as loaded_tar:
                loaded_tar.extractall(self.data_path)

            logging.info('Extraction completed...')

            if archive_path.is_file():
                archive_path.unlink()

    def _get_text_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return UnimodalDataset(inputs=df.texts.values,
                               labels=df.labels.values)

    def _get_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return UnimodalDataset(inputs=df.audio.values,
                               labels=df.labels.values)

    def _get_text_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return MultimodalDataset(texts=df.texts.values,
                                 audio=df.audio.values,
                                 labels=df.labels.values)

    @cached_property
    def data(
            self
    ) -> pd.DataFrame:
        return pd.DataFrame.from_dict({
            'texts': self.texts,
            'audio': self.audio,
            'labels': self.labels
        })

    def get_default_splits(
            self,
            as_iterator: bool = False
    ) -> Union[List[SplitInfo], SplitInfo]:
        split_info = self.build_info_from_splits(train_df=self.data,
                                                 val_df=pd.DataFrame.empty,
                                                 test_df=pd.DataFrame.empty)
        return [split_info] if as_iterator else split_info

    def get_mancini_2022_splits(
            self
    ) -> List[SplitInfo]:
        folds_path = self.data_path.joinpath('mancini_2022_folds.json')
        if not folds_path.is_file():
            download(
                url='https://raw.githubusercontent.com/lt-nlp-lab-unibo/multimodal-am/main/deasy-speech/prebuilt_folds/aaai2016_all_folds.json',
                file_path=folds_path)

        with folds_path.open('r') as json_file:
            folds_data = sj.load(json_file)
            folds_data = sorted(folds_data.items(), key=lambda item: int(item[0].split('_')[-1]))

        split_info = []
        for _, fold in folds_data:
            train_df = self.data.iloc[fold['train']]
            val_df = self.data.iloc[fold['validation']]
            test_df = self.data.iloc[fold['test']]

            fold_info = self.build_info_from_splits(train_df=train_df, val_df=val_df, test_df=test_df)
            split_info.append(fold_info)

        return split_info


class MMUSED(Loader):
    """
    Official MM-USED loader.
    From: Multimodal Argument Mining: A Case Study in Political Debates
    Url: https://aclanthology.org/2022.argmining-1.15/
    """

    def __init__(
            self,
            sample_rate=16000,
            **kwargs
    ):
        super().__init__(**kwargs)

        assert self.task_name in ['asd', 'acc']

        self.sample_rate = sample_rate
        self.folder_name = 'MMUSED'

        # Files: download_links.csv, dataset.pkl
        self.archive_url = 'https://zenodo.org/api/records/14938592/files-archive'

        self.data_path = Path(self.base_data_path, self.folder_name).resolve()
        self.audio_path = self.data_path.joinpath('audio_recordings')
        self.clips_path = self.data_path.joinpath('audio_clips')
        self.dataset_path = self.data_path.joinpath('dataset.pkl')

        self.load()

    def generate_clips(
            self,
    ) -> pd.DataFrame:
        """
        :return: None. The function generates, for each debate, the audio clips corresponding to each sentence in the
                 dataset. The audio files are saved in 'files/audio_clips' in subfolders corresponding to each debate.
                 For each debate it creates a new dataset in which the column corresponding to the debate_ids of the clips
                 is filled with the debate_ids of the corresponding generated clip.
        """
        df = pd.read_pickle(self.data_path.joinpath('dataset.pkl'))
        dl_df = pd.read_csv(self.data_path.joinpath("download_links.csv"), sep=';')

        dialogue_ids = set(df.dialogue_id.values)

        for dialogue_id in tqdm(dialogue_ids, desc='Building clips...', total=len(dialogue_ids)):
            recording_filepath = self.audio_path.joinpath(dialogue_id, 'full_audio.wav')
            recording = AudioSegment.from_file(recording_filepath)

            debate_dl_df = dl_df[dl_df.id == dialogue_id]

            trim_start_time = debate_dl_df['startMin'].values[0] * 60 + debate_dl_df['startSec'].values[0]
            trim_end_time = debate_dl_df['endMin'].values[0] * 60 + debate_dl_df['endSec'].values[0]
            recording = recording[trim_start_time * 1000:-trim_end_time * 1000]

            dialogue_df = df[df.dialogue_id == dialogue_id]

            for row_idx, row in dialogue_df.iterrows():
                for sent_idx, start_time, end_time in zip(row['speech_indexes'],
                                                          row['speech_start_time'],
                                                          row['speech_end_time']):

                    clip_filepath = self.clips_path.joinpath(row['dialogue_id'], f'{sent_idx}.wav')
                    clip_filepath.parent.resolve().mkdir(parents=True, exist_ok=True)

                    if clip_filepath.exists():
                        continue

                    audio_clip = recording[start_time * 1000:end_time * 1000]
                    audio_clip = audio_clip.set_frame_rate(self.sample_rate)
                    audio_clip = audio_clip.set_channels(1)
                    audio_clip.export(clip_filepath, format="wav")

        return df

    def download_audio(
            self
    ):
        dl_df = pd.read_csv(self.data_path.joinpath("download_links.csv"), sep=';')
        youtube_download(save_path=self.audio_path,
                         debate_ids=dl_df.id.values,
                         debate_urls=dl_df.link.values)

    def load(
            self
    ):
        if not self.data_path.exists():
            logging.info(f'Downloading {self.folder_name} data...')
            self.data_path.mkdir(parents=True, exist_ok=True)
            tmp_path = self.data_path.joinpath('data.zip')
            download(url=self.archive_url, file_path=tmp_path)

            with zipfile.ZipFile(tmp_path, 'r') as loaded_zip:
                loaded_zip.extractall(self.data_path)

            internal_tmp_path = self.data_path.joinpath(f'{self.folder_name}.zip')
            with zipfile.ZipFile(internal_tmp_path, 'r') as loaded_zip:
                loaded_zip.extractall(self.data_path)

            tmp_path.unlink()
            internal_tmp_path.unlink()

            logging.info('Download completed!')

        if not self.clips_path.exists():
            logging.info('Downloading audio data...')
            self.download_audio()
            logging.info('Download completed!')

            logging.info('Building audio clips...')
            self.generate_clips()
            logging.info('Build completed')

            # clear
            shutil.rmtree(self.audio_path)

    def _get_text_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return UnimodalDataset(inputs=df.speech.values,
                               labels=df.component.values)

    def _get_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return UnimodalDataset(inputs=df['speech_paths'].values,
                               labels=df.component.values)

    def _get_text_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return MultimodalDataset(texts=df.speech.values,
                                 audio=df['speech_paths'].values,
                                 labels=df.component.values)

    def _build_audio_data(
            self,
            df: pd.DataFrame
    ):
        speech_paths = []
        for row_idx, row in tqdm(df.iterrows(), desc='Mapping audio data...', total=df.shape[0]):
            row_paths = []
            for sent_idx in row['speech_indexes']:
                clip_filepath = self.clips_path.joinpath(row['dialogue_id'], f'{sent_idx}.wav')
                row_paths.append(clip_filepath)
            speech_paths.append(row_paths)

        return speech_paths

    @cached_property
    def data(
            self
    ) -> pd.DataFrame:
        df = pd.read_pickle(self.dataset_path)

        speech_paths = self._build_audio_data(df=df)
        df['speech_paths'] = speech_paths

        if self.task_name == 'acc':
            df = df[df.component.isin(['Premise', 'Claim'])]
            df.loc[df.component == 'Premise', 'component'] = 0
            df.loc[df.component == 'Claim', 'component'] = 1

            # drop rows where Component is Other
            df = df[df['component'] != 'O']
        else:
            df.loc[df.component.isin(['Premise', 'Claim']), 'component'] = 'Arg'
            df.loc[df.component != 'Arg', 'component'] = 0
            df.loc[df.component == 'Arg', 'component'] = 1

        return df

    def get_default_splits(
            self,
            as_iterator: bool = False
    ) -> Union[List[SplitInfo], SplitInfo]:
        split_info = self.build_info_from_splits(train_df=self.data[self.data.split == 'TRAIN'],
                                                 val_df=self.data[self.data.split == 'VALIDATION'],
                                                 test_df=self.data[self.data.split == 'TEST'])
        return [split_info] if as_iterator else split_info


class MMUSEDFallacy(Loader):
    """
    Official MM-USEDFallacy loader.
    From: Multimodal Fallacy Classification in Political Debates
    Url: https://aclanthology.org/2024.eacl-short.16/
    """

    def __init__(
            self,
            with_context: bool = False,
            context_window: int = 3,
            sample_rate=16000,
            **kwargs
    ):
        super().__init__(**kwargs)

        assert self.task_name in ['afc', 'afd']

        self.with_context = with_context
        self.context_window = context_window
        self.sample_rate = sample_rate
        self.folder_name = 'MMUSED-fallacy'

        # Files: download_links.csv, link_ids.csv, dataset.pkl
        self.archive_url = 'https://zenodo.org/api/records/15229681/files-archive'

        self.data_path = Path(self.base_data_path, self.folder_name).resolve()
        self.audio_path = self.data_path.joinpath('audio_recordings')
        self.clips_path = self.data_path.joinpath('audio_clips')
        self.dataset_path = self.data_path.joinpath('dataset.pkl')

        self.load()

        self.add_splits(method=self.get_mancini_2024_splits,
                        key='mancini-et-al-2024')
        self.add_splits(method=self.get_mmarg_fallacy_splits,
                        key='mm-argfallacy-2025')

    def generate_clips(
            self
    ):
        df = pd.read_pickle(self.data_path.joinpath('dataset.pkl'))
        dl_df = pd.read_csv(self.data_path.joinpath("download_links.csv"), sep=';')

        dialogue_ids = set(df.dialogue_id.values)

        for dialogue_id in tqdm(dialogue_ids, desc='Building clips...', total=len(dialogue_ids)):
            recording_filepath = self.audio_path.joinpath(dialogue_id, 'full_audio.wav')
            recording = AudioSegment.from_file(recording_filepath)

            debate_dl_df = dl_df[dl_df.id == dialogue_id]

            trim_start_time = debate_dl_df['startMin'].values[0] * 60 + debate_dl_df['startSec'].values[0]
            trim_end_time = debate_dl_df['endMin'].values[0] * 60 + debate_dl_df['endSec'].values[0]
            if trim_end_time > 0:
                recording = recording[trim_start_time * 1000:-trim_end_time * 1000]

            dialogue_df = df[df.dialogue_id == dialogue_id]
            for row_idx, row in dialogue_df.iterrows():
                # Dialogues
                for sent_idx, sent, start_time, end_time in zip(row['dialogue_indexes'],
                                                                row['dialogue_sentences'],
                                                                row['dialogue_start_time'],
                                                                row['dialogue_end_time']):
                    clip_name = f'{sent_idx}.wav'
                    clip_filepath = self.clips_path.joinpath(row['dialogue_id'], clip_name)
                    clip_filepath.parent.resolve().mkdir(parents=True, exist_ok=True)

                    if clip_filepath.exists():
                        continue

                    audio_clip = recording[start_time * 1000: end_time * 1000]
                    audio_clip = audio_clip.set_frame_rate(self.sample_rate)
                    audio_clip = audio_clip.set_channels(1)
                    audio_clip.export(clip_filepath, format="wav")

                # Snippet
                for sent_idx, sent, start_time, end_time in zip(row['snippet_indexes'],
                                                                row['snippet_sentences'],
                                                                row['snippet_start_time'],
                                                                row['snippet_end_time']):
                    clip_name = f'{sent_idx}.wav'
                    clip_filepath = self.clips_path.joinpath(row['dialogue_id'], clip_name)
                    clip_filepath.parent.resolve().mkdir(parents=True, exist_ok=True)

                    if clip_filepath.exists():
                        continue

                    audio_clip = recording[start_time * 1000: end_time * 1000]
                    audio_clip = audio_clip.set_frame_rate(self.sample_rate)
                    audio_clip = audio_clip.set_channels(1)
                    audio_clip.export(clip_filepath, format="wav")

        df.to_pickle(self.dataset_path)

    def download_audio(
            self
    ):
        dl_df = pd.read_csv(self.data_path.joinpath("download_links.csv"), sep=';')
        youtube_download(save_path=self.audio_path,
                         debate_ids=dl_df.id.values,
                         debate_urls=dl_df.link.values)

    def load(
            self
    ):
        if not self.data_path.exists():
            logging.info(f'Downloading {self.folder_name} data...')
            self.data_path.mkdir(parents=True, exist_ok=True)
            tmp_path = self.data_path.joinpath('data.zip')
            download(url=self.archive_url, file_path=tmp_path)

            with zipfile.ZipFile(tmp_path, 'r') as loaded_zip:
                loaded_zip.extractall(self.data_path)

            internal_tmp_path = self.data_path.joinpath(f'{self.folder_name}.zip')
            with zipfile.ZipFile(internal_tmp_path, 'r') as loaded_zip:
                loaded_zip.extractall(self.data_path)

            tmp_path.unlink()
            internal_tmp_path.unlink()

            logging.info('Download completed!')

        if not self.clips_path.exists():
            logging.info('Downloading audio data...')
            self.download_audio()
            logging.info('Download completed!')

            logging.info('Building audio clips...')
            self.generate_clips()
            logging.info('Build completed')

            # clear
            shutil.rmtree(self.audio_path)

    def _get_text_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        if self.task_name == 'afc':
            inputs = df.snippet.values
            context = df.dialogue.values if self.with_context else None
            labels = df.fallacy.values
        else:
            inputs = df.sentence.values
            context = df.context.values if self.with_context else None
            labels = df.label.values

        return UnimodalDataset(inputs=inputs,
                               labels=labels,
                               context=context)

    def _get_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        if self.task_name == 'afc':
            inputs = df.snippet_paths.values
            context = df.dialogue_path.values if self.with_context else None
            labels = df.fallacy.values
        else:
            inputs = df.sentence_path.values
            context = df.context_paths.values if self.with_context else None
            labels = df.label.values

        return UnimodalDataset(inputs=inputs,
                               labels=labels,
                               context=context)

    def _get_text_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        if self.task_name == 'afc':
            texts = df.snippet.values
            text_context = df.dialogue.values if self.with_context else None
            audio = df.snippet_paths.values
            audio_context = df.dialogue_path.values if self.with_context else None
            labels = df.fallacy.values
        else:
            texts = df.sentence.values
            text_context = df.context.values if self.with_context else None
            audio = df.sentence_path.values
            audio_context = df.context_paths.values if self.with_context else None
            labels = df.label.values

        return MultimodalDataset(texts=texts,
                                 audio=audio,
                                 labels=labels,
                                 text_context=text_context,
                                 audio_context=audio_context)

    def _build_afd_dataset(
            self,
            df: pd.DataFrame
    ):
        dialogue_ids = np.unique(df.dialogue_id.values)

        afd_data = []
        for dialogue_id in tqdm(dialogue_ids, desc='Building AFD data...'):
            dialogue_df = df.loc[df.dialogue_id == dialogue_id]

            dialogue_sentences = [(sent_idx, sent, 0, start_time, end_time)
                                  for sent_idx, sent, start_time, end_time in zip(chain(*dialogue_df.dialogue_indexes),
                                                                                  chain(
                                                                                      *dialogue_df.dialogue_sentences),
                                                                                  chain(
                                                                                      *dialogue_df.dialogue_start_time),
                                                                                  chain(*dialogue_df.dialogue_end_time))
                                  if sent_idx not in list(chain(*dialogue_df.snippet_indexes))]
            snippet_sentences = [(sent_idx, sent, 1 if label is not None else 0, start_time, end_time)
                                 for sent_idx, sent, start_time, end_time, label in
                                 zip(chain(*dialogue_df.snippet_indexes),
                                     chain(*dialogue_df.snippet_sentences),
                                     chain(*dialogue_df.snippet_start_time),
                                     chain(*dialogue_df.snippet_end_time),
                                     chain(dialogue_df.fallacy))]

            sentences = sorted(set(dialogue_sentences + snippet_sentences), key=lambda item: item[0])
            context_window = self.context_window if self.context_window >= 0 else len(sentences)

            for rel_idx, (sent_idx, sent, label, start_time, end_time) in enumerate(sentences):
                past_boundary = max(rel_idx - context_window, 0)
                past_context = sentences[past_boundary:rel_idx]

                context_audio_paths = [self.clips_path.joinpath(dialogue_id, f'{item[0]}.wav')
                                       for item in past_context]
                context_sentences = [item[1] for item in past_context]

                afd_data.append(
                    {
                        'context_paths': context_audio_paths,
                        'context': ' '.join(context_sentences),
                        'sentence': sent,
                        'sentence_path': self.clips_path.joinpath(dialogue_id, f'{sent_idx}.wav'),
                        'label': label,
                        'dialogue_id': dialogue_id,
                        'filename': dialogue_df.filename.values[0]
                    }
                )

        return pd.DataFrame(afd_data)

    def _build_afc_context(
            self,
            df: pd.DataFrame
    ):
        info = []
        for row_idx, row in tqdm(df.iterrows(), desc='Building AFC Context', total=df.shape[0]):
            snippet_index = min(row['snippet_indexes'])

            # Past context
            context_indexes = [item_idx for item_idx, dial_idx in enumerate(row['dialogue_indexes'])
                               if dial_idx < snippet_index][-self.context_window:]

            dialogue_sentences = [item for idx, item in enumerate(row['dialogue_sentences']) if idx in context_indexes]
            dialogue = ' '.join(dialogue_sentences)
            dialogue_indexes = [item for idx, item in enumerate(row['dialogue_indexes']) if idx in context_indexes]
            dialogue_paths = [item for idx, item in enumerate(row['dialogue_paths']) if idx in context_indexes]

            info.append({
                'dialogue_indexes': dialogue_indexes,
                'dialogue_sentences': dialogue_sentences,
                'dialogue': dialogue,
                'dialogue_id': row['dialogue_id'],
                'dialogue_paths': dialogue_paths,
                'snippet_indexes': row['snippet_indexes'],
                'snippet_sentences': row['snippet_sentences'],
                'snippet': row['snippet'],
                'snippet_paths': row['snippet_paths'],
                'fallacy': row['fallacy']
            })

        return pd.DataFrame(info)

    def _build_audio_data(
            self,
            df: pd.DataFrame
    ):
        total_dialogue_paths = []
        total_snippet_paths = []
        for row_idx, row in tqdm(df.iterrows(), desc='Mapping audio data...', total=df.shape[0]):
            dialogue_paths = []
            snippet_paths = []

            # Dialogues
            for sent_idx, sent, start_time, end_time in zip(row['dialogue_indexes'],
                                                            row['dialogue_sentences'],
                                                            row['dialogue_start_time'],
                                                            row['dialogue_end_time']):
                clip_name = f'{sent_idx}.wav'
                clip_filepath = self.clips_path.joinpath(row['dialogue_id'], clip_name)
                dialogue_paths.append(clip_filepath)

            total_dialogue_paths.append(dialogue_paths)

            # Snippet
            for sent_idx, sent, start_time, end_time in zip(row['snippet_indexes'],
                                                            row['snippet_sentences'],
                                                            row['snippet_start_time'],
                                                            row['snippet_end_time']):
                clip_name = f'{sent_idx}.wav'
                clip_filepath = self.clips_path.joinpath(row['dialogue_id'], clip_name)
                snippet_paths.append(clip_filepath)

            total_snippet_paths.append(snippet_paths)

        return total_dialogue_paths, total_snippet_paths

    @cached_property
    def data(
            self
    ) -> pd.DataFrame:
        df = pd.read_pickle(self.dataset_path)

        dialogue_paths, snippet_paths = self._build_audio_data(df=df)
        df['dialogue_paths'] = dialogue_paths
        df['snippet_paths'] = snippet_paths

        # afc
        if self.task_name == 'afc':
            df = self._build_afc_context(df=df)

            df.loc[df.fallacy == 'AppealtoEmotion', 'fallacy'] = 0
            df.loc[df.fallacy == 'AppealtoAuthority', 'fallacy'] = 1
            df.loc[df.fallacy == 'AdHominem', 'fallacy'] = 2
            df.loc[df.fallacy == 'FalseCause', 'fallacy'] = 3
            df.loc[df.fallacy == 'Slipperyslope', 'fallacy'] = 4
            df.loc[df.fallacy == 'Slogans', 'fallacy'] = 5

            return df

        # afd
        return self._build_afd_dataset(df=df)

    def get_default_splits(
            self,
            as_iterator: bool = False
    ) -> Union[List[SplitInfo], SplitInfo]:
        split_info = self.build_info_from_splits(train_df=self.data,
                                                 val_df=pd.DataFrame(columns=self.data.columns),
                                                 test_df=pd.DataFrame(columns=self.data.columns))
        return [split_info] if as_iterator else split_info

    def get_mancini_2024_splits(
            self
    ) -> Iterable[SplitInfo]:
        dialogues = set(self.data['dialogue_id'].values).difference({'47_2024', '48_2024'})
        for dialogue_id in dialogues:
            train_df = self.data[self.data['dialogue_id'] != dialogue_id]
            test_df = self.data[self.data['dialogue_id'] == dialogue_id]

            train_dialogues = set(train_df['dialogue_id'].values)
            val_dialogues = np.random.choice(list(train_dialogues), size=4)

            val_df = train_df[train_df['dialogue_id'].isin(val_dialogues)]
            train_df = train_df[~train_df['dialogue_id'].isin(val_dialogues)]

            split_info = self.build_info_from_splits(train_df=train_df,
                                                     val_df=val_df,
                                                     test_df=test_df)
            yield split_info

    def get_mmarg_fallacy_splits(
            self
    ) -> Iterable[SplitInfo]:
        train_df = self.data[~self.data['dialogue_id'].isin(['47_2024', '48_2024'])]
        test_df = self.data[self.data['dialogue_id'].isin(['47_2024', '48_2024'])]

        split_info = self.build_info_from_splits(train_df=train_df,
                                                 val_df=pd.DataFrame(columns=self.data.columns),
                                                 test_df=test_df)
        return [split_info]


class MArg(Loader):
    """
    Official M-Arg loader.
    From: M-Arg: Multimodal Argument Mining Dataset for Political Debates with Audio and Transcripts
    Url: https://aclanthology.org/2021.argmining-1.8/
    """

    def __init__(
            self,
            confidence,
            **kwargs
    ):
        super().__init__(**kwargs)

        assert confidence in [0.85, 1.0]
        assert self.task_name in ['arc']

        self.confidence = confidence
        self.folder_name = 'MArg'

        self.data_path = Path(self.base_data_path, self.folder_name).resolve()
        self.feature_path = self.data_path.joinpath('data', 'preprocessed full dataset',
                                                    'full_feature_extraction_dataset.csv')
        self.aggregated_path = self.data_path.joinpath('annotated dataset', 'aggregated_dataset.csv')
        self.final_path = self.data_path.joinpath(f'final_dataset_{self.confidence:.2f}.csv')
        self.audio_path = self.data_path.joinpath('data', 'audio sentences')

        self.speakers_map = {
            'Chris Wallace': 'Chris Wallace',
            'Vice President Joe Biden': 'Joe Biden',
            'President Donald J. Trump': 'Donald Trump',
            'Chris Wallace:': 'Chris Wallace',
            'Kristen Welker': 'Kristen Welker',
            'Donald Trump': 'Donald Trump',
            'Joe Biden': 'Joe Biden',
            'George Stephanopoulos': 'George Stephanopoulos',
            'Nicholas Fed': 'Audience Member 1',
            'Kelly Lee': 'Audience Member 2',
            'Anthony Archer': 'Audience Member 3',
            'Voice Over': 'Voice Over',
            'Cedric Humphrey': 'Audience Member 4',
            'George Stephanopoulus': 'George Stephanopoulos',
            'Angelia Politarhos': 'Audience Member 5',
            'Speaker 1': 'Voice Over',
            'Nathan Osburn': 'Audience Member 6',
            'Andrew Lewis': 'Audience Member 7',
            'Speaker 2': 'Voice Over',
            'Michele Ellison': 'Audience Member 8',
            'Mark Hoffman': 'Audience Member 9',
            'Mieke Haeck': 'Audience Member 10',
            'Speaker 3': 'Voice Over',
            'Keenan Wilson': 'Audience Member 11',
            'Savannah Guthrie': 'Savannah Guthrie',
            'President Trump': 'Donald Trump',
            'Jacqueline Lugo': 'Audience Member 12',
            'Barbara Peña': 'Audience Member 13',
            'Isabella Peña': 'Audience Member 14',
            'Savannah': 'Savannah Guthrie',
            'Cristy Montesinos Alonso': 'Audience Member 15',
            'Adam Schucher': 'Audience Member 16',
            'Moriah Geene': 'Audience Member 17',
            'Cindy Velez': 'Audience Member 18',
            'Paulette Dale': 'Audience Member 19',
            'Susan Page': 'Susan Page',
            'Kamala Harris': 'Kamala Harris',
            'Mike Pence': 'Mike Pence',
            'Kamala Harris ': 'Kamala Harris'
        }
        self.speakers_to_id = {
            'error': 'error'
        }
        self.debate_id_map = {
            'us_election_2020_1st_presidential_debate_part1_timestamp.csv': '00',
            'us_election_2020_1st_presidential_debate_part2_timestamp.csv': '01',
            'us_election_2020_2nd_presidential_debate_part1_timestamp.csv': '02',
            'us_election_2020_2nd_presidential_debate_part2_timestamp.csv': '03',
            'us_election_2020_biden_town_hall_part1_timestamp.csv': '04',
            'us_election_2020_biden_town_hall_part2_timestamp.csv': '05',
            'us_election_2020_biden_town_hall_part3_timestamp.csv': '06',
            'us_election_2020_biden_town_hall_part4_timestamp.csv': '07',
            'us_election_2020_biden_town_hall_part5_timestamp.csv': '08',
            'us_election_2020_biden_town_hall_part6_timestamp.csv': '09',
            'us_election_2020_biden_town_hall_part7_timestamp.csv': '10',
            'us_election_2020_trump_town_hall_1_timestamp.csv': '11',
            'us_election_2020_trump_town_hall_2_timestamp.csv': '12',
            'us_election_2020_trump_town_hall_3_timestamp.csv': '13',
            'us_election_2020_trump_town_hall_4_timestamp.csv': '14',
            'us_election_2020_vice_presidential_debate_1_timestamp.csv': '15',
            'us_election_2020_vice_presidential_debate_2_timestamp.csv': '16'
        }
        self.file_map_timestamp = {
            'us_election_2020_1st_presidential_debate_part1_timestamp.csv': 'us_election_2020_1st_presidential_debate_split.csv',
            'us_election_2020_1st_presidential_debate_part2_timestamp.csv': 'us_election_2020_1st_presidential_debate_split.csv',
            'us_election_2020_2nd_presidential_debate_part1_timestamp.csv': 'us_election_2020_2nd_presidential_debate_split.csv',
            'us_election_2020_2nd_presidential_debate_part2_timestamp.csv': 'us_election_2020_2nd_presidential_debate_split.csv',
            'us_election_2020_biden_town_hall_part1_timestamp.csv': 'us_election_2020_biden_town_hall_split.csv',
            'us_election_2020_biden_town_hall_part2_timestamp.csv': 'us_election_2020_biden_town_hall_split.csv',
            'us_election_2020_biden_town_hall_part3_timestamp.csv': 'us_election_2020_biden_town_hall_split.csv',
            'us_election_2020_biden_town_hall_part4_timestamp.csv': 'us_election_2020_biden_town_hall_split.csv',
            'us_election_2020_biden_town_hall_part5_timestamp.csv': 'us_election_2020_biden_town_hall_split.csv',
            'us_election_2020_biden_town_hall_part6_timestamp.csv': 'us_election_2020_biden_town_hall_split.csv',
            'us_election_2020_biden_town_hall_part7_timestamp.csv': 'us_election_2020_biden_town_hall_split.csv',
            'us_election_2020_trump_town_hall_1_timestamp.csv': 'us_election_2020_trump_town_hall_split.csv',
            'us_election_2020_trump_town_hall_2_timestamp.csv': 'us_election_2020_trump_town_hall_split.csv',
            'us_election_2020_trump_town_hall_3_timestamp.csv': 'us_election_2020_trump_town_hall_split.csv',
            'us_election_2020_trump_town_hall_4_timestamp.csv': 'us_election_2020_trump_town_hall_split.csv',
            'us_election_2020_vice_presidential_debate_1_timestamp.csv': 'us_election_2020_vice_presidential_debate_split.csv',
            'us_election_2020_vice_presidential_debate_2_timestamp.csv': 'us_election_2020_vice_presidential_debate_split.csv'}

        self.data_path.mkdir(parents=True, exist_ok=True)

        if not self.final_path.exists():
            self.build()

        self.add_splits(method=self.get_mancini_2022_splits,
                        key='mancini-et-al-2022')

    def _build_complete_dataset(
            self,
            feature_df,
            aggregated_df
    ):
        df_final = pd.DataFrame(columns=['id',
                                         'relation',
                                         'confidence',
                                         'sentence_1',
                                         'sentence_2',
                                         'sentence_1_audio',
                                         'sentence_2_audio'])

        for index, row in aggregated_df.iterrows():
            # ids
            id1 = row["pair_id"]

            # labels
            relation1 = row["relation"]

            # label confidence
            conf1 = row["relation:confidence"]

            # sentences
            s1t = row["sentence_1"]
            s2t = row["sentence_2"]

            # corresponding audio sentences based on the text
            s1a = feature_df['audio_file'].loc[feature_df['text'] == s1t].values[0]
            s2a = feature_df['audio_file'].loc[feature_df['text'] == s2t].values[0]

            # If we want to filter by annotation confidence we can add here the following if statement
            if row["relation:confidence"] >= self.confidence:
                df_final = df_final._append(
                    {'id': id1,
                     'relation': relation1,
                     'confidence': conf1,
                     'sentence_1': s1t,
                     'sentence_2': s2t,
                     'sentence_1_audio': s1a,
                     'sentence_1_audio_path': self.audio_path.joinpath(s1a + '.wav'),
                     'sentence_2_audio': s2a,
                     'sentence_2_audio_path': self.audio_path.joinpath(s2a + '.wav'),
                     }, ignore_index=True)

        return df_final

    def build_chunks(
            self
    ):
        tokenized_path = self.data_path.joinpath('data', 'tokenised text')

        idx = 0
        for filepath in self.data_path.joinpath('data', 'original data').glob('*.csv'):

            speaker = []
            text = []

            df = pd.read_csv(filepath,
                             header=0,
                             names=['speaker', 'minute', 'text'])

            for r in df.index:
                # Tokenised text
                split_text = sent_tokenize(df['text'].iloc[r])
                text += split_text
                # Match speakers to their names using the dictionary above
                try:
                    speaker_name = self.speakers_map[df['speaker'].iloc[r]]
                except:  # In theory this exception is never raised
                    speaker_name = 'Speaker not found'
                speaker += [speaker_name] * len(split_text)
                # Assign IDs to speakers
                # A dictionary of speakers_to_id is filled for later use
                if not (speaker_name in self.speakers_to_id):
                    if idx < 10:
                        self.speakers_to_id[speaker_name] = '0' + str(idx)
                    else:
                        self.speakers_to_id[speaker_name] = str(idx)
                    idx = idx + 1

            df_split = pd.DataFrame({'speaker': speaker, 'text': text})
            df_split.to_csv(tokenized_path.joinpath(filepath.stem + '_split.csv'), index=False)

            # We also create a version in plan text (used by the force aligner tool)
            with open(tokenized_path.joinpath(filepath.stem + '_plain.txt'), 'w', encoding='utf8') as f:
                for t in df_split['text']:
                    f.write(t + '\n')

        filepath_topic = self.data_path.joinpath('data', 'preprocessed full dataset')
        df_topic = pd.read_csv(filepath_topic.joinpath('dataset_context_definition.csv'))
        filepath_timestamps = self.data_path.joinpath('data', 'timestamps')
        filepath_audio = self.data_path.joinpath('data', 'split audio')
        filepath_audio_save = self.data_path.joinpath('data', 'audio sentences')

        filepath_audio_save.mkdir(parents=True, exist_ok=True)

        buffer = 2

        big_df = pd.DataFrame(columns=['id', 'text', 'speaker', 'speaker_id', 'audio_file', 'context', 'debate'])

        for timestamp_name in self.debate_id_map:
            df = pd.read_csv(filepath_timestamps.joinpath(timestamp_name),
                             header=None,
                             names=['id', 'start', 'end', 'text'])

            debate_mp3 = AudioSegment.from_mp3(filepath_audio.joinpath(timestamp_name.split('_timestamp')[0] + '.mp3'))

            speak_df = pd.read_csv(tokenized_path.joinpath(self.file_map_timestamp[timestamp_name]),
                                   header=0,
                                   names=['speaker', 'text'])

            f_idx = self.debate_id_map[timestamp_name]
            idx = 0
            for index, row in df.iterrows():
                if row['start'] < buffer:
                    extract = debate_mp3[row['start'] * 1000:(row['end'] + buffer) * 1000]
                else:
                    try:
                        extract = debate_mp3[(row['start'] - buffer) * 1000:(row['end'] + buffer) * 1000]
                    except:
                        extract = debate_mp3[(row['start'] - buffer) * 1000:row['end'] * 1000]

                str_idx = str(idx)
                while len(str_idx) < 4:
                    str_idx = '0' + str_idx
                str_idx = str(f_idx) + str_idx

                idx = idx + 1
                try:
                    speaker = speak_df[speak_df['text'] == row['text']]['speaker'].iloc[0]
                except:
                    logging.info(f'Error parsing {row["text"]}')
                    speaker = 'error'

                context = df_topic[df_topic['id'] == 'n' + str_idx]['context'].values[0]
                if context == 'Ignore':
                    context = None

                timestamp = row['id']
                big_df = big_df._append({'id': 'n' + str_idx,
                                         'text': row['text'],
                                         'speaker': speaker,
                                         'speaker_id': self.speakers_to_id[speaker],
                                         'audio_file': ('a' + str_idx), 'context': context,
                                         'debate': str(f_idx), 'timestamp': timestamp},
                                        ignore_index=True)

                extract.export(filepath_audio_save.joinpath('a' + str_idx + '.wav'), format="wav")

        big_df.to_csv(filepath_topic.joinpath('full_feature_extraction_dataset.csv'), index=False)

    def build(self):
        if not any(self.data_path.iterdir()):
            logging.info('Downloading M-Arg data...')
            tmp_path = self.data_path.joinpath('data.zip')
            download(url='https://zenodo.org/api/records/5653504/files-archive',
                     file_path=tmp_path)
            logging.info('Download completed...')

            with zipfile.ZipFile(tmp_path, 'r') as loaded_zip:
                loaded_zip.extractall(self.data_path)

            tmp_path.unlink()

            dl_tmp_path = self.data_path.joinpath('rafamestre', 'm-arg_multimodal-argumentation-dataset-v1.0.0.zip')
            with zipfile.ZipFile(dl_tmp_path, 'r') as loaded_zip:
                loaded_zip.extractall(self.data_path)

            dl_tmp_path.unlink()

            source_path = self.data_path.joinpath('rafamestre-m-arg_multimodal-argumentation-dataset-851736f')
            copy_tree(source_path.as_posix(), self.data_path.as_posix())

            shutil.rmtree(source_path)

        logging.info('Building M-Arg dataset...')
        self.build_chunks()
        feature_df = pd.read_csv(self.feature_path)
        aggregated_df = pd.read_csv(self.aggregated_path)
        train_df = self._build_complete_dataset(feature_df=feature_df, aggregated_df=aggregated_df)

        # Add index for cv routine
        train_df['index'] = np.arange(train_df.shape[0])

        train_df.to_csv(self.final_path, index=False)

    def _get_text_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return PairUnimodalDataset(a_inputs=df.sentence_1.values,
                                   b_inputs=df.sentence_2.values,
                                   labels=df.relation.values)

    def _get_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return PairUnimodalDataset(a_inputs=df.sentence_1_audio_path.values,
                                   b_inputs=df.sentence_2_audio_path.values,
                                   labels=df.relation.values)

    def _get_text_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return PairMultimodalDataset(a_texts=df.sentence_1.values,
                                     b_texts=df.sentence_2.values,
                                     a_audio=df.sentence_1_audio_path.values,
                                     b_audio=df.sentence_2_audio_path.values,
                                     labels=df.relation.values)

    @cached_property
    def data(
            self
    ) -> pd.DataFrame:
        df = pd.read_csv(self.final_path)

        df['sentence_1_audio_path'] = [Path(item) for item in df['sentence_1_audio_path'].values]
        df['sentence_2_audio_path'] = [Path(item) for item in df['sentence_2_audio_path'].values]

        if self.task_name == 'arc':
            df.loc[df.relation == 'neither', 'relation'] = 0
            df.loc[df.relation == 'support', 'relation'] = 1
            df.loc[df.relation == 'attack', 'relation'] = 2

        return df

    def get_default_splits(
            self,
            as_iterator: bool = False
    ) -> Union[List[SplitInfo], SplitInfo]:
        split_info = self.build_info_from_splits(train_df=self.data,
                                                 val_df=pd.DataFrame(columns=self.data.columns),
                                                 test_df=pd.DataFrame(columns=self.data.columns))
        return [split_info] if as_iterator else split_info

    def get_mancini_2022_splits(
            self,
    ) -> List[SplitInfo]:
        folds_path = self.data_path.joinpath('mancini_2022_folds.json')
        if not folds_path.is_file():
            download(
                url=f'https://raw.githubusercontent.com/lt-nlp-lab-unibo/multimodal-am/main/deasy-speech/prebuilt_folds/m_arg_folds_{self.confidence:.2f}.json',
                file_path=folds_path)

        with folds_path.open('r') as json_file:
            folds_data = sj.load(json_file)
            folds_data = sorted(folds_data.items(), key=lambda item: int(item[0].split('_')[-1]))

        split_info = []
        for _, fold in folds_data:
            train_df = self.data.iloc[fold['train']]
            val_df = self.data.iloc[fold['validation']]
            test_df = self.data.iloc[fold['test']]

            fold_info = self.build_info_from_splits(train_df=train_df, val_df=val_df, test_df=test_df)
            split_info.append(fold_info)

        return split_info
