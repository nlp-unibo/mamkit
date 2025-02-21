import abc
import logging
import os
import shutil
import tarfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from distutils.dir_util import copy_tree
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional, List, Callable, Union, Dict

import numpy as np
import pandas as pd
import simplejson as sj
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
from torch.utils.data import Dataset
from tqdm import tqdm

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
            labels
    ):
        """
        ``UnimodalDataset`` constructor.

        Args:
            inputs: list of inputs
            labels: list of labels

        """
        self.inputs = inputs
        self.labels = labels

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
        return self.inputs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class PairUnimodalDataset(Dataset):

    def __init__(
            self,
            a_inputs,
            b_inputs,
            labels
    ):
        self.a_inputs = a_inputs
        self.b_inputs = b_inputs
        self.labels = labels

    def __getitem__(
            self,
            idx
    ):
        return self.a_inputs[idx], self.b_inputs[idx], self.labels[idx]

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


class PairMultimodalDataset(Dataset):

    def __init__(
            self,
            a_texts,
            b_texts,
            a_audio,
            b_audio,
            labels
    ):
        self.a_texts = a_texts
        self.b_texts = b_texts
        self.a_audio = a_audio
        self.b_audio = b_audio
        self.labels = labels

    def __getitem__(
            self,
            idx
    ):
        return self.a_texts[idx], self.b_texts[idx], self.a_audio[idx], self.b_audio[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


MAMDataset = Union[UnimodalDataset, PairUnimodalDataset, MultimodalDataset, PairMultimodalDataset]


@dataclass
class SplitInfo:
    train: MAMDataset
    val: Optional[MAMDataset]
    test: Optional[MAMDataset]


class Loader(abc.ABC):

    def __init__(
            self,
            task_name: str,
            input_mode: InputMode,
            base_data_path: Optional[Path] = None
    ):
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

    def __init__(
            self,
            sample_rate=16000,
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

    @property
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
    The following debates from USED are missing: 13_1988, 17_1992, 42_2016, 43_2016
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
        self.archive_url = 'https://zenodo.org/api/records/11179380/files-archive'

        self.data_path = Path(self.base_data_path, self.folder_name).resolve()
        self.audio_path = self.data_path.joinpath('audio_recordings')
        self.clips_path = self.data_path.joinpath('audio_clips')
        self.dataset_path = self.data_path.joinpath('dataset.pkl')

        self.data_path.mkdir(parents=True, exist_ok=True)

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

        df['audio_paths'] = None
        for row_idx, row in tqdm(df.iterrows(), desc='Building clips...', total=df.shape[0]):
            recording_filepath = self.audio_path.joinpath(row['dialogue_id'], 'full_audio.wav')
            recording = AudioSegment.from_file(recording_filepath)

            clip_filepath = self.clips_path.joinpath(row['dialogue_id'], f'{row["text_index"]}.wav')
            clip_filepath.parent.resolve().mkdir(parents=True, exist_ok=True)

            df.loc[row_idx, 'audio_paths'] = clip_filepath

            if clip_filepath.exists():
                continue

            audio_clip = recording[row['text_start_time']:row['text_end_time']]
            audio_clip = audio_clip.set_frame_rate(self.sample_rate)
            audio_clip = audio_clip.set_channels(1)
            audio_clip.export(clip_filepath, format="wav")

        return df

    def build_clips(
            self
    ):
        dl_df = pd.read_csv(self.data_path.joinpath("download_links.csv"), sep=';')

        youtube_download(save_path=self.audio_path,
                         debate_ids=dl_df.id.values,
                         debate_urls=dl_df.link.values)

        dl_df = self.generate_clips()
        dl_df.to_pickle(self.dataset_path)

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
            logging.info('Building audio clips...')
            self.build_clips()
            logging.info('Build completed!')

            # clear
            shutil.rmtree(self.audio_path)

    def _get_text_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return UnimodalDataset(inputs=df.text.values,
                               labels=df.component.values)

    def _get_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return UnimodalDataset(inputs=df['audio_paths'].values,
                               labels=df.component.values)

    def _get_text_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return MultimodalDataset(texts=df.text.values,
                                 audio=df['audio_paths'].values,
                                 labels=df.component.values)

    @property
    def data(
            self
    ) -> pd.DataFrame:
        df = pd.read_pickle(self.dataset_path)

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
        self.archive_url = 'https://zenodo.org/api/records/11179390/files-archive'

        self.data_path = Path(self.base_data_path, self.folder_name).resolve()
        self.audio_path = self.data_path.joinpath('audio_recordings')
        self.clips_path = self.data_path.joinpath('audio_clips')
        self.dataset_path = self.data_path.joinpath('dataset.pkl')

        self.load()

        self.add_splits(method=self.get_mancini_2024_splits,
                        key='mancini-et-al-2024')

    def generate_clips(
            self
    ) -> pd.DataFrame:
        df = pd.read_pickle(self.data_path.joinpath('dataset.pkl'))

        df['dialogue_paths'] = None
        df['snippet_paths'] = None
        for row_idx, row in tqdm(df.iterrows(), desc='Building clips...', total=df.shape[0]):
            dialogue_paths = []
            snippet_paths = []

            recording_filepath = self.audio_path.joinpath(row['dialogue_id'], 'full_audio.wav')
            recording = AudioSegment.from_file(recording_filepath)

            # Dialogues
            for sent_idx, sent, start_time, end_time in zip(row['dialogue_indexes'],
                                                            row['dialogue_sentences'],
                                                            row['dialogue_start_time'],
                                                            row['dialogue_end_time']):
                clip_name = f'{sent_idx}.wav'
                clip_filepath = self.clips_path.joinpath(row['dialogue_id'], clip_name)
                clip_filepath.parent.resolve().mkdir(parents=True, exist_ok=True)

                dialogue_paths.append(clip_filepath)

                if clip_filepath.exists():
                    continue

                audio_clip = recording[start_time:end_time]
                audio_clip = audio_clip.set_frame_rate(self.sample_rate)
                audio_clip = audio_clip.set_channels(1)
                audio_clip.export(clip_filepath, format="wav")

            df.loc[row_idx, 'dialogue_paths'] = dialogue_paths

            # Snippet
            for sent_idx, sent, start_time, end_time in zip(row['snippet_indexes'],
                                                            row['snippet_sentences'],
                                                            row['snippet_start_time'],
                                                            row['snippet_end_time']):
                clip_name = f'{row["dialogue_id"]}_{sent_idx}'
                clip_filepath = self.clips_path.joinpath(row['dialogue_id'], clip_name)
                clip_filepath.parent.resolve().mkdir(parents=True, exist_ok=True)

                snippet_paths.append(clip_filepath)

                if clip_filepath.exists():
                    continue

                audio_clip = recording[start_time:end_time]
                audio_clip = audio_clip.set_frame_rate(self.sample_rate)
                audio_clip = audio_clip.set_channels(1)
                audio_clip.export(clip_filepath, format="wav")

            df.loc[row_idx, 'snippet_paths'] = snippet_paths

        return df

    def build_clips(
            self
    ):
        dl_df = pd.read_csv(self.data_path.joinpath("download_links.csv"), sep=';')
        link_df = pd.read_csv(self.data_path.joinpath('link_ids.csv'), sep=';')
        valid_debate_ids = [item for item in link_df['mm_id'].values if item != 'NOT_FOUND']

        dl_df = dl_df[dl_df.id.isin(valid_debate_ids)]

        youtube_download(save_path=self.audio_path,
                         debate_ids=dl_df.id.values,
                         debate_urls=dl_df.link.values)

        dl_df = self.generate_clips()
        dl_df.to_pickle(self.dataset_path)

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
            logging.info('Building audio clips...')
            self.build_clips()
            logging.info('Build completed!')

            # clear
            shutil.rmtree(self.audio_path)

    def _get_text_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        if self.task_name == 'afc':
            inputs = df.snippet.values if not self.with_context else (df.snippet.values, df.dialogue.values)
            labels = df.fallacy.values
        else:
            inputs = df.sentences.values if not self.with_context else (df.sentence.values, df.context.values)
            labels = df.label.values

        return UnimodalDataset(inputs=inputs,
                               labels=labels)

    def _get_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        if self.task_name == 'afc':
            inputs = df.snippet_paths.values if not self.with_context else (
            df.snippet_path.values, df.dialogue_path.values)
            labels = df.fallacy.values
        else:
            inputs = df.sentence_path.values if not self.with_context else (
            df.sentence_path.values, df.context_paths.values)
            labels = df.label.values

        return UnimodalDataset(inputs=inputs,
                               labels=labels)

    def _get_text_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        if self.task_name == 'afc':
            texts = df.snippet.values if not self.with_context else (df.snippet.values, df.dialogue.values)
            audio = df.snippet_paths.values if not self.with_context else (
            df.snippet_path.values, df.dialogue_path.values)
            labels = df.fallacy.values
        else:
            texts = df.sentences.values if not self.with_context else (df.sentence.values, df.context.values)
            audio = df.sentence_path.values if not self.with_context else (
            df.sentence_path.values, df.context_paths.values)
            labels = df.label.values

        return MultimodalDataset(texts=texts,
                                 audio=audio,
                                 labels=labels)

    def _build_afd_dataset(
            self,
            df: pd.DataFrame
    ):
        dialogue_ids = np.unique(df.dialogue_id.values)

        afd_data = []
        for dialogue_id in dialogue_ids:
            dialogue_df = df.loc[df.dialogue_id == dialogue_id]

            dialogue_sentences = [(sent_idx, sent, 0, start_time, end_time)
                                  for sent_idx, sent, start_time, end_time in zip(dialogue_df.dialogue_indexes,
                                                                                  dialogue_df.dialogue_sentences,
                                                                                  dialogue_df.dialogue_start_time,
                                                                                  dialogue_df.dialogue_end_time)]
            snippet_sentences = [(sent_idx, sent, 1, start_time, end_time)
                                 for sent_idx, sent, start_time, end_time in zip(dialogue_df.snippet_indexes,
                                                                                 dialogue_df.snippet_sentences,
                                                                                 dialogue_df.snippet_start_time,
                                                                                 dialogue_df.snippet_end_time)]

            sentences = sorted(set(dialogue_sentences + snippet_sentences), key=lambda item: item[0])
            context_window = self.context_window if self.context_window > 0 else len(sentences)

            for sent_idx, sent, label, start_time, end_time in enumerate(sentences):
                past_boundary = max(sent_idx - context_window, 0)
                past_context = sentences[past_boundary:sent_idx]

                context_audio_paths = [self.clips_path.joinpath(dialogue_id, f'{item[0]}.wav')
                                       for item in past_context]
                context_sentences = [item[1] for item in past_context]
                context_start_time = [item[3] for item in past_context]
                context_end_time = [item[4] for item in past_context]

                afd_data.append(
                    {
                        'context_paths': context_audio_paths,
                        'context_start_time': context_start_time,
                        'context_end_time': context_end_time,
                        'context_sentences': context_sentences,
                        'context': ' '.join(context_sentences),
                        'sentence': sent,
                        'sentence_path': self.clips_path.joinpath(dialogue_id, f'{sent_idx}.wav'),
                        'label': label,
                        'dialogue_id': dialogue_id,
                        'filename': dialogue_df.filename.values[0],
                        'split': dialogue_df.split.values[0]
                    }
                )

        return pd.DataFrame(afd_data)

    @property
    def data(
            self
    ) -> pd.DataFrame:
        df = pd.read_pickle(self.dataset_path)

        # afc
        if self.task_name == 'afc':
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
    ) -> List[SplitInfo]:
        dialogues = set(self.data['dialogue_id'].values)
        splits_info = []
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
            splits_info.append(split_info)

        return splits_info


class MArg(Loader):

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
            if row["relation:confidence"] > self.confidence:
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
        train_df = self._build_complete_dataset(feature_df=feature_df,
                                                aggregated_df=aggregated_df)

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

    @property
    def data(
            self
    ) -> pd.DataFrame:
        df = pd.read_csv(self.final_path)

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
