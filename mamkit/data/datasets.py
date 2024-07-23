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
            **kwargs
    ):
        super().__init__(**kwargs)

        assert self.task_name == 'asd'

        self.download_url = 'http://argumentationmining.disi.unibo.it/dataset_aaai2016.tgz'
        self.folder_name = 'UKDebates'
        self.data_path = Path(self.base_data_path, self.folder_name).resolve()
        self.audio_path = self.data_path.joinpath('dataset', 'audio')

        if not self.data_path.exists():
            self.load()
        else:
            logging.info(f'Found existing data at {self.data_path}')

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
        logging.info('Downloading UKDebates dataset...')
        self.data_path.mkdir(parents=True)
        archive_path = self.data_path.parent.joinpath('ukdebate.tar.gz')
        download(url=self.download_url,
                 file_path=archive_path)
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

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        assert self.task_name in ['asd', 'acc']

        self.folder_name = 'MMUSED'
        self.deleted_ids = ['13_1988, 17_1992, 42_2016, 43_2016']

        self.data_path = Path(self.base_data_path, self.folder_name).resolve()
        self.files_path = self.data_path.joinpath('files')
        self.aeneas_path = self.data_path.joinpath('aeneas')
        self.audio_path = self.files_path.joinpath('debates_audio_recordings')
        self.datasets_path = self.files_path.joinpath('datasets')
        self.transcripts_path = self.files_path.joinpath('transcripts')
        self.alignments_path = self.files_path.joinpath('alignment_results')
        self.clips_path = self.files_path.joinpath('audio_clips')
        self.final_path = self.files_path.joinpath('MM-USElecDeb60to16')

        self.data_path.mkdir(parents=True, exist_ok=True)

        self.load()

    def trim_audio(
            self,
            debate_ids: List,
            start_min: List,
            start_sec: List,
            end_min: List,
            end_sec: List
    ) -> None:
        """

        :param debate_ids: list of strings representing debates IDs
        :param start_min: list of strings representing the number of minutes to be cut from the beginning of the file
        :param start_sec: list of strings representing the number of seconds to be cut from the beginning of the file
        :param end_min: list of strings representing the number of minutes to be cut from the end of the file
        :param end_sec: list of strings representing the number of seconds to be cut from the end of the file
        :return None: None. The function removes from the original audio file the portions of audio corresponding
                          to the specified seconds and minutes and saves a new version of the file '_trim.wav' in
                          'files/debates_audio_recordings' (in the corresponding debate's sub folder).
        """
        for idx in tqdm(range(len(debate_ids)), desc='Trimming audio files...'):
            db_folder_id = debate_ids[idx]

            export_filename = self.audio_path.joinpath(db_folder_id, 'full_audio_trim.wav')
            import_filename = self.audio_path.joinpath(db_folder_id, 'full_audio.wav')

            # importing file from location by giving its path
            sound = AudioSegment.from_file(import_filename)

            # Selecting Portion we want to cut
            db_start_min = start_min[idx]
            db_start_sec = start_sec[idx]
            duration = sound.duration_seconds
            db_end_min, db_end_sec = divmod(duration, 60)
            db_end_min = db_end_min - end_min[idx]
            db_end_sec = db_end_sec - end_sec[idx]

            # Time to milliseconds conversion
            db_start_time = db_start_min * 60 * 1000 + db_start_sec * 1000
            db_end_time = db_end_min * 60 * 1000 + db_end_sec * 1000
            # print(EndTime)

            # Opening file and extracting portion of it
            extract = sound[db_start_time:db_end_time]
            # Saving file in required location
            extract.export(export_filename, format="wav")  # wav conversion is faster than mp3 conversion

            # clear
            import_filename.unlink()

    def copy_transcripts(
            self,
            debate_ids: List
    ):
        """

        :param debate_ids: list of strings representing debates IDs
        :return: None. The function copies transcripts from the original dataset folder to the 'files/transcripts' folder.
        """
        for el in debate_ids:
            if el not in self.deleted_ids:
                current_path = self.datasets_path.joinpath('YesWeCan', 'ElecDeb60to16', f'{el}.txt')
                transcript_folder = self.transcripts_path.joinpath(el)
                transcript_folder.mkdir(parents=True, exist_ok=False)
                shutil.copy(current_path, transcript_folder)

    def create_plain_text(
            self,
            debate_ids: List
    ) -> None:
        """

        :param debate_ids: list of strings representing debates IDs
        :return: None. The function creates the plain version of each transcript, saving a new version '_plain.txt'
                 in the subdirectory of the corresponding debate. The function creates the plain version of each
                 transcript, saving a new version '_plain.txt' in the subdirectory of the corresponding debate.
                 In the plain version, speaker information is removed and the text is tokenized by sentences.
                 The plain text thus contains one sentence per line.
        """
        for i in range(len(debate_ids)):

            folder_id = debate_ids[i]
            filename = self.transcripts_path.joinpath(folder_id, f'{folder_id}.txt')
            new_filename = self.transcripts_path.joinpath(folder_id, f'{folder_id}_plain.txt')

            with filename.open('r') as f:
                lines = f.readlines()

            new_text = []
            for line in lines:
                new_line = line[line.index(':') + 2:]  # remove speaker
                sentences = sent_tokenize(new_line)
                for s in sentences:
                    new_text.append(s)

            with new_filename.open('w') as f:
                f.write(os.linesep.join(new_text))

    def generate_chunks(
            self,
            debate_ids: List
    ) -> None:
        """

        :param debate_ids: list of strings representing debates IDs
        :return: None. The function generates the 20-minute chunks for each debate and saves them in the 'split'
                 sub-folders of each debate in 'files/debates_audio_recordings'
        """
        for debate_id in tqdm(range(len(debate_ids)), desc='Generating audio chunks...'):
            folder_id = debate_ids[debate_id]
            filename = self.audio_path.joinpath(folder_id, 'full_audio_trim.wav')
            chunks_folder = self.audio_path.joinpath(folder_id, 'splits')
            chunks_folder.mkdir(parents=True, exist_ok=True)

            sound = AudioSegment.from_mp3(filename)
            duration = sound.duration_seconds
            cut_sec = 1200
            n_chunks = round(duration / cut_sec)  # we split files in chunks of 20 mins

            for chunk_id in range(n_chunks):
                start_sec = cut_sec * chunk_id
                end_sec = cut_sec * (chunk_id + 1)
                if chunk_id == n_chunks - 1:
                    end_sec = duration
                start_time = start_sec * 1000
                end_time = end_sec * 1000
                extract = sound[start_time:end_time]
                chunk_filename = chunks_folder.joinpath(f'split_{chunk_id}.wav')

                # Saving file in required location
                extract.export(chunk_filename, format="wav")

    def generate_empty_transcript_files(
            self,
            debate_ids: List
    ) -> None:
        """

        :param debate_ids: list of strings representing debates IDs
        :return: None. The function generates as many empty '.txt' files as there are chunks generated for each debate and
                 saves them in the 'splits' subdirectory of each debate in the 'files/transcripts' folder
        """
        for debate_id in range(len(debate_ids)):
            folder_id = debate_ids[debate_id]
            split_transcripts_path = self.transcripts_path.joinpath(folder_id, 'splits')
            splits_audio_path = self.audio_path.joinpath(folder_id, 'splits')
            split_transcripts_path.mkdir(parents=True, exist_ok=False)
            for filename in splits_audio_path.glob('*'):
                if filename.stem != '.DS_Store':  # MacOS hidden files check
                    txt_filename = filename.with_suffix('.txt')
                    txt_filename.open().close()

    def run_aeneas(
            self,
            debate_ids: List
    ) -> None:
        """

        :param debate_ids: list of strings representing debates IDs
        :return: None. For each debate it executes the script to perform the alignment of audio and text.
                 The '.json' files resulting from the alignment come in 'files/alignment_results'.
                 A subfolder for each debate.
        """

        for i in tqdm(range(len(debate_ids)), desc='Alignment w/ Aeneas'):
            folder_id = debate_ids[i]

            split_transcripts_path = self.transcripts_path.joinpath(folder_id, 'splits')
            splits_audio_path = self.audio_path.joinpath(folder_id, 'splits')
            dest_clip_folder = self.alignments_path.joinpath(folder_id)
            dest_clip_folder.mkdir(parents=True)
            for filename in splits_audio_path.glob('*'):
                if filename.stem != '.DS_Store':
                    txt_filename = filename.with_suffix('.txt')
                    split_text_path = split_transcripts_path.joinpath(txt_filename.name)
                    shutil.copy(split_text_path, self.aeneas_path.joinpath(split_text_path.name))
                    shutil.copy(filename, self.aeneas_path.joinpath(filename.name))

            current_path = Path.cwd()
            os.chdir(self.aeneas_path)
            os.system('./run.sh')

            for filename in self.aeneas_path.glob('*.json'):
                shutil.move(filename, dest_clip_folder.joinpath(filename.name))

            for filename in self.aeneas_path.glob('**/*'):
                if filename.suffix in ['.wav', 'txt']:
                    filename.unlink()

            os.chdir(current_path)

    def generate_dataset(
            self,
            debate_ids: List
    ):
        """

        :param debate_ids: list of strings representing debates IDs
        :return: None. The function generates a new dataset '.csv' for each debate from the original dataset.
                Each new dataset contains 3 new columns corresponding to the new start and end timestamps calculated
                through the alignment with 'aeneas' and the debate_ids of the clip corresponding to each sentence.
                The function also saves a 'duplicates.txt' file for each debate, containing the duplicated
                sentences and the number of occurrences.
        """

        for debate_id in tqdm(range(len(debate_ids)), desc='Generating dataset...'):
            folder_id = debate_ids[debate_id]

            directory_alignments = self.alignments_path.joinpath(folder_id)
            full_dataset_path = self.datasets_path.joinpath('YesWeCan', 'sentence_db_candidate.csv')
            new_files_path = self.datasets_path.joinpath(folder_id)
            new_files_path.mkdir(parents=True)

            df_debates = pd.read_csv(full_dataset_path)

            # count_rows of debate
            count_row_debate = 0
            for i, row in df_debates.iterrows():
                if row['Document'] == folder_id:
                    count_row_debate += 1

            # generate new dataframe
            rows_new_df = []
            for i, row in df_debates.iterrows():
                if row['Document'] == folder_id:
                    rows_new_df.append(row)
            new_df = pd.DataFrame(rows_new_df)

            # set new datasets columns
            new_col_begin = ['NOT_FOUND' for i in range(count_row_debate)]
            new_col_end = ['NOT_FOUND' for i in range(count_row_debate)]
            new_col_id = ['NOT_FOUND' for i in range(count_row_debate)]
            new_df['NewBegin'] = new_col_begin
            new_df['NewEnd'] = new_col_end
            new_df['idClip'] = new_col_id

            count_matches = 0
            matches = []
            count_matches_no_duplicates = 0
            for filepath in directory_alignments.glob('*.json'):
                df = pd.read_json(filepath, orient=str)
                filename = filepath.stem.split('_')
                split_index = float(filename[-1])
                mul_factor = split_index * 1200.00
                for j, r in tqdm(df.iterrows(), total=df.shape[0], position=0):
                    for i, row in new_df.iterrows():
                        if row['Speech'].strip() == r.fragments['lines'][0].strip():
                            # print(r.fragments['lines'][0].strip())
                            new_df.at[i, "NewBegin"] = round(float(r.fragments['begin']) + mul_factor, 3)
                            # print(round(float(r.fragments['begin'])+mul_factor,3))
                            new_df.at[i, "NewEnd"] = round(float(r.fragments['end']) + mul_factor, 3)
                            if row['Speech'].strip() not in matches:
                                count_matches_no_duplicates += 1
                            count_matches += 1
                            matches.append(row['Speech'].strip())

            a = dict(Counter(matches))
            for k, v in a.items():
                if v > 1:
                    print(k, v)

            # save csv
            new_df.to_csv(new_files_path.joinpath('dataset.csv'))

            # save files of duplicates for future removal of those lines from the dataset
            with open(new_files_path.joinpath('duplicates.txt'), 'w') as f:
                for k, v in a.items():
                    if v > 1:
                        line = k + ' : ' + str(v) + '\n'
                        f.write(line)

    def generate_clips(
            self,
            debate_ids: List
    ):
        """

        :param debate_ids: list of strings representing debates IDs
        :return: None. The function generates, for each debate, the audio clips corresponding to each sentence in the
                 dataset. The audio files are saved in 'files/audio_clips' in subfolders corresponding to each debate.
                 For each debate it creates a new dataset in which the column corresponding to the debate_ids of the clips
                 is filled with the debate_ids of the corresponding generated clip.
        """

        for debate_id in tqdm(range(len(debate_ids)), desc='Generating clips...'):
            folder_id = debate_ids[debate_id]
            dataset_path = self.datasets_path.joinpath(folder_id, 'dataset.csv')
            dataset_clip_path = self.datasets_path.joinpath(folder_id, 'dataset_clip.csv')
            full_audio_path = self.audio_path.joinpath(folder_id, 'full_audio_trim.wav')
            audio_clips_path = self.clips_path.joinpath(folder_id)
            audio_clips_path.mkdir(parents=True, exist_ok=True)

            # read dataframe with timestamps
            df = pd.read_csv(dataset_path)

            # generate clips
            sound = AudioSegment.from_file(full_audio_path)
            for i, row in df.iterrows():
                start_time = row['NewBegin']
                idClip = 'clip_' + str(i)
                if start_time != 'NOT_FOUND':
                    start_time = float(row['NewBegin']) * 1000  # sec -> ms conversion
                    end_time = float(row['NewEnd']) * 1000  # sec -> ms conversion
                    clip_name = audio_clips_path.joinpath(f'{idClip}.wav')
                    extract = sound[start_time:end_time]
                    extract.export(clip_name, format="wav")
                    df.at[i, "idClip"] = idClip

            # clear
            full_audio_path.unlink()

            # save new csv
            df.to_csv(dataset_clip_path)

        # clear
        shutil.rmtree(self.audio_path)

    def remove_duplicates(
            self,
            debate_ids: List
    ):
        """

        :param debate_ids: list of strings representing debates IDs
        :return: None. The function removes duplicates in the dataset
        """
        for debate_id in tqdm(range(len(debate_ids)), desc='Removing duplicates...'):
            folder_id = debate_ids[debate_id]
            dataset_clip_path = self.datasets_path.joinpath(folder_id, 'dataset_clip.csv')
            duplicates_file_path = self.datasets_path.joinpath(folder_id, 'duplicates.txt')
            dataset_no_dup_path = self.datasets_path.joinpath(folder_id, 'dataset_clip_nodup.csv')
            df = pd.read_csv(dataset_clip_path)
            df['Component'] = df['Component'].fillna('999')

            with duplicates_file_path.open('r') as f:
                lines = f.readlines()
            lines_sentences = []

            # get only text
            for el in lines:
                lines_sentences.append(el.split(':')[0].strip())

            lines_component_nan = []
            indexes_component_nan = []
            for i, row in df.iterrows():
                if row['Component'] == '999':
                    lines_component_nan.append((row, i))

            for i, row in df.iterrows():
                for line in lines_component_nan:
                    if row['Speech'] == line[0]['Speech'] and row['Component'] != '999':
                        indexes_component_nan.append(line[1])

            new_df = df.drop(indexes_component_nan, axis=0)
            indexes_duplicates = []

            duplicates_without_compnull = []
            lines_nan = []
            for el in lines_component_nan:
                lines_nan.append(el[0]['Speech'])

            flag = 0
            for el in lines_sentences:
                for l in lines_nan:
                    if el.strip() == l.strip():
                        flag = 1
                if flag == 0:
                    duplicates_without_compnull.append(el)
                flag = 0

            for i, row in new_df.iterrows():
                for line in duplicates_without_compnull:
                    if row['Speech'].strip() == line.strip():
                        # print(row['Speech'])
                        indexes_duplicates.append(i)
            final_df = new_df.drop(indexes_duplicates, axis=0)

            nan = []
            for i, row in final_df.iterrows():
                if row['Component'] == '999':
                    nan.append(i)
            fdf = final_df.drop(nan, axis=0)

            fdf.to_csv(dataset_no_dup_path)

    def remove_not_found(
            self,
            debate_ids: List
    ):
        """

        :param debate_ids: list of strings representing debates IDs
        :return: None. The function removes samples marked 'NOT_FOUND', i.e. sentences for which a match with the alignment
                 results was not found.
        """
        for debate_id in tqdm(range(len(debate_ids)), desc='Removing not found...'):
            folder_id = debate_ids[debate_id]
            dataset_clip_path = self.datasets_path.joinpath(folder_id, 'dataset_clip_nodup.csv')
            dataset_no_dup_path_no_nf = self.datasets_path.joinpath(folder_id, 'dataset_clip_final.csv')
            df = pd.read_csv(dataset_clip_path)

            df = df.drop(df[df['idClip'].str.strip().str.match('NOT_FOUND', na=False)].index)

            # save df without duplicates
            df.to_csv(dataset_no_dup_path_no_nf)

    def unify_datasets_debates(
            self,
            debate_ids: List
    ):
        """

        :param debate_ids: list of strings representing debates IDs
        :return: None. The function combines the datasets created for each debate to create the new dataset MM-ElecDeb60to16
        """
        for debate_id in range(len(debate_ids)):
            folder_id = debate_ids[debate_id]
            dataset_no_dup_path_no_nf = self.datasets_path.joinpath(folder_id, 'dataset_clip_final.csv')
            df = pd.read_csv(dataset_no_dup_path_no_nf)
            break

        for debate_id in range(1, len(debate_ids)):
            folder_id = debate_ids[debate_id]
            dataset_no_dup_path_no_nf = self.datasets_path.joinpath(folder_id, 'dataset_clip_final.csv')
            df_1 = pd.read_csv(dataset_no_dup_path_no_nf)
            df = pd.concat([df, df_1])

        df = df.loc[:, ~df.columns.str.match('Unnamed')]

        # save
        final_dataset_path = self.datasets_path.joinpath('final_dataset', 'final_dataset.csv')
        final_dataset_path.parent.mkdir(parents=True)
        df.to_csv(final_dataset_path)

        # compare dimension to original dataframe
        full_dataset_path = self.datasets_path.joinpath('YesWeCan', 'sentence_db_candidate.csv')
        df_full = pd.read_csv(full_dataset_path)
        logging.info("Actual shape: ", df.shape, "Original shape: ", df_full.shape)

    def copy_final_csv(
            self,
    ):
        """

        :return: None. The function copies the generated dataset into the official 'MM-USElecDeb60to16' folder,
                 renaming the file to 'MM-USElecDeb60to16.csv'
        """
        final_csv_path = self.datasets_path.joinpath('final_dataset', 'final_dataset.csv')
        dest_final_csv = self.files_path.joinpath('MM-USElecDeb60to16', 'final_dataset.csv')
        shutil.copy(final_csv_path, dest_final_csv)
        dest_final_csv.rename(dest_final_csv.with_name('MM-USElecDeb60to16.csv'))

    def build_from_scratch(
            self
    ):
        df = pd.read_csv(self.files_path.joinpath('dictionary.csv'), sep=';')
        df.columns = ['id', 'link', 'startMin', 'startSec', 'endMin', 'endSec']
        debate_ids = df.id.values
        link = df.link.values
        start_min = df.startMin.values
        start_sec = df.startSec.values
        end_min = df.endMin.values
        end_sec = df.endSec.values

        youtube_download(save_path=self.audio_path, debate_ids=debate_ids, debate_urls=link)
        self.trim_audio(debate_ids, start_min, start_sec, end_min, end_sec)
        self.copy_transcripts(debate_ids)
        self.create_plain_text(debate_ids)
        self.generate_chunks(debate_ids)
        self.generate_empty_transcript_files(debate_ids)
        self.run_aeneas(debate_ids)
        self.generate_dataset(debate_ids)
        self.generate_clips(debate_ids)
        self.remove_duplicates(debate_ids)
        self.remove_not_found(debate_ids)
        self.unify_datasets_debates(debate_ids)
        self.copy_final_csv()

    def build_audio(
            self
    ):
        df = pd.read_csv(self.files_path.joinpath('dictionary.csv'), sep=';')
        df.columns = ['id', 'link', 'startMin', 'startSec', 'endMin', 'endSec']
        debate_ids = df.id.values
        link = df.link.values
        start_min = df.startMin.values
        start_sec = df.startSec.values
        end_min = df.endMin.values
        end_sec = df.endSec.values

        youtube_download(save_path=self.audio_path, debate_ids=debate_ids, debate_urls=link)
        self.trim_audio(debate_ids, start_min, start_sec, end_min, end_sec)
        self.generate_chunks(debate_ids)
        self.generate_clips(debate_ids)

    def load(
            self
    ):
        if not self.files_path.exists():
            logging.info('Downloading MMUSED data...This might take several minutes, enjoy a coffee ;)')
            tmp_path = self.data_path.joinpath('data.zip')
            download(url='https://zenodo.org/api/records/11179380/files-archive',
                     file_path=tmp_path)

            with zipfile.ZipFile(tmp_path, 'r') as loaded_zip:
                loaded_zip.extractall(self.data_path)

            internal_tmp_path = self.data_path.joinpath('MMUSED-data.zip')

            with zipfile.ZipFile(internal_tmp_path, 'r') as loaded_zip:
                loaded_zip.extractall(self.data_path)

            tmp_path.unlink()
            internal_tmp_path.unlink()

            source_path = self.data_path.joinpath('multimodal-dataset')
            copy_tree(source_path.as_posix(), self.data_path.as_posix())
            shutil.rmtree(source_path)

            logging.info('Download completed!')

        if self.input_mode != InputMode.AUDIO_ONLY and not self.final_path.joinpath('MM-USElecDeb60to16.csv').is_file():
            self.build_from_scratch()
        elif not self.clips_path.exists():
            self.build_audio()

    def _get_text_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return UnimodalDataset(inputs=df.Text.values,
                               labels=df.Component.values)

    def _get_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return UnimodalDataset(inputs=df['audio_paths'].values,
                               labels=df.Component.values)

    def _get_text_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return MultimodalDataset(texts=df.Text.values,
                                 audio=df['audio_paths'].values,
                                 labels=df.Component.values)

    @property
    def data(
            self
    ) -> pd.DataFrame:
        df = pd.read_csv(self.final_path.joinpath('MM-USElecDeb60to16.csv'))

        # drop rare samples where audio is empty
        df = df[df['NewBegin'] != df['NewEnd']]

        audio_paths = [self.clips_path.joinpath(document_id, f'{clip_id}.wav')
                       for document_id, clip_id in zip(df.Document.values, df['idClip'].values)]
        df['audio_paths'] = audio_paths

        if self.task_name == 'acc':
            df = df[df['Component'].isin(['Premise', 'Claim'])]
            df.loc[df.Component == 'Premise', 'Component'] = 0
            df.loc[df.Component == 'Claim', 'Component'] = 1

            # drop rows where Component is Other
            df = df[df['Component'] != 'O']
        else:
            df.loc[df['Component'].isin(['Premise', 'Claim']), 'Component'] = 'Arg'
            df.loc[df.Component != 'Arg', 'Component'] = 0
            df.loc[df.Component == 'Arg', 'Component'] = 1

        return df

    def get_default_splits(
            self,
            as_iterator: bool = False
    ) -> Union[List[SplitInfo], SplitInfo]:
        split_info = self.build_info_from_splits(train_df=self.data[self.data.Set == 'TRAIN'],
                                                 val_df=self.data[self.data.Set == 'VALIDATION'],
                                                 test_df=self.data[self.data.Set == 'TEST'])
        return [split_info] if as_iterator else split_info


class MMUSEDFallacy(Loader):

    def __init__(
            self,
            sample_rate=16000,
            clip_modality='full',
            n_files=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        assert self.task_name in ['afc']
        assert clip_modality in ['full', 'partial']

        self.sample_rate = sample_rate
        self.clip_modality = clip_modality
        self.n_files = n_files
        self.folder_name = 'MMUSED-fallacy'

        self.data_path = Path(self.base_data_path, self.folder_name).resolve()
        self.resources_path = self.data_path.joinpath('resources')
        self.audio_path = self.resources_path.joinpath('debates_audio_recordings')
        self.clips_path = self.data_path.joinpath('audio_clips')
        self.dataset_url = 'https://raw.githubusercontent.com/lt-nlp-lab-unibo/multimodal-am-fallacy/main/local_database/MM-DatasetFallacies/no_duplicates/dataset.csv'
        self.dataset_path = self.data_path.joinpath('dataset.csv')

        self.load()

        self.add_splits(method=self.get_mancini_2024_splits,
                        key='mancini-et-al-2024')

    def generate_clips(
            self,
            element,
            ids,
            dataset_path,
    ):
        # element can be sentences dialogue, snippets, components and dialogues
        # ids is a list of ids of the elements to be processed
        # modality is the modality of the audio clips to be generated
        # dataset_path is the path to the dataset
        # sample_rate is the sample rate of the audio clips to be generated

        df = pd.read_csv(dataset_path, sep='\t')
        if self.clip_modality == 'full':
            ids = df.id_map.unique()

        main_folder_path = self.clips_path.joinpath(element)
        if main_folder_path.exists():
            shutil.rmtree(main_folder_path)

        main_folder_path.mkdir(parents=True)

        for doc_id in tqdm(range(len(ids))):
            folder_id = ids[doc_id]

            dataset_path_clip_empty = self.resources_path.joinpath('clips_generation', element, folder_id,
                                                                   'dataset.csv')
            full_audio_path = self.audio_path.joinpath(folder_id, 'full_audio.wav')

            audio_clips_path = main_folder_path.joinpath(folder_id)
            audio_clips_path.mkdir(parents=True, exist_ok=True)

            df = pd.read_csv(dataset_path_clip_empty, sep='\t')
            if element == 'dial_sent':
                unique_dialogue_rows = {}
                sound = AudioSegment.from_file(full_audio_path)
                for idx, row in df.iterrows():
                    timestamps_dial_begin = list(row['DialogueAlignmentBegin'][1:-1].strip().split(','))
                    timestamps_dial_end = list(row['DialogueAlignmentEnd'][1:-1].strip().split(','))
                    dialogue = row['Dialogue']

                    id_clip_dialogues = []
                    if dialogue not in unique_dialogue_rows.keys():
                        for j in range(len(timestamps_dial_begin)):
                            if timestamps_dial_begin[j].strip() != 'NOT_FOUND' and timestamps_dial_end[
                                j].strip() != 'NOT_FOUND':
                                start_time = float(timestamps_dial_begin[j].strip().replace('\'', '')) * 1000 - 1005
                                end_time = float(timestamps_dial_end[j].strip().replace('\'', '')) * 1000 + 100
                                id_clip = 'clip_' + str(idx) + '_' + str(j)
                                clip_name = audio_clips_path.joinpath(f'{id_clip}.wav')
                                extract = sound[start_time:end_time]
                                extract = extract.set_frame_rate(self.sample_rate)
                                extract = extract.set_channels(1)
                                extract.export(clip_name, format="wav")
                                id_clip_dialogues.append(id_clip)
                        df.at[idx, "idClipDialSent"] = id_clip_dialogues
                        unique_dialogue_rows[dialogue] = id_clip_dialogues
                    else:
                        df.at[idx, "idClipDialSent"] = unique_dialogue_rows[dialogue]
            elif element == 'snippet':
                unique_snippet_rows = {}
                sound = AudioSegment.from_file(full_audio_path)
                for idx, row in df.iterrows():
                    start_time = row['BeginSnippet']
                    end_time = row['EndSnippet']
                    snippet = row['Snippet']
                    if snippet not in unique_snippet_rows.keys():
                        id_clip = 'clip_' + str(idx)
                        if start_time != 'NOT_FOUND' and end_time != 'NOT_FOUND':
                            start_time = float(row['BeginSnippet'].strip().replace('\'', '')) * 1000 - 1005
                            end_time = float(row['EndSnippet'].strip().replace('\'', '')) * 1000 + 100
                            clip_name = audio_clips_path.joinpath(f'{id_clip}.wav')
                            extract = sound[start_time:end_time]
                            extract = extract.set_frame_rate(self.sample_rate)
                            extract = extract.set_channels(1)
                            extract.export(clip_name, format="wav")
                            df.at[idx, "idClipSnippet"] = id_clip
                            unique_snippet_rows[snippet] = id_clip
                    else:
                        df.at[idx, "idClipSnippet"] = unique_snippet_rows[snippet]

    def build_clips(
            self
    ):
        dict_download_links = self.resources_path.joinpath("download", "download_links.csv")
        dict_mapping_ids = self.resources_path.joinpath("download", "link_ids.csv")

        df_mapping = pd.read_csv(dict_mapping_ids, sep=';')
        df = pd.read_csv(dict_download_links, sep=';')

        id_mapping = df_mapping.mm_id
        id_links = df.id.values
        link = df.link.values
        start_min_df = df.startMin
        start_sec_df = df.startSec
        end_min_df = df.endMin
        end_sec_df = df.endSec

        youtube_download(save_path=self.audio_path,
                         debate_ids=id_links,
                         debate_urls=link)

        tmp = []
        for x in id_links:
            tmp.append(str(x))

        id_links = tmp
        tmp = []
        for x in id_mapping:
            tmp.append(str(x))
        id_mapping = tmp

        ids = []
        links = []
        start_min = []
        start_sec = []
        end_min = []
        end_sec = []

        for i in range(len(id_links)):
            if id_links[i] in id_mapping:
                ids.append(id_links[i])
                links.append(link[i])
                start_min.append(start_min_df[i])
                start_sec.append(start_sec_df[i])
                end_min.append(end_min_df[i])
                end_sec.append(end_sec_df[i])

        if self.clip_modality == "partial":
            # if modality is partial, the user must specify the number of files to be downloaded
            # the first n_files in "dictionary.csv" will be downloaded
            if self.n_files is not None:
                n_files = int(self.n_files)
                ids = ids[:n_files]

        base_dir_support_datasets = self.resources_path.joinpath("clips_generation")

        output_cleaned_dataset_path_csv = os.path.join(base_dir_support_datasets, 'trial_cleaned.csv')
        output_snippet_dataset_path_csv = os.path.join(base_dir_support_datasets, 'trial_snippet.csv')

        # generate clips for sentences
        self.generate_clips(element="dial_sent",
                            ids=ids,
                            dataset_path=output_cleaned_dataset_path_csv)

        self.generate_clips(element="snippet",
                            ids=ids,
                            dataset_path=output_snippet_dataset_path_csv)

    def load(
            self
    ):
        if not self.resources_path.exists():
            logging.info('Downloading MMUSED-fallacy data...')
            self.data_path.mkdir(parents=True, exist_ok=True)
            tmp_path = self.data_path.joinpath('data.zip')
            download(url='https://zenodo.org/api/records/11179390/files-archive',
                     file_path=tmp_path)

            with zipfile.ZipFile(tmp_path, 'r') as loaded_zip:
                loaded_zip.extractall(self.data_path)

            internal_tmp_path = self.data_path.joinpath('MMUSED-fallacy.zip')
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

        if not self.dataset_path.exists():
            download(self.dataset_url,
                     file_path=self.dataset_path)

    def _get_text_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return UnimodalDataset(inputs=df['SentenceSnippet'].values,
                               labels=df['Fallacy'].values)

    def _get_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return UnimodalDataset(inputs=df['audio_paths'].values,
                               labels=df['Fallacy'].values)

    def _get_text_audio_data(
            self,
            df: pd.DataFrame
    ) -> MAMDataset:
        return MultimodalDataset(texts=df['SentenceSnippet'].values,
                                 audio=df['audio_paths'].values,
                                 labels=df['Fallacy'].values)

    @property
    def data(
            self
    ) -> pd.DataFrame:
        df = pd.read_csv(self.dataset_path, sep='\t')

        # drop rare samples where audio is empty
        df = df[df['BeginSnippet'] != df['EndSnippet']]

        audio_paths = [self.clips_path.joinpath('snippet', debate_id, f'{clip_id}.wav')
                       for debate_id, clip_id in zip(df['Dialogue ID'].values, df['idClipSnippet'].values)]
        df['audio_paths'] = audio_paths

        if self.task_name == 'afc':
            df.loc[df.Fallacy == 'AppealtoEmotion', 'Fallacy'] = 0
            df.loc[df.Fallacy == 'AppealtoAuthority', 'Fallacy'] = 1
            df.loc[df.Fallacy == 'AdHominem', 'Fallacy'] = 2
            df.loc[df.Fallacy == 'FalseCause', 'Fallacy'] = 3
            df.loc[df.Fallacy == 'Slipperyslope', 'Fallacy'] = 4
            df.loc[df.Fallacy == 'Slogans', 'Fallacy'] = 5

        return df

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
        dialogues = set(self.data['Dialogue ID'].values)
        splits_info = []
        for dialogue_id in dialogues:
            train_df = self.data[self.data['Dialogue ID'] != dialogue_id]
            test_df = self.data[self.data['Dialogue ID'] == dialogue_id]

            train_dialogues = set(train_df['Dialogue ID'].values)
            val_dialogues = np.random.choice(list(train_dialogues), size=4)

            val_df = train_df[train_df['Dialogue ID'].isin(val_dialogues)]
            train_df = train_df[~train_df['Dialogue ID'].isin(val_dialogues)]

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
                df_final = df_final.append(
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
                big_df = big_df.append({'id': 'n' + str_idx,
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
