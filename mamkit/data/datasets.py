import abc
import logging
import os
import shutil
import tarfile
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List, Callable, Union, Dict

import numpy as np
import pandas as pd
import simplejson as sj
import yt_dlp
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
from torch.utils.data import Dataset
from tqdm import tqdm

from mamkit.utility.data import download, download_from_git


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


class Loader(abc.ABC):

    def __init__(
            self,
            task_name: str,
            input_mode: InputMode
    ):
        self.task_name = task_name
        self.input_mode = input_mode

        self.splitters = {
            'default': self.get_default_splits
        }
        self.data_retriever: Dict[InputMode, Callable[[pd.DataFrame], Dataset]] = {
            InputMode.TEXT_ONLY: self.get_text_data,
            InputMode.AUDIO_ONLY: self.get_audio_data,
            InputMode.TEXT_AUDIO: self.get_text_audio_data
        }

    def add_splits(
            self,
            method: Callable[[pd.DataFrame], List[SplitInfo]],
            key: str
    ):
        self.splitters[key] = method

    @abc.abstractmethod
    def get_text_data(
            self,
            df: pd.DataFrame
    ) -> Dataset:
        pass

    @abc.abstractmethod
    def get_audio_data(
            self,
            df: pd.DataFrame
    ) -> Dataset:
        pass

    @abc.abstractmethod
    def get_text_audio_data(
            self,
            df: pd.DataFrame
    ) -> Dataset:
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
            data: pd.DataFrame
    ) -> SplitInfo:
        pass

    def get_splits(
            self,
            key: str = 'default'
    ) -> Union[List[SplitInfo], SplitInfo]:
        return self.splitters[key](self.data)

    @property
    @abc.abstractmethod
    def data(
            self
    ) -> pd.DataFrame:
        pass


class UKDebate(Loader):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        assert self.task_name in ['asd', 'cd']

        self.download_url = 'http://argumentationmining.disi.unibo.it/dataset_aaai2016.tgz'
        self.folder_name = 'UKDebate'
        self.data_path = Path(Path.cwd().parent, 'data', self.folder_name).resolve()
        self.audio_path = self.data_path.joinpath('audio')

        if not self.data_path.exists():
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

        annotations_path = self.data_path.joinpath('dataset', f'{speaker.capitalize()}Labels.txt')
        with annotations_path.open('r') as txt:
            labels = txt.readlines()
            labels = [1 if label == 'C' else 0 for label in labels]

        audio = [self.audio_path.joinpath(speaker.capitalize(), f'{idx}.wav') for idx in range(len(texts))]

        return texts, audio, labels

    def load(
            self
    ):
        logging.getLogger(__name__).info('Downloading UKDebate dataset...')
        archive_path = self.data_path.parent.joinpath('ukdebate.tar.gz')
        download(url=self.download_url,
                 file_path=archive_path)
        logging.getLogger(__name__).info('Download completed...')

        logging.getLogger(__name__).info('Extracting data...')
        self.data_path.mkdir(parents=True)
        with tarfile.open(archive_path) as loaded_tar:
            loaded_tar.extractall(self.data_path)

        logging.getLogger(__name__).info('Extraction completed...')

        if archive_path.is_file():
            archive_path.unlink()

    def get_text_data(
            self,
            df: pd.DataFrame
    ) -> Dataset:
        return UnimodalDataset(inputs=df.texts.values,
                               labels=df.labels.values)

    def get_audio_data(
            self,
            df: pd.DataFrame
    ) -> Dataset:
        return UnimodalDataset(inputs=df.audio.values,
                               labels=df.labels.values)

    def get_text_audio_data(
            self,
            df: pd.DataFrame
    ) -> Dataset:
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
            data: pd.DataFrame
    ) -> SplitInfo:
        return self.build_info_from_splits(train_df=data, val_df=pd.DataFrame.empty, test_df=pd.DataFrame.empty)

    def get_mancini_2022_splits(
            self,
            data: pd.DataFrame
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
            train_df = data.iloc[fold['train']]
            val_df = data.iloc[fold['validation']]
            test_df = data.iloc[fold['test']]

            fold_info = self.build_info_from_splits(train_df=train_df, val_df=val_df, test_df=test_df)
            split_info.append(fold_info)

        return split_info


# TODO: complete
# TODO: integrate EM's script for building the corpus
class MMUSED(Loader):

    def __init__(
            self,
            force_download=False,
            **kwargs
    ):
        super().__init__(**kwargs)

        assert self.task_name in ['asd', 'acc']

        self.folder_name = 'MMUSED'
        self.deleted_ids = ['13_1988, 17_1992, 42_2016, 43_2016']

        self.data_path = Path(Path.cwd().parent, 'data', self.folder_name).resolve()
        self.files_path = self.data_path.joinpath('files')
        self.aeneas_path = self.data_path.joinpath('aeneas')
        self.audio_path = self.files_path.joinpath('debates_audio_recordings')
        self.datasets_path = self.files_path.joinpath('datasets')
        self.transcripts_path = self.files_path.joinpath('transcripts')
        self.alignments_path = self.files_path.joinpath('alignment_results')
        self.clips_path = self.files_path.joinpath('audio_clips')
        self.final_path = self.files_path.joinpath('MM-USElecDeb60to16')

        self.alignments_path.mkdir(parents=True, exist_ok=True)
        self.clips_path.mkdir(parents=True, exist_ok=True)
        self.aeneas_path.mkdir(parents=True, exist_ok=True)
        self.final_path.mkdir(parents=True, exist_ok=True)

        download_from_git(repo='multimodal-am',
                          org='lt-nlp-lab-unibo',
                          folder=Path('multimodal-dataset', 'files'),
                          destination=self.files_path,
                          force_download=force_download)

        download_from_git(repo='multimodal-am',
                          org='lt-nlp-lab-unibo',
                          folder=Path('multimodal-dataset', 'run_aeneas'),
                          destination=self.aeneas_path,
                          force_download=force_download)

        if self.input_mode != InputMode.AUDIO_ONLY and not self.final_path.joinpath('MM-USElecDeb60to16.csv').is_file():
            self.build_from_scratch()
        elif not self.final_path.joinpath('audio_clips').exists():
            self.build_audio()

    def youtube_download(
            self,
            debate_ids: List,
            debate_urls: List
    ) -> None:
        """

        :param debate_ids: list of strings representing debates IDs
        :param debate_urls: list of strings representing the urls to the YouTube videos of the debates
        :return: None. The function populates the folder 'files/debates_audio_recordings' by creating a folder for each
                 debate. Each folder contains the audio file extracted from the corresponding video
        """

        map_debate_link = dict(zip(debate_ids, debate_urls))
        for doc, link in tqdm(map_debate_link.items()):
            doc_path = self.audio_path.joinpath(doc)
            if not self.audio_path.exists():
                self.audio_path.mkdir(parents=True)

            doc_path.mkdir(parents=True, exist_ok=True)
            filename = doc_path.joinpath("full_audio")

            if filename.with_suffix('.wav').exists():
                logging.getLogger(__name__).info(f'Skipping {link} since {filename.name} already exists...')
                continue
            else:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'outtmpl': filename
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([link])
                os.system("youtube-dl --rm-cache-dir")

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
        for idx in tqdm(range(len(debate_ids))):
            db_folder_id = debate_ids[idx]

            export_filename = self.audio_path.joinpath(db_folder_id, 'full_audio.wav')
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
        for debate_id in range(len(debate_ids)):
            folder_id = debate_ids[debate_id]
            filename = self.audio_path.joinpath(folder_id, 'full_audio.wav')
            chunks_folder = self.audio_path.joinpath(folder_id, 'splits')
            chunks_folder.mkdir(parents=True, exist_ok=True)

            sound = AudioSegment.from_mp3(filename)
            duration = sound.duration_seconds
            cut_sec = 1200
            n_chunks = round(duration / cut_sec)  # we split files in chunks of 20 mins

            for chunk_id in tqdm(range(n_chunks)):
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

        for i in tqdm(range(len(debate_ids))):
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

        for debate_id in tqdm(range(len(debate_ids))):
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

        for debate_id in tqdm(range(len(debate_ids))):
            folder_id = debate_ids[debate_id]
            dataset_path = self.datasets_path.joinpath(folder_id, 'dataset.csv')
            dataset_clip_path = self.datasets_path.joinpath(folder_id, 'dataset_clip.csv')
            full_audio_path = self.audio_path.joinpath(folder_id, 'full_audio.wav')
            audio_clips_path = self.clips_path.joinpath(folder_id)
            audio_clips_path.mkdir(parents=True, exist_ok=True)

            # read dataframe with timestamps
            df = pd.read_csv(dataset_path)

            # generate clips
            sound = AudioSegment.from_file(full_audio_path)
            total_len = df.shape[0]
            for i, row in tqdm(df.iterrows(), total=total_len, position=0):
                start_time = row['NewBegin']
                idClip = 'clip_' + str(i)
                if start_time != 'NOT_FOUND':
                    start_time = float(row['NewBegin']) * 1000  # sec -> ms conversion
                    end_time = float(row['NewEnd']) * 1000  # sec -> ms conversion
                    clip_name = audio_clips_path.joinpath(f'{idClip}.wav')
                    extract = sound[start_time:end_time]
                    extract.export(clip_name, format="wav")
                    df.at[i, "idClip"] = idClip

            # save new csv
            df.to_csv(dataset_clip_path)

    def remove_duplicates(
            self,
            debate_ids: List
    ):
        """

        :param debate_ids: list of strings representing debates IDs
        :return: None. The function removes duplicates in the dataset
        """
        for debate_id in tqdm(range(len(debate_ids))):
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
        for debate_id in tqdm(range(len(debate_ids))):
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
        for debate_id in tqdm(range(len(debate_ids))):
            folder_id = debate_ids[debate_id]
            dataset_no_dup_path_no_nf = self.datasets_path.joinpath(folder_id, 'dataset_clip_final.csv')
            df = pd.read_csv(dataset_no_dup_path_no_nf)
            break

        for debate_id in tqdm(range(1, len(debate_ids))):
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
        logging.getLogger(__name__).info("Actual shape: ", df.shape, "Original shape: ", df_full.shape)

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

        self.youtube_download(debate_ids, link)
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

        self.youtube_download(debate_ids, link)
        self.trim_audio(debate_ids, start_min, start_sec, end_min, end_sec)
        self.generate_chunks(debate_ids)
        self.generate_clips(debate_ids)

    def get_text_data(
            self,
            df: pd.DataFrame
    ) -> Dataset:
        return UnimodalDataset(inputs=df.Text.values,
                               labels=df.Component.values)

    def get_audio_data(
            self,
            df: pd.DataFrame
    ) -> Dataset:
        return UnimodalDataset(inputs=df['audio_paths'].values,
                               labels=df.Component.values)

    def get_text_audio_data(
            self,
            df: pd.DataFrame
    ) -> Dataset:
        return MultimodalDataset(texts=df.Text.values,
                                 audio=df['audio_paths'].values,
                                 labels=df.Component.values)

    @property
    def data(
            self
    ) -> pd.DataFrame:
        df = pd.read_csv(self.final_path.joinpath('MM-USElecDeb60to16.csv'))

        audio_paths = [self.clips_path.joinpath(document_id, f'{clip_id}.wav')
                       for document_id, clip_id in zip(df.Document.values, df['idClip'].values)]
        df['audio_paths'] = audio_paths

        if self.task_name == 'acd':
            df = df[df['Component'].isin(['Premise', 'Claim'])]
        else:
            df.loc[df['Component'].isin(['Premise', 'Claim']), 'Component'] = 'Arg'

        return df

    def get_default_splits(
            self,
            data: pd.DataFrame
    ) -> SplitInfo:
        return self.build_info_from_splits(train_df=data[data.Set == 'TRAIN'],
                                           val_df=data[data.Set == 'VALIDATION'],
                                           test_df=data[data.Set == 'TEST'])


class MMUSEDFallacy(Loader):

    def __init__(
            self,
            force_download=False,
            **kwargs
    ):
        super().__init__(**kwargs)

        assert self.task_name in ['afc']

        self.folder_name = 'MMUSED-fallacy'

        self.data_path = Path(Path.cwd().parent, 'data', self.folder_name).resolve()
        self.audio_path = self.data_path.joinpath('debates_audio_recordings')
        self.clips_path = self.data_path.joinpath('audio_clips')

        self.clips_path.mkdir(parents=True, exist_ok=True)


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

        self.data_path = Path(Path.cwd().parent, 'data', self.folder_name).resolve()
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

        self.df = self.build()

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
                    logging.getLogger(__name__).info(f'Error parsing {row["text"]}')
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
        if not self.final_path.exists():
            if not any(self.data_path.iterdir()):
                logging.getLogger(__name__).info('Download M-Arg data...')
                download_from_git(repo='m-arg_multimodal-argumentation-dataset',
                                  org='rafamestre',
                                  folder=Path('.'),
                                  destination=self.data_path)
                logging.getLogger(__name__).info('Download completed...')

            logging.getLogger(__name__).info('Building M-Arg dataset...')
            self.build_chunks()
            feature_df = pd.read_csv(self.feature_path)
            aggregated_df = pd.read_csv(self.aggregated_path)
            train_df = self._build_complete_dataset(feature_df=feature_df,
                                                    aggregated_df=aggregated_df)

            # Add index for cv routine
            train_df['index'] = np.arange(train_df.shape[0])

            train_df.to_csv(self.final_path, index=False)
        else:
            train_df = pd.read_csv(self.final_path)

        return train_df

    def get_text_data(
            self,
            df: pd.DataFrame
    ) -> Dataset:
        return PairUnimodalDataset(a_inputs=df.sentence_1.values,
                                   b_inputs=df.sentence_2.values,
                                   labels=df.relation.values)

    def get_audio_data(
            self,
            df: pd.DataFrame
    ) -> Dataset:
        return PairUnimodalDataset(a_inputs=df.sentence_1_audio_path.values,
                                   b_inputs=df.sentence_2_audio_path.values,
                                   labels=df.relation.values)

    def get_text_audio_data(
            self,
            df: pd.DataFrame
    ) -> Dataset:
        return PairMultimodalDataset(a_texts=df.sentence_1.values,
                                     b_texts=df.sentence_2.values,
                                     a_audio=df.sentence_1_audio_path.values,
                                     b_audio=df.sentence_2_audio_path.values,
                                     labels=df.relation.values)

    @property
    def data(
            self
    ) -> pd.DataFrame:
        return self.df

    def get_default_splits(
            self,
            data: pd.DataFrame
    ) -> SplitInfo:
        return self.build_info_from_splits(train_df=data, val_df=pd.DataFrame.empty, test_df=pd.DataFrame.empty)

    def get_mancini_2022_splits(
            self,
            data: pd.DataFrame
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
            train_df = data.iloc[fold['train']]
            val_df = data.iloc[fold['validation']]
            test_df = data.iloc[fold['test']]

            fold_info = self.build_info_from_splits(train_df=train_df, val_df=val_df, test_df=test_df)
            split_info.append(fold_info)

        return split_info
