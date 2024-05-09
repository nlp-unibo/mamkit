import abc
import logging
import os
import shutil
import tarfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List, Callable

import pandas as pd
import simplejson as sj
import yt_dlp
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
from torch.utils.data import Dataset
from tqdm import tqdm

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

        return SplitInfo(train=MultimodalDataset(texts=train_df.texts.values, audio=train_df.audio.values,
                                                 labels=train_df.labels.values),
                         val=MultimodalDataset(texts=val_df.texts.values, audio=val_df.audio.values,
                                               labels=val_df.labels.values),
                         test=MultimodalDataset(texts=test_df.texts.values, audio=test_df.audio.values,
                                                labels=test_df.labels.values))

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

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        assert self.task_name in ['asd', 'acc']

        self.folder_name = 'MMUSED'
        self.filepath = Path(Path.cwd().parent, 'data', self.folder_name).resolve()
        self.audio_path = self.filepath.joinpath('audio')
        self.text_path = self.filepath.joinpath('text')
        self.transcripts_path = self.filepath.joinpath('transcripts')
        self.deleted_ids = ['13_1988, 17_1992, 42_2016, 43_2016']

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

            doc_path.mkdir(parents=True, exist_ok=False)
            filename = doc_path.joinpath("full_audio.wav")
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
                current_path = self.text_path.joinpath(f'{el}.txt')
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
            filename = self.audio_path.joinpath(folder_id, 'full_audio_trim.wav')
            chunks_folder = self.audio_path.joinpath(folder_id, 'splits')
            chunks_folder.mkdir(parents=True, exist_ok=False)

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
                    txt_filename = filename.with_suffix('txt')
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
            curr_dir = os.getcwd()
            aeneas_script_folder = Path.cwd().joinpath('run_aeneas')
            DEST_CLIP_FOLDER = Path.cwd().joinpath(') + '/files/alignment_results/' + folder_id
            os.mkdir(DEST_CLIP_FOLDER)
            for filename in os.listdir(splits_audio_path):
                if filename != '.DS_Store':
                    parts = filename.split('.')
                    txt_file = parts[0] + '.txt'
                    split_audio_path = splits_audio_path + '/' + filename
                    split_text_path = split_transcripts_path + txt_file
                    copy_command_text = "cp " + split_text_path + " " + aeneas_script_folder
                    copy_command_audio = "cp " + split_audio_path + " " + aeneas_script_folder
                    os.system(copy_command_text)
                    os.system(copy_command_audio)
            os.chdir('run_aeneas')

            AENEAS_COMMAND = "./run.sh"
            os.system(AENEAS_COMMAND)

            for filename in os.listdir(os.getcwd()):
                if filename != '.DS_Store':
                    parts = filename.split('.')
                    if parts[-1] == 'json':
                        shutil.move(os.getcwd() + '/' + filename, DEST_CLIP_FOLDER)

            for filename in os.listdir(os.getcwd()):
                if filename != '.DS_Store':
                    parts = filename.split('.')
                    if parts[-1] == 'wav' or parts[-1] == 'txt':
                        os.system("rm " + filename)
            os.chdir('..')

    def build_from_scratch(
            self
    ):
        df = pd.read_csv('files/dictionary.csv', sep=';')
        df.columns = ['debate_ids', 'debate_urls', 'start_min', 'start_sec', 'end_min', 'end_sec']
        debate_ids = df.id
        link = df.link
        start_min = df.startMin
        start_sec = df.startSec
        end_min = df.endMin
        end_sec = df.endSec

        self.youtube_download(debate_ids, link)
        self.trim_audio(debate_ids, start_min, start_sec, end_min, end_sec)
        self.copy_transcripts(debate_ids)
        self.create_plain_text(debate_ids)
        self.generate_chunks(debate_ids)
        self.generate_empty_transcript_files(debate_ids)
        utils.run_aeneas(debate_ids)
        utils.generate_dataset(debate_ids)
        utils.generate_clips(debate_ids)
        utils.remove_duplicates(debate_ids)
        utils.remove_not_found(debate_ids)
        utils.unify_datasets_debates(debate_ids)

        utils.copy_final_csv()
        utils.copy_clips()


# TODO: complete
class MMUSEDFallacy(Loader):
    pass


# TODO: complete
class MArg(Loader):
    pass
