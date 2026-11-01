"""Loaders that download a corpus and turn its rows into a ``MAMDataset``.

A loader exposes ``data``, one table row per sample, and ``dataset``, which
turns any subset of those rows into a ``MAMDataset`` for its input mode. Split
strategies in ``mamkit.components.splits`` choose the subsets.
"""

import logging
import re
import shutil
import tarfile
import zipfile
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, Optional

import nltk
import numpy as np
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

from mamkit.components.datasets import MAMDataset
from mamkit.utility.data import download, youtube_download

#: The modalities each input mode reads.
MODALITIES = {
    "text-only": ("text",),
    "audio-only": ("audio",),
    "text-audio": ("text", "audio"),
}


class Loader:
    """Shared fields and the conversion from table rows to a ``MAMDataset``.

    A subclass sets ``folder_name`` and ``tasks``, and its ``columns`` method
    names the table columns of each modality, of their context, and of the
    label.
    """

    folder_name: str
    tasks: tuple

    def __init__(
        self,
        task_name: str,
        input_mode: str,
        base_data_path: Optional[str] = None,
    ):
        if task_name not in self.tasks:
            raise ValueError(f"{type(self).__name__} supports tasks {self.tasks}")
        if input_mode not in MODALITIES:
            raise ValueError(f"Input mode must be one of {list(MODALITIES)}")
        self.task_name = task_name
        self.input_mode = input_mode
        base = Path.cwd() if base_data_path is None else Path(base_data_path)
        self.data_path = base.joinpath(self.folder_name).resolve()

    @property
    def needs_audio(self) -> bool:
        return "audio" in MODALITIES[self.input_mode]

    def columns(self) -> Dict:
        """``{"inputs": ..., "context": ..., "label": ...}`` column names.

        ``inputs`` and ``context`` map a modality to a column, or ``a`` and
        ``b`` to such dictionaries. ``context`` may be ``None``.
        """
        raise NotImplementedError

    @property
    def data(self) -> pd.DataFrame:
        raise NotImplementedError

    def dataset(self, df: pd.DataFrame) -> MAMDataset:
        columns = self.columns()
        modalities = MODALITIES[self.input_mode]

        def pick(names: Dict) -> Dict:
            if "a" in names:
                return {side: pick(names[side]) for side in ("a", "b")}
            return {
                modality: list(df[column])
                for modality, column in names.items()
                if modality in modalities
            }

        context = columns["context"]
        return MAMDataset(
            inputs=pick(columns["inputs"]),
            labels=list(df[columns["label"]]),
            context=pick(context) if context is not None else None,
        )


def build_once(path: Path, build: Callable[[Path], None]):
    """Runs ``build`` on a ``.part`` sibling of ``path``, then renames it to ``path``.

    ``path`` exists only after a complete build, so the next run retries a build
    that failed. The sibling keeps what the failed build wrote, so that steps
    which skip existing files can resume.
    """
    if path.exists():
        return
    partial = path.with_name(path.name + ".part")
    partial.mkdir(parents=True, exist_ok=True)
    build(partial)
    partial.rename(path)


def window_start(position: int, window: int) -> int:
    """The first of the ``window`` items before ``position``; 0 when ``window`` < 0."""
    return 0 if window < 0 else max(position - window, 0)


def split_sentences(text: str):
    """Splits ``text`` into sentences, downloading the NLTK model on first use."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    return nltk.sent_tokenize(text)


def extract_zenodo_archive(url: str, data_path: Path, inner_name: str):
    """Downloads a Zenodo record archive and extracts the archive inside it."""
    logging.info(f"Downloading {url}...")
    data_path.mkdir(parents=True, exist_ok=True)
    archive = data_path.joinpath("data.zip")
    download(url=url, file_path=archive)
    with zipfile.ZipFile(archive) as files:
        files.extractall(data_path)
    inner = data_path.joinpath(inner_name)
    with zipfile.ZipFile(inner) as files:
        files.extractall(data_path)
    archive.unlink()
    inner.unlink()


def trimmed_recording(dialogue_id: str, audio_path: Path, links: pd.DataFrame):
    """The full debate recording without the trimmed start and end."""
    recording = AudioSegment.from_file(
        audio_path.joinpath(dialogue_id, "full_audio.wav")
    )
    row = links[links.id == dialogue_id]
    start = row["startMin"].values[0] * 60 + row["startSec"].values[0]
    end = row["endMin"].values[0] * 60 + row["endSec"].values[0]
    # A zero end trim keeps the end, since recording[start:-0] is empty.
    return recording[start * 1000 : -end * 1000 if end > 0 else None]


def export_clip(recording, path: Path, start: float, end: float, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    clip = recording[start * 1000 : end * 1000]
    clip.set_frame_rate(sample_rate).set_channels(1).export(path, format="wav")


class UKDebates(Loader):
    """UKDebates, from Lippi and Torroni, AAAI 2016.

    Argument Mining from Speech: Detecting Claims in Political Debates.
    """

    folder_name = "UKDebates"
    tasks = ("asd",)
    archive_url = "http://argumentationmining.disi.unibo.it/dataset_aaai2016.tgz"
    speakers = ("Miliband", "Clegg", "Cameron")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audio_path = self.data_path.joinpath("dataset", "audio")
        # The archive holds text and audio together.
        build_once(self.data_path, self.download)

    def download(self, target: Path):
        logging.info(f"Downloading {self.folder_name} data...")
        archive = target.parent.joinpath(f"{self.folder_name}.tar.gz")
        download(url=self.archive_url, file_path=archive)
        with tarfile.open(archive) as files:
            # The data filter rejects absolute paths and links out of the folder.
            if hasattr(tarfile, "data_filter"):
                files.extractall(target, filter="data")
            else:  # pragma: no cover, Python 3.10 before 3.10.12
                files.extractall(target)
        archive.unlink()

    def columns(self):
        return {
            "inputs": {"text": "texts", "audio": "audio"},
            "context": None,
            "label": "labels",
        }

    @cached_property
    def data(self) -> pd.DataFrame:
        rows = []
        for speaker in self.speakers:
            folder = self.data_path.joinpath("dataset")
            texts = folder.joinpath(f"{speaker}.txt").read_text().splitlines(True)
            labels = folder.joinpath(f"{speaker}Labels.txt").read_text().splitlines()
            for index, (text, label) in enumerate(zip(texts, labels)):
                rows.append(
                    {
                        "texts": " ".join(text.split(" ")[1:]),
                        "audio": self.audio_path.joinpath(speaker, f"{index + 1}.wav"),
                        "labels": 1 if label.strip() == "C" else 0,
                    }
                )
        return pd.DataFrame(rows)


class MMUSED(Loader):
    """MM-USED, from Mancini et al., ArgMining 2022.

    Multimodal Argument Mining: A Case Study in Political Debates.
    """

    folder_name = "MMUSED"
    tasks = ("asd", "acc")
    archive_url = "https://zenodo.org/api/records/14938592/files-archive"

    def __init__(self, sample_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.audio_path = self.data_path.joinpath("audio_recordings")
        self.clips_path = self.data_path.joinpath("audio_clips")
        self.dataset_path = self.data_path.joinpath("dataset.pkl")

        build_once(
            self.data_path,
            lambda target: extract_zenodo_archive(
                self.archive_url, target, "MMUSED.zip"
            ),
        )
        if self.needs_audio:
            build_once(self.clips_path, self.build_clips)

    def build_clips(self, target: Path):
        """Downloads the debate recordings and cuts one clip per sentence."""
        links = pd.read_csv(self.data_path.joinpath("download_links.csv"), sep=";")
        youtube_download(
            save_path=self.audio_path,
            debate_ids=links.id.values,
            debate_urls=links.link.values,
        )
        df = pd.read_pickle(self.dataset_path)
        for dialogue_id in tqdm(sorted(set(df.dialogue_id)), desc="Building clips"):
            recording = trimmed_recording(dialogue_id, self.audio_path, links)
            for _, row in df[df.dialogue_id == dialogue_id].iterrows():
                for index, start, end in zip(
                    row["speech_indexes"],
                    row["speech_start_time"],
                    row["speech_end_time"],
                ):
                    path = target.joinpath(dialogue_id, f"{index}.wav")
                    export_clip(recording, path, start, end, self.sample_rate)
        shutil.rmtree(self.audio_path)

    def columns(self):
        return {
            "inputs": {"text": "speech", "audio": "speech_paths"},
            "context": None,
            "label": "component",
        }

    @cached_property
    def data(self) -> pd.DataFrame:
        df = pd.read_pickle(self.dataset_path)
        df["speech_paths"] = [
            [
                self.clips_path.joinpath(row["dialogue_id"], f"{index}.wav")
                for index in row["speech_indexes"]
            ]
            for _, row in df.iterrows()
        ]

        if self.task_name == "acc":
            df = df[df.component.isin(["Premise", "Claim"])].copy()
            df["component"] = (df.component == "Claim").astype(int)
        else:
            df["component"] = df.component.isin(["Premise", "Claim"]).astype(int)
        return df


FALLACIES = (
    "AppealtoEmotion",
    "AppealtoAuthority",
    "AdHominem",
    "FalseCause",
    "Slipperyslope",
    "Slogans",
)


class MMUSEDFallacy(Loader):
    """MM-USED-fallacy, from Mancini et al., EACL 2024.

    Multimodal Fallacy Classification in Political Debates. The ``afc`` task
    classifies fallacious snippets, and the ``afd`` task detects fallacious
    sentences. With ``with_context``, each input carries up to
    ``context_window`` previous sentences of its dialogue as context.
    """

    folder_name = "MMUSED-fallacy"
    tasks = ("afc", "afd")
    archive_url = "https://zenodo.org/api/records/17143462/files-archive"

    def __init__(
        self,
        with_context: bool = False,
        context_window: int = 3,
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.with_context = with_context
        self.context_window = context_window
        self.sample_rate = sample_rate
        self.audio_path = self.data_path.joinpath("audio_recordings")
        self.clips_path = self.data_path.joinpath("audio_clips")
        self.dataset_path = self.data_path.joinpath("dataset.pkl")

        build_once(
            self.data_path,
            lambda target: extract_zenodo_archive(
                self.archive_url, target, "MMUSED-fallacy.zip"
            ),
        )
        if self.needs_audio:
            build_once(self.clips_path, self.build_clips)

    def build_clips(self, target: Path):
        """Downloads the debate recordings and cuts one clip per sentence."""
        links = pd.read_csv(self.data_path.joinpath("download_links.csv"), sep=";")
        youtube_download(
            save_path=self.audio_path,
            debate_ids=links.id.values,
            debate_urls=links.link.values,
        )
        df = pd.read_pickle(self.dataset_path)
        for dialogue_id in tqdm(sorted(set(df.dialogue_id)), desc="Building clips"):
            recording = trimmed_recording(dialogue_id, self.audio_path, links)
            for _, row in df[df.dialogue_id == dialogue_id].iterrows():
                for part in ("dialogue", "snippet"):
                    for index, start, end in zip(
                        row[f"{part}_indexes"],
                        row[f"{part}_start_time"],
                        row[f"{part}_end_time"],
                    ):
                        path = target.joinpath(dialogue_id, f"{index}.wav")
                        export_clip(recording, path, start, end, self.sample_rate)
        shutil.rmtree(self.audio_path)

    def columns(self):
        if self.task_name == "afc":
            inputs = {"text": "snippet", "audio": "snippet_paths"}
            context = {"text": "dialogue", "audio": "dialogue_paths"}
            label = "fallacy"
        else:
            inputs = {"text": "sentence", "audio": "sentence_path"}
            context = {"text": "context", "audio": "context_paths"}
            label = "label"
        return {
            "inputs": inputs,
            "context": context if self.with_context else None,
            "label": label,
        }

    def clip_paths(self, row: pd.Series, part: str):
        return [
            self.clips_path.joinpath(row["dialogue_id"], f"{index}.wav")
            for index in row[f"{part}_indexes"]
        ]

    @cached_property
    def data(self) -> pd.DataFrame:
        df = pd.read_pickle(self.dataset_path)
        df["dialogue_paths"] = [
            self.clip_paths(row, "dialogue") for _, row in df.iterrows()
        ]
        df["snippet_paths"] = [
            self.clip_paths(row, "snippet") for _, row in df.iterrows()
        ]
        if self.task_name == "afc":
            df = self.afc_data(df)
            df["fallacy"] = df.fallacy.map(FALLACIES.index)
            return df
        return self.afd_data(df)

    def afc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """One row per snippet, with the dialogue sentences before it as context."""
        rows = []
        for _, row in df.iterrows():
            start = min(row["snippet_indexes"])
            kept = [
                position
                for position, index in enumerate(row["dialogue_indexes"])
                if index < start
            ]
            kept = kept[window_start(len(kept), self.context_window) :]
            sentences = [row["dialogue_sentences"][position] for position in kept]
            rows.append(
                {
                    "dialogue_indexes": [row["dialogue_indexes"][p] for p in kept],
                    "dialogue_sentences": sentences,
                    "dialogue": " ".join(sentences),
                    "dialogue_id": row["dialogue_id"],
                    "dialogue_paths": [row["dialogue_paths"][p] for p in kept],
                    "snippet_indexes": row["snippet_indexes"],
                    "snippet_sentences": row["snippet_sentences"],
                    "snippet": row["snippet"],
                    "snippet_paths": row["snippet_paths"],
                    "fallacy": row["fallacy"],
                }
            )
        return pd.DataFrame(rows)

    def afd_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """One row per dialogue sentence, labelled 1 when a snippet holds it."""
        rows = []
        for dialogue_id in tqdm(np.unique(df.dialogue_id.values), desc="Building AFD"):
            dialogue_df = df.loc[df.dialogue_id == dialogue_id]
            snippet_indexes = set(chain(*dialogue_df.snippet_indexes))
            dialogue_sentences = [
                (index, sentence, 0)
                for index, sentence in zip(
                    chain(*dialogue_df.dialogue_indexes),
                    chain(*dialogue_df.dialogue_sentences),
                )
                if index not in snippet_indexes
            ]
            # Every sentence of a snippet carries the label of its snippet.
            snippet_sentences = [
                (index, sentence, 1 if row["fallacy"] is not None else 0)
                for _, row in dialogue_df.iterrows()
                for index, sentence in zip(
                    row["snippet_indexes"], row["snippet_sentences"]
                )
            ]
            sentences = sorted(
                set(dialogue_sentences + snippet_sentences), key=lambda item: item[0]
            )
            for position, (index, sentence, label) in enumerate(sentences):
                past = sentences[window_start(position, self.context_window) : position]
                rows.append(
                    {
                        "context_paths": [
                            self.clips_path.joinpath(dialogue_id, f"{item[0]}.wav")
                            for item in past
                        ],
                        "context": " ".join(item[1] for item in past),
                        "sentence": sentence,
                        "sentence_path": self.clips_path.joinpath(
                            dialogue_id, f"{index}.wav"
                        ),
                        "label": label,
                        "dialogue_id": dialogue_id,
                        "filename": dialogue_df.filename.values[0],
                    }
                )
        return pd.DataFrame(rows)


class MArg(Loader):
    """M-Arg, from Mestre et al., ArgMining 2021.

    M-Arg: Multimodal Argument Mining Dataset for Political Debates with Audio
    and Transcripts. Pairs of sentences are kept when their annotation
    confidence reaches ``confidence``.
    """

    folder_name = "MArg"
    tasks = ("arc",)
    archive_url = "https://zenodo.org/api/records/5653504/files-archive"
    relations = ("neither", "support", "attack")

    def __init__(self, confidence: float, **kwargs):
        super().__init__(**kwargs)
        if confidence not in (0.85, 1.0):
            raise ValueError("MArg confidence must be 0.85 or 1.0")
        self.confidence = confidence
        self.feature_path = self.data_path.joinpath(
            "data", "preprocessed full dataset", "full_feature_extraction_dataset.csv"
        )
        self.aggregated_path = self.data_path.joinpath(
            "annotated dataset", "aggregated_dataset.csv"
        )
        self.final_path = self.data_path.joinpath(
            f"final_dataset_{self.confidence:.2f}.csv"
        )
        self.audio_path = self.data_path.joinpath("data", "audio sentences")

        self.speakers_map = {
            "Chris Wallace": "Chris Wallace",
            "Vice President Joe Biden": "Joe Biden",
            "President Donald J. Trump": "Donald Trump",
            "Chris Wallace:": "Chris Wallace",
            "Kristen Welker": "Kristen Welker",
            "Donald Trump": "Donald Trump",
            "Joe Biden": "Joe Biden",
            "George Stephanopoulos": "George Stephanopoulos",
            "Nicholas Fed": "Audience Member 1",
            "Kelly Lee": "Audience Member 2",
            "Anthony Archer": "Audience Member 3",
            "Voice Over": "Voice Over",
            "Cedric Humphrey": "Audience Member 4",
            "George Stephanopoulus": "George Stephanopoulos",
            "Angelia Politarhos": "Audience Member 5",
            "Speaker 1": "Voice Over",
            "Nathan Osburn": "Audience Member 6",
            "Andrew Lewis": "Audience Member 7",
            "Speaker 2": "Voice Over",
            "Michele Ellison": "Audience Member 8",
            "Mark Hoffman": "Audience Member 9",
            "Mieke Haeck": "Audience Member 10",
            "Speaker 3": "Voice Over",
            "Keenan Wilson": "Audience Member 11",
            "Savannah Guthrie": "Savannah Guthrie",
            "President Trump": "Donald Trump",
            "Jacqueline Lugo": "Audience Member 12",
            "Barbara Peña": "Audience Member 13",
            "Isabella Peña": "Audience Member 14",
            "Savannah": "Savannah Guthrie",
            "Cristy Montesinos Alonso": "Audience Member 15",
            "Adam Schucher": "Audience Member 16",
            "Moriah Geene": "Audience Member 17",
            "Cindy Velez": "Audience Member 18",
            "Paulette Dale": "Audience Member 19",
            "Susan Page": "Susan Page",
            "Kamala Harris": "Kamala Harris",
            "Mike Pence": "Mike Pence",
            "Kamala Harris ": "Kamala Harris",
        }
        self.speakers_to_id = {"error": "error"}
        self.debate_id_map = {
            "us_election_2020_1st_presidential_debate_part1_timestamp.csv": "00",
            "us_election_2020_1st_presidential_debate_part2_timestamp.csv": "01",
            "us_election_2020_2nd_presidential_debate_part1_timestamp.csv": "02",
            "us_election_2020_2nd_presidential_debate_part2_timestamp.csv": "03",
            "us_election_2020_biden_town_hall_part1_timestamp.csv": "04",
            "us_election_2020_biden_town_hall_part2_timestamp.csv": "05",
            "us_election_2020_biden_town_hall_part3_timestamp.csv": "06",
            "us_election_2020_biden_town_hall_part4_timestamp.csv": "07",
            "us_election_2020_biden_town_hall_part5_timestamp.csv": "08",
            "us_election_2020_biden_town_hall_part6_timestamp.csv": "09",
            "us_election_2020_biden_town_hall_part7_timestamp.csv": "10",
            "us_election_2020_trump_town_hall_1_timestamp.csv": "11",
            "us_election_2020_trump_town_hall_2_timestamp.csv": "12",
            "us_election_2020_trump_town_hall_3_timestamp.csv": "13",
            "us_election_2020_trump_town_hall_4_timestamp.csv": "14",
            "us_election_2020_vice_presidential_debate_1_timestamp.csv": "15",
            "us_election_2020_vice_presidential_debate_2_timestamp.csv": "16",
        }

        build_once(self.data_path, self.download)
        if not self.final_path.exists():
            self.build()

    def columns(self):
        return {
            "inputs": {
                "a": {"text": "sentence_1", "audio": "sentence_1_audio_path"},
                "b": {"text": "sentence_2", "audio": "sentence_2_audio_path"},
            },
            "context": None,
            "label": "relation",
        }

    @cached_property
    def data(self) -> pd.DataFrame:
        df = pd.read_csv(self.final_path)
        for column in ("sentence_1_audio_path", "sentence_2_audio_path"):
            df[column] = [Path(item) for item in df[column]]
        df["relation"] = df.relation.map(self.relations.index)
        return df

    def _build_complete_dataset(self, feature_df, aggregated_df):
        df_final = pd.DataFrame(
            columns=[
                "id",
                "relation",
                "confidence",
                "sentence_1",
                "sentence_2",
                "sentence_1_audio",
                "sentence_2_audio",
            ]
        )

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
            s1a = feature_df["audio_file"].loc[feature_df["text"] == s1t].values[0]
            s2a = feature_df["audio_file"].loc[feature_df["text"] == s2t].values[0]

            if row["relation:confidence"] >= self.confidence:
                df_final = df_final._append(
                    {
                        "id": id1,
                        "relation": relation1,
                        "confidence": conf1,
                        "sentence_1": s1t,
                        "sentence_2": s2t,
                        "sentence_1_audio": s1a,
                        "sentence_1_audio_path": self.audio_path.joinpath(s1a + ".wav"),
                        "sentence_2_audio": s2a,
                        "sentence_2_audio_path": self.audio_path.joinpath(s2a + ".wav"),
                    },
                    ignore_index=True,
                )

        return df_final

    def build_chunks(self):
        tokenized_path = self.data_path.joinpath("data", "tokenised text")

        idx = 0
        for filepath in self.data_path.joinpath("data", "original data").glob("*.csv"):
            speaker = []
            text = []

            df = pd.read_csv(filepath, header=0, names=["speaker", "minute", "text"])

            for r in df.index:
                # Tokenised text
                split_text = split_sentences(df["text"].iloc[r])
                text += split_text
                # Match speakers to their names using the dictionary above
                try:
                    speaker_name = self.speakers_map[df["speaker"].iloc[r]]
                except Exception:  # In theory this exception is never raised
                    speaker_name = "Speaker not found"
                speaker += [speaker_name] * len(split_text)
                # Assign IDs to speakers
                # A dictionary of speakers_to_id is filled for later use
                if speaker_name not in self.speakers_to_id:
                    if idx < 10:
                        self.speakers_to_id[speaker_name] = "0" + str(idx)
                    else:
                        self.speakers_to_id[speaker_name] = str(idx)
                    idx = idx + 1

            df_split = pd.DataFrame({"speaker": speaker, "text": text})
            df_split.to_csv(
                tokenized_path.joinpath(filepath.stem + "_split.csv"), index=False
            )

            # We also create a version in plan text (used by the force aligner tool)
            with open(
                tokenized_path.joinpath(filepath.stem + "_plain.txt"),
                "w",
                encoding="utf8",
            ) as f:
                for t in df_split["text"]:
                    f.write(t + "\n")

        filepath_topic = self.data_path.joinpath("data", "preprocessed full dataset")
        df_topic = pd.read_csv(
            filepath_topic.joinpath("dataset_context_definition.csv")
        )
        filepath_timestamps = self.data_path.joinpath("data", "timestamps")
        filepath_audio = self.data_path.joinpath("data", "split audio")
        filepath_audio_save = self.data_path.joinpath("data", "audio sentences")

        filepath_audio_save.mkdir(parents=True, exist_ok=True)

        buffer = 2

        big_df = pd.DataFrame(
            columns=[
                "id",
                "text",
                "speaker",
                "speaker_id",
                "audio_file",
                "context",
                "debate",
            ]
        )

        for timestamp_name in self.debate_id_map:
            df = pd.read_csv(
                filepath_timestamps.joinpath(timestamp_name),
                header=None,
                names=["id", "start", "end", "text"],
            )

            debate_mp3 = AudioSegment.from_mp3(
                filepath_audio.joinpath(timestamp_name.split("_timestamp")[0] + ".mp3")
            )

            speak_df = pd.read_csv(
                tokenized_path.joinpath(
                    re.sub(r"_(part)?\d+_timestamp\.csv$", "_split.csv", timestamp_name)
                ),
                header=0,
                names=["speaker", "text"],
            )

            f_idx = self.debate_id_map[timestamp_name]
            idx = 0
            for index, row in df.iterrows():
                if row["start"] < buffer:
                    extract = debate_mp3[
                        row["start"] * 1000 : (row["end"] + buffer) * 1000
                    ]
                else:
                    try:
                        extract = debate_mp3[
                            (row["start"] - buffer) * 1000 : (row["end"] + buffer)
                            * 1000
                        ]
                    except Exception:
                        extract = debate_mp3[
                            (row["start"] - buffer) * 1000 : row["end"] * 1000
                        ]

                str_idx = str(idx)
                while len(str_idx) < 4:
                    str_idx = "0" + str_idx
                str_idx = str(f_idx) + str_idx

                idx = idx + 1
                try:
                    speaker = speak_df[speak_df["text"] == row["text"]]["speaker"].iloc[
                        0
                    ]
                except Exception:
                    logging.info(f"Error parsing {row['text']}")
                    speaker = "error"

                context = df_topic[df_topic["id"] == "n" + str_idx]["context"].values[0]
                if context == "Ignore":
                    context = None

                timestamp = row["id"]
                big_df = big_df._append(
                    {
                        "id": "n" + str_idx,
                        "text": row["text"],
                        "speaker": speaker,
                        "speaker_id": self.speakers_to_id[speaker],
                        "audio_file": ("a" + str_idx),
                        "context": context,
                        "debate": str(f_idx),
                        "timestamp": timestamp,
                    },
                    ignore_index=True,
                )

                extract.export(
                    filepath_audio_save.joinpath("a" + str_idx + ".wav"), format="wav"
                )

        big_df.to_csv(
            filepath_topic.joinpath("full_feature_extraction_dataset.csv"), index=False
        )

    def download(self, target: Path):
        logging.info("Downloading M-Arg data...")
        archive = target.joinpath("data.zip")
        download(url=self.archive_url, file_path=archive)
        with zipfile.ZipFile(archive, "r") as loaded_zip:
            loaded_zip.extractall(target)
        archive.unlink()

        inner = target.joinpath(
            "rafamestre", "m-arg_multimodal-argumentation-dataset-v1.0.0.zip"
        )
        with zipfile.ZipFile(inner, "r") as loaded_zip:
            loaded_zip.extractall(target)
        inner.unlink()

        source_path = target.joinpath(
            "rafamestre-m-arg_multimodal-argumentation-dataset-851736f"
        )
        shutil.copytree(source_path, target, dirs_exist_ok=True)
        shutil.rmtree(source_path)

    def build(self):
        logging.info("Building M-Arg dataset...")
        self.build_chunks()
        feature_df = pd.read_csv(self.feature_path)
        aggregated_df = pd.read_csv(self.aggregated_path)
        train_df = self._build_complete_dataset(
            feature_df=feature_df, aggregated_df=aggregated_df
        )

        # Add index for cv routine
        train_df["index"] = np.arange(train_df.shape[0])

        train_df.to_csv(self.final_path, index=False)


class Toy(Loader):
    """A synthetic text corpus for smoke tests, with no download.

    Each text holds random filler words and one cue word, ``good`` for label 1
    and ``bad`` for label 0. The ``split`` column assigns 60% of the rows to
    training and 20% each to validation and test.
    """

    folder_name = "Toy"
    tasks = ("toy",)
    filler = ("the", "a", "debate", "policy", "tax", "vote", "plan", "now")

    def __init__(self, size: int = 200, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        if self.input_mode != "text-only":
            raise ValueError("The toy corpus has text only")
        self.size = size
        self.seed = seed

    def columns(self):
        return {"inputs": {"text": "text"}, "context": None, "label": "label"}

    @cached_property
    def data(self) -> pd.DataFrame:
        generator = np.random.default_rng(self.seed)
        labels = generator.integers(0, 2, self.size)
        texts = []
        for label in labels:
            words = list(generator.choice(self.filler, size=5))
            words.insert(generator.integers(0, 6), "good" if label else "bad")
            texts.append(" ".join(words))
        train, val = int(0.6 * self.size), int(0.8 * self.size)
        splits = ["TRAIN"] * train + ["VALIDATION"] * (val - train)
        splits += ["TEST"] * (self.size - val)
        return pd.DataFrame({"text": texts, "label": labels.tolist(), "split": splits})
