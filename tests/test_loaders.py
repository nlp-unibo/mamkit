"""Loaders and split strategies on prepared data directories, offline."""

import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile
from cinnamon.registry import Registry

import mamkit
from mamkit.components import loaders
from mamkit.configurations import keys


def no_audio_download(*args, **kwargs):
    raise AssertionError("audio downloaded")


def no_download(*args, **kwargs):
    raise AssertionError("downloaded")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Fails any download that a test does not replace."""
    monkeypatch.setattr(loaders, "download", no_download)
    monkeypatch.setattr(loaders, "youtube_download", no_audio_download)
    monkeypatch.setattr("mamkit.components.splits.download", no_download)


SPEAKERS = {
    "Miliband": (["1 we will act\n", "2 thank you\n"], ["C\n", "N\n"]),
    "Clegg": (["1 fair taxes now\n"], ["C\n"]),
    "Cameron": (["1 good evening\n"], ["N\n"]),
}


def write_ukdebates(root: Path):
    folder = root / "dataset"
    folder.mkdir(parents=True)
    for speaker, (texts, labels) in SPEAKERS.items():
        (folder / f"{speaker}.txt").write_text("".join(texts))
        (folder / f"{speaker}Labels.txt").write_text("".join(labels))


@pytest.fixture
def ukdebates_root(tmp_path):
    write_ukdebates(tmp_path / "UKDebates")
    return tmp_path


def ukdebates(root, mode="text-only"):
    return loaders.UKDebates(task_name="asd", input_mode=mode, base_data_path=root)


def test_ukdebates_reads_speakers_in_order(ukdebates_root):
    data = ukdebates(ukdebates_root).data
    assert list(data.texts) == [
        "we will act\n",
        "thank you\n",
        "fair taxes now\n",
        "good evening\n",
    ]
    assert list(data.labels) == [1, 0, 1, 0]
    assert data.audio[2].parts[-2:] == ("Clegg", "1.wav")


def test_loaders_keep_the_modalities_of_their_input_mode(ukdebates_root):
    rows = ukdebates(ukdebates_root).data
    assert list(ukdebates(ukdebates_root).dataset(rows).inputs) == ["text"]
    audio = ukdebates(ukdebates_root, "audio-only").dataset(rows)
    assert list(audio.inputs) == ["audio"]
    both = ukdebates(ukdebates_root, "text-audio").dataset(rows)
    assert list(both.inputs) == ["text", "audio"]
    assert both.labels == [1, 0, 1, 0]
    assert both.context is None


def test_loaders_reject_unknown_tasks_and_input_modes(ukdebates_root):
    with pytest.raises(ValueError, match="tasks"):
        loaders.UKDebates(task_name="acc", input_mode="text-only")
    with pytest.raises(ValueError, match="Input mode"):
        ukdebates(ukdebates_root, mode="video")


def test_ukdebates_downloads_and_extracts_its_archive(monkeypatch, tmp_path):
    source = tmp_path / "source"
    write_ukdebates(source)

    def fake_download(url, file_path):
        assert url == loaders.UKDebates.archive_url
        with tarfile.open(file_path, "w:gz") as archive:
            archive.add(source / "dataset", arcname="dataset")

    monkeypatch.setattr(loaders, "download", fake_download)
    data = ukdebates(tmp_path / "root").data
    assert len(data) == 4
    assert [p.name for p in (tmp_path / "root").iterdir()] == ["UKDebates"]


MMUSED_ROWS = pd.DataFrame(
    {
        "dialogue_id": ["d1", "d1", "d2", "d2"],
        "speech": ["we act", "taxes rise", "hello", "good plan"],
        "component": ["Claim", "Premise", "O", "Claim"],
        "split": ["TRAIN", "TRAIN", "VALIDATION", "TEST"],
        "speech_indexes": [[0], [1, 2], [0], [1]],
        "speech_start_time": [[0.0], [0.1, 0.2], [0.0], [0.1]],
        "speech_end_time": [[0.1], [0.2, 0.3], [0.1], [0.2]],
    }
)
LINKS = "id;link;startMin;startSec;endMin;endSec\nd1;u1;0;0;0;0\nd2;u2;0;0;0;0\n"


@pytest.fixture
def mmused_root(tmp_path):
    folder = tmp_path / "MMUSED"
    folder.mkdir()
    MMUSED_ROWS.to_pickle(folder / "dataset.pkl")
    (folder / "download_links.csv").write_text(LINKS)
    return tmp_path


def mmused(root, task="asd", mode="text-only"):
    return loaders.MMUSED(task_name=task, input_mode=mode, base_data_path=root)


def test_mmused_labels_argumentative_sentences_for_asd(mmused_root, monkeypatch):
    data = mmused(mmused_root).data
    assert list(data.component) == [1, 1, 0, 1]
    assert data.speech_paths[1][1].parts[-2:] == ("d1", "2.wav")


def test_mmused_labels_claims_and_premises_for_acc(mmused_root):
    data = mmused(mmused_root, task="acc").data
    assert list(data.speech) == ["we act", "taxes rise", "good plan"]
    assert list(data.component) == [1, 0, 1]


def fake_youtube_download(save_path, debate_ids, debate_urls):
    # One second of audio per debate, at 16 kHz.
    for debate in debate_ids:
        (save_path / debate).mkdir(parents=True, exist_ok=True)
        soundfile.write(save_path / debate / "full_audio.wav", np.zeros(16_000), 16_000)


def test_mmused_cuts_clips_only_when_audio_is_needed(mmused_root, monkeypatch):
    # Text-only mode downloads no audio, which #2 reported for MM-USED-fallacy.
    mmused(mmused_root)
    assert not (mmused_root / "MMUSED" / "audio_clips").exists()

    monkeypatch.setattr(loaders, "youtube_download", fake_youtube_download)
    mmused(mmused_root, mode="audio-only")
    clips = mmused_root / "MMUSED" / "audio_clips"
    assert sorted(p.name for p in (clips / "d1").iterdir()) == [
        "0.wav",
        "1.wav",
        "2.wav",
    ]
    assert soundfile.info(clips / "d1" / "2.wav").duration == pytest.approx(0.1)
    assert not (mmused_root / "MMUSED" / "audio_recordings").exists()


def zip_archive(files: dict) -> bytes:
    """A zip archive of ``files``, as a Zenodo file link serves it."""
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return content.getvalue()


def test_mmused_downloads_its_zenodo_record(monkeypatch, tmp_path):
    pickled = io.BytesIO()
    MMUSED_ROWS.to_pickle(pickled)
    content = zip_archive(
        {"dataset.pkl": pickled.getvalue(), "download_links.csv": LINKS}
    )
    monkeypatch.setattr(
        loaders, "download", lambda url, file_path: file_path.write_bytes(content)
    )
    assert len(mmused(tmp_path).data) == 4
    assert sorted(p.name for p in (tmp_path / "MMUSED").iterdir()) == [
        "dataset.pkl",
        "download_links.csv",
    ]


FALLACY_ROWS = pd.DataFrame(
    {
        "dialogue_id": ["d1", "d1"],
        "filename": ["f1", "f1"],
        "fallacy": ["AdHominem", "Slogans"],
        "dialogue_indexes": [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5]],
        "dialogue_sentences": [
            ["s0", "s1", "s2", "s3"],
            ["s0", "s1", "s2", "s3", "s4", "s5"],
        ],
        "dialogue_start_time": [[0.0, 0.1, 0.2, 0.3], [0.0] * 6],
        "dialogue_end_time": [[0.1, 0.2, 0.3, 0.4], [0.1] * 6],
        "snippet_indexes": [[2, 3], [5]],
        "snippet_sentences": [["s2", "s3"], ["s5"]],
        "snippet_start_time": [[0.2, 0.3], [0.5]],
        "snippet_end_time": [[0.3, 0.4], [0.6]],
        "snippet": ["s2 s3", "s5"],
    }
)


@pytest.fixture
def fallacy_root(tmp_path):
    folder = tmp_path / "MMUSED-fallacy"
    folder.mkdir()
    FALLACY_ROWS.to_pickle(folder / "dataset.pkl")
    (folder / "download_links.csv").write_text(LINKS.replace("d2;u2;0;0;0;0\n", ""))
    return tmp_path


def fallacy(root, task="afc", mode="text-only", **kwargs):
    return loaders.MMUSEDFallacy(
        task_name=task, input_mode=mode, base_data_path=root, **kwargs
    )


def test_mmused_fallacy_afc_keeps_previous_sentences_as_context(fallacy_root):
    data = fallacy(fallacy_root, context_window=2).data
    assert list(data.fallacy) == [2, 5]
    assert list(data.dialogue) == ["s0 s1", "s3 s4"]
    assert [p.name for p in data.dialogue_paths[1]] == ["3.wav", "4.wav"]


def test_mmused_fallacy_afd_labels_every_snippet_sentence(fallacy_root):
    # The old loader paired snippet sentences with one fallacy per snippet,
    # so it dropped the sentences after the first of each dialogue's snippets.
    data = fallacy(fallacy_root, task="afd", context_window=1).data
    assert list(data.sentence) == ["s0", "s1", "s2", "s3", "s4", "s5"]
    assert list(data.label) == [0, 0, 1, 1, 0, 1]
    assert list(data.context) == ["", "s0", "s1", "s2", "s3", "s4"]


def test_mmused_fallacy_context_reaches_the_dataset(fallacy_root):
    (fallacy_root / "MMUSED-fallacy" / "audio_clips").mkdir()
    loader = fallacy(fallacy_root, mode="text-audio", with_context=True)
    dataset = loader.dataset(loader.data)
    assert dataset.context["text"] == ["s0 s1", "s2 s3 s4"]
    assert [p.name for p in dataset.context["audio"][0]] == ["0.wav", "1.wav"]
    assert fallacy(fallacy_root).dataset(loader.data).context is None


def test_mmused_fallacy_cuts_dialogue_and_snippet_clips(fallacy_root, monkeypatch):
    fallacy(fallacy_root, task="afd")
    monkeypatch.setattr(loaders, "youtube_download", fake_youtube_download)
    fallacy(fallacy_root, mode="audio-only")
    clips = fallacy_root / "MMUSED-fallacy" / "audio_clips" / "d1"
    assert sorted(p.name for p in clips.iterdir()) == [f"{i}.wav" for i in range(6)]


MARG_ROWS = pd.DataFrame(
    {
        "sentence_1": ["a", "b", "c"],
        "sentence_2": ["x", "y", "z"],
        "sentence_1_audio_path": ["a1.wav", "b1.wav", "c1.wav"],
        "sentence_2_audio_path": ["x2.wav", "y2.wav", "z2.wav"],
        "relation": ["support", "attack", "neither"],
    }
)


@pytest.fixture
def marg_root(tmp_path):
    folder = tmp_path / "MArg"
    folder.mkdir()
    MARG_ROWS.to_csv(folder / "final_dataset_0.85.csv", index=False)
    return tmp_path


def marg(root, mode="text-audio"):
    return loaders.MArg(
        task_name="arc", input_mode=mode, base_data_path=root, confidence=0.85
    )


def test_marg_builds_pairs(marg_root):
    dataset = marg(marg_root).dataset(marg(marg_root).data)
    assert dataset.labels == [1, 2, 0]
    assert dataset.inputs["a"]["text"] == ["a", "b", "c"]
    assert dataset.inputs["b"]["audio"][0] == Path("x2.wav")
    assert dataset[0] == (
        {
            "a": {"text": "a", "audio": Path("a1.wav")},
            "b": {"text": "x", "audio": Path("x2.wav")},
        },
        1,
    )


def test_marg_accepts_only_published_confidences(marg_root):
    with pytest.raises(ValueError, match="confidence"):
        loaders.MArg(
            task_name="arc",
            input_mode="text-only",
            base_data_path=marg_root,
            confidence=0.5,
        )


@pytest.fixture(scope="module")
def registry():
    valid, invalid = Registry.build(directory=Path(mamkit.__file__).parent)
    assert not invalid
    return valid


def test_every_declared_split_key_is_registered(registry):
    declared = {value for name, value in vars(keys).items() if name.endswith("_SPLIT")}
    assert declared <= registry


def sizes(split):
    return [None if part is None else len(part) for part in split]


def test_all_train_split(registry, ukdebates_root):
    loader = ukdebates(ukdebates_root)
    [split] = Registry.from_key(keys.ALL_TRAIN_SPLIT)(loader)
    assert sizes(split) == [4, None, None]


def test_mmused_split_reads_the_split_column(registry, mmused_root):
    [split] = Registry.from_key(keys.COLUMN_SPLIT)(mmused(mmused_root))
    assert sizes(split) == [2, 1, 1]
    assert split.test.inputs["text"] == ["good plan"]


FOLDS = {
    "fold_10": {"train": [2, 3], "validation": [1], "test": [0]},
    "fold_2": {"train": [0, 1], "validation": [2], "test": [3]},
}


def test_fold_files_follow_fold_numbers(registry, ukdebates_root, monkeypatch):
    def fake_download(url, file_path):
        assert url.endswith("/aaai2016_all_folds.json")
        file_path.write_text(json.dumps(FOLDS))

    monkeypatch.setattr("mamkit.components.splits.download", fake_download)
    strategy = Registry.from_key(keys.UKDEBATES_MANCINI_2022_SPLIT)
    splits = strategy(ukdebates(ukdebates_root))
    assert [split.test.inputs["text"] for split in splits] == [
        ["good evening\n"],
        ["we will act\n"],
    ]
    # The file is kept under its own name and read again without a download.
    assert (ukdebates_root / "UKDebates" / "aaai2016_all_folds.json").is_file()
    monkeypatch.setattr("mamkit.components.splits.download", no_download)
    assert len(strategy(ukdebates(ukdebates_root))) == 2


def test_marg_folds_name_their_confidence(registry):
    strategy = Registry.from_key(keys.MARG_085_MANCINI_2022_SPLIT)
    assert strategy.url.endswith("/m_arg_folds_0.85.json")


class Dialogues:
    """A loader stand-in whose rows are dialogue ids."""

    def __init__(self, dialogues):
        self.data = pd.DataFrame({"dialogue_id": dialogues})

    def dataset(self, df):
        return sorted(set(df.dialogue_id))


DIALOGUES = [f"{index}_2020" for index in range(8)] + ["47_2024", "48_2024"]


def test_leave_one_dialogue_out_folds(registry):
    strategy = Registry.from_key(keys.MMUSED_FALLACY_MANCINI_2024_SPLIT)
    np.random.seed(0)
    splits = strategy(Dialogues(DIALOGUES))
    assert [test for _, _, test in splits] == [[d] for d in DIALOGUES[:8]]
    for train, val, test in splits:
        assert len(val) == 4
        assert not set(train) & set(val)
        assert set(train) | set(val) | set(test) == set(DIALOGUES)
    np.random.seed(0)
    assert strategy(Dialogues(DIALOGUES)) == splits


def test_leave_one_dialogue_out_seed_fixes_the_draw():
    from mamkit.components.splits import LeaveOneDialogueOut

    strategy = LeaveOneDialogueOut(["47_2024", "48_2024"], seed=3)
    np.random.seed(0)
    first = strategy(Dialogues(DIALOGUES))
    np.random.seed(1)
    assert strategy(Dialogues(DIALOGUES)) == first


def test_mm_argfallacy_2025_split_is_registered(registry):
    # #3: this split raised a KeyError when requested by name.
    strategy = Registry.from_key(keys.MMUSED_FALLACY_MM_ARGFALLACY_2025_SPLIT)
    [split] = strategy(Dialogues(DIALOGUES))
    assert split.train == DIALOGUES[:8]
    assert split.val is None
    assert split.test == ["47_2024", "48_2024"]


def test_build_once_retries_a_failed_build(tmp_path):
    path = tmp_path / "corpus"

    def failing(target):
        (target / "half.txt").write_text("")
        raise OSError("network down")

    with pytest.raises(OSError):
        loaders.build_once(path, failing)
    assert not path.exists()

    def complete(target):
        # The files of the failed attempt are still there to resume from.
        assert (target / "half.txt").exists()
        (target / "done.txt").write_text("")

    loaders.build_once(path, complete)
    assert sorted(p.name for p in path.iterdir()) == ["done.txt", "half.txt"]
    loaders.build_once(path, failing)


@pytest.mark.parametrize(
    "window, context", [(-1, ["s0 s1", "s0 s1 s2 s3 s4"]), (0, ["", ""])]
)
def test_mmused_fallacy_afc_context_window_keeps_all_or_none(
    fallacy_root, window, context
):
    data = fallacy(fallacy_root, context_window=window).data
    assert list(data.dialogue) == context


def test_split_sentences_downloads_the_nltk_model_once(monkeypatch):
    downloads = []

    def missing(name):
        raise LookupError(name)

    monkeypatch.setattr(loaders.nltk.data, "find", missing)
    monkeypatch.setattr(loaders.nltk, "download", lambda *a, **k: downloads.append(a))
    monkeypatch.setattr(loaders.nltk, "sent_tokenize", lambda text: text.split(". "))
    assert loaders.split_sentences("One. Two") == ["One", "Two"]
    assert downloads == [("punkt_tab",)]


# Interrupted builds (#32). Each test fails one build step of a loader, checks
# that the loader folder does not exist, and checks that a second run completes
# with the data of an uninterrupted run.


def fail_once(monkeypatch, owner, name):
    """Makes ``owner.name`` do its work and then raise, on its first call only.

    The work is done before the error, so the failed build leaves the files a
    real interruption would leave, and the retry has to cope with them.
    """
    original = getattr(owner, name)
    calls = []

    def flaky(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(name)
        if len(calls) == 1:
            raise OSError(f"{name} interrupted")
        return result

    monkeypatch.setattr(owner, name, flaky)
    return calls


def ukdebates_download(source):
    def fake_download(url, file_path):
        with tarfile.open(file_path, "w:gz") as archive:
            archive.add(source / "dataset", arcname="dataset")

    return fake_download


@pytest.mark.parametrize(
    "owner, name",
    [(loaders, "download"), (tarfile.TarFile, "extractall")],
    ids=["download", "extraction"],
)
def test_ukdebates_recovers_from_an_interrupted_build(
    monkeypatch, tmp_path, owner, name
):
    write_ukdebates(tmp_path / "source")
    monkeypatch.setattr(loaders, "download", ukdebates_download(tmp_path / "source"))
    expected = ukdebates(tmp_path / "clean").data

    calls = fail_once(monkeypatch, owner, name)
    with pytest.raises(OSError, match="interrupted"):
        ukdebates(tmp_path / "root")
    # Only the partial folder remains, with the archive inside it.
    assert [p.name for p in (tmp_path / "root").iterdir()] == ["UKDebates.part"]

    data = ukdebates(tmp_path / "root").data
    assert len(calls) > 1  # the retry ran the failed step again
    pd.testing.assert_frame_equal(
        data.drop(columns="audio"), expected.drop(columns="audio")
    )
    assert not (tmp_path / "root" / "UKDebates.part").exists()


def mmused_record(rows, links):
    pickled = io.BytesIO()
    rows.to_pickle(pickled)
    content = zip_archive(
        {"dataset.pkl": pickled.getvalue(), "download_links.csv": links}
    )
    return lambda url, file_path: file_path.write_bytes(content)


ZENODO_CORPORA = {
    "MMUSED": (
        lambda root, mode: mmused(root, mode=mode),
        lambda: mmused_record(MMUSED_ROWS, LINKS),
    ),
    "MMUSED-fallacy": (
        lambda root, mode: fallacy(root, mode=mode),
        lambda: mmused_record(FALLACY_ROWS, LINKS.replace("d2;u2;0;0;0;0\n", "")),
    ),
}


@pytest.mark.parametrize("corpus", sorted(ZENODO_CORPORA))
@pytest.mark.parametrize(
    "owner, name",
    [
        (loaders, "download"),
        (zipfile.ZipFile, "extractall"),
        (loaders, "youtube_download"),
        (loaders, "export_clip"),
    ],
    ids=["download", "extraction", "recordings", "clips"],
)
def test_zenodo_corpora_recover_from_an_interrupted_build(
    monkeypatch, tmp_path, corpus, owner, name
):
    build, record = ZENODO_CORPORA[corpus]
    monkeypatch.setattr(loaders, "download", record())
    monkeypatch.setattr(loaders, "youtube_download", fake_youtube_download)
    expected = build(tmp_path / "clean", "audio-only").data

    calls = fail_once(monkeypatch, owner, name)
    with pytest.raises(OSError, match="interrupted"):
        build(tmp_path / "root", "audio-only")
    folder = tmp_path / "root" / corpus
    clips = folder / "audio_clips"
    if name in ("youtube_download", "export_clip"):
        # The corpus is complete, and only its clip folder is missing.
        assert folder.exists() and not clips.exists()
    else:
        assert not folder.exists()
    partial = folder / "audio_clips.part"
    kept = {
        path.relative_to(partial): path.stat().st_mtime_ns
        for path in partial.glob("*/*.wav")
    }
    if name == "export_clip":
        assert kept, "the failed clip build cut no clip"

    data = build(tmp_path / "root", "audio-only").data
    assert len(calls) > 1  # the retry ran the failed step again
    assert len(data) == len(expected)
    assert sorted(p.relative_to(clips).as_posix() for p in clips.rglob("*.wav")) == (
        sorted(
            p.relative_to(tmp_path / "clean" / corpus / "audio_clips").as_posix()
            for p in (tmp_path / "clean" / corpus / "audio_clips").rglob("*.wav")
        )
    )
    # A retried clip build keeps the clips that the failed build already cut.
    for relative, written in kept.items():
        assert clips.joinpath(relative).stat().st_mtime_ns == written
    assert not list(tmp_path.joinpath("root").rglob("*.part"))


def marg_record():
    source = "rafamestre-m-arg_multimodal-argumentation-dataset-851736f"
    content = zip_archive(
        {f"{source}/final_dataset_0.85.csv": MARG_ROWS.to_csv(index=False)}
    )
    return lambda url, file_path: file_path.write_bytes(content)


@pytest.mark.parametrize(
    "owner, name",
    [
        (loaders, "download"),
        (zipfile.ZipFile, "extractall"),
        (loaders.shutil, "copytree"),
    ],
    ids=["download", "extraction", "copy"],
)
def test_marg_recovers_from_an_interrupted_download(monkeypatch, tmp_path, owner, name):
    monkeypatch.setattr(loaders, "download", marg_record())
    expected = marg(tmp_path / "clean").data

    calls = fail_once(monkeypatch, owner, name)
    with pytest.raises(OSError, match="interrupted"):
        marg(tmp_path / "root")
    assert not (tmp_path / "root" / "MArg").exists()

    data = marg(tmp_path / "root").data
    assert len(calls) > 1  # the retry ran the failed step again
    pd.testing.assert_frame_equal(data, expected)
    assert not (tmp_path / "root" / "MArg.part").exists()


def write_marg_sources(folder: Path, debates):
    """The M-Arg source files with one sentence per debate, and two pairs."""
    data = folder / "data"
    for name in ("original data", "tokenised text", "timestamps", "split audio"):
        (data / name).mkdir(parents=True, exist_ok=True)
    (data / "preprocessed full dataset").mkdir()
    (folder / "annotated dataset").mkdir()
    contexts = []
    for timestamp_name, debate in debates.items():
        stem = re.sub(r"_(part)?\d+_timestamp\.csv$", "", timestamp_name)
        text = f"Sentence {debate}."
        # A transcript names each speaker once per paragraph.
        pd.DataFrame(
            {"speaker": ["Chris Wallace"], "minute": ["00:00"], "text": [text]}
        ).to_csv(data / "original data" / f"{stem}.csv", index=False, mode="a")
        pd.DataFrame([["t0", 0.5, 1.5, text]]).to_csv(
            data / "timestamps" / timestamp_name, header=False, index=False
        )
        (
            data / "split audio" / (timestamp_name.split("_timestamp")[0] + ".mp3")
        ).touch()
        contexts.append({"id": f"n{debate}0000", "context": "Ignore"})
    pd.DataFrame(contexts).to_csv(
        data / "preprocessed full dataset" / "dataset_context_definition.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "pair_id": [1, 2],
            "relation": ["support", "attack"],
            "relation:confidence": [1.0, 0.5],
            "sentence_1": ["Sentence 00.", "Sentence 01."],
            "sentence_2": ["Sentence 16.", "Sentence 02."],
        }
    ).to_csv(folder / "annotated dataset" / "aggregated_dataset.csv", index=False)


def test_marg_builds_its_dataset_from_the_source_files(monkeypatch, tmp_path):
    from pydub import AudioSegment

    folder = tmp_path / "MArg"
    # A loader built without its data gives the debate names to write.
    monkeypatch.setattr(loaders, "build_once", lambda path, build: None)
    monkeypatch.setattr(loaders.MArg, "build", lambda self: None)
    loader = loaders.MArg(
        task_name="arc",
        input_mode="text-only",
        confidence=0.85,
        base_data_path=tmp_path,
    )
    monkeypatch.undo()
    write_marg_sources(folder, loader.debate_id_map)
    monkeypatch.setattr(loaders, "split_sentences", lambda text: [text])
    monkeypatch.setattr(
        loaders.AudioSegment, "from_mp3", lambda path: AudioSegment.silent(3000)
    )

    data = marg(tmp_path).data
    assert list(data.relation) == [1]
    assert list(data.sentence_1) == ["Sentence 00."]
    assert [p.name for p in data.sentence_2_audio_path] == ["a160000.wav"]
    # The clip keeps its start, which is within the buffer, and 2 s after its end.
    clip = AudioSegment.from_wav(data.sentence_2_audio_path[0])
    assert len(clip) == 2500
    features = pd.read_csv(loader.feature_path, dtype={"speaker_id": str})
    assert list(features.speaker_id.unique()) == ["00"]
    assert features.context.isna().all()
