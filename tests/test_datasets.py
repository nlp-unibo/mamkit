"""Dataset containers and the UKDebates loader on a prepared data directory.

The other loaders download archives and cut audio clips from remote
recordings; their tests arrive with the loader rewrite.
"""

import json

import numpy as np
import pandas as pd
import pytest

from mamkit.data.datasets import (
    InputMode,
    MMUSEDFallacy,
    MultimodalDataset,
    PairMultimodalDataset,
    PairUnimodalDataset,
    UKDebates,
    UnimodalDataset,
)


def test_unimodal_dataset_items():
    data = UnimodalDataset(inputs=["a", "b"], labels=[0, 1], context=["x", "y"])
    assert len(data) == 2
    assert data[1] == ("b", 1, "y")
    assert UnimodalDataset(inputs=["a"], labels=[0])[0] == ("a", 0, None)


def test_multimodal_dataset_items():
    data = MultimodalDataset(texts=["t"], audio=["w"], labels=[1])
    assert data[0] == ("t", "w", 1, None, None)


def test_pair_unimodal_dataset_items():
    data = PairUnimodalDataset(a_inputs=["a"], b_inputs=["b"], labels=[2])
    assert data[0] == ("a", "b", 2, None, None)


def test_pair_multimodal_dataset_items():
    data = PairMultimodalDataset(
        a_texts=["a"], b_texts=["b"], a_audio=["x"], b_audio=["y"], labels=[0]
    )
    assert data[0] == ("a", "b", "x", "y", 0, None, None, None, None)


SPEAKERS = {
    "Miliband": (["1 we will act\n", "2 thank you\n"], ["C\n", "N\n"]),
    "Clegg": (["1 fair taxes now\n"], ["C\n"]),
    "Cameron": (["1 good evening\n"], ["N\n"]),
}


@pytest.fixture
def ukdebates_root(tmp_path):
    dataset = tmp_path / "UKDebates" / "dataset"
    dataset.mkdir(parents=True)
    for speaker, (texts, labels) in SPEAKERS.items():
        (dataset / f"{speaker}.txt").write_text("".join(texts))
        (dataset / f"{speaker}Labels.txt").write_text("".join(labels))
    folds = {
        "fold_1": {"train": [2, 3], "validation": [1], "test": [0]},
        "fold_0": {"train": [0, 1], "validation": [2], "test": [3]},
    }
    (tmp_path / "UKDebates" / "mancini_2022_folds.json").write_text(json.dumps(folds))
    return tmp_path


def ukdebates(root, mode=InputMode.TEXT_ONLY):
    return UKDebates(task_name="asd", input_mode=mode, base_data_path=root)


def test_ukdebates_parses_speakers_in_order(ukdebates_root):
    loader = ukdebates(ukdebates_root)
    assert loader.texts == [
        "we will act\n",
        "thank you\n",
        "fair taxes now\n",
        "good evening\n",
    ]
    assert loader.labels == [1, 0, 1, 0]
    assert loader.audio[2].parts[-2:] == ("Clegg", "1.wav")


def test_ukdebates_mancini_splits_follow_fold_order(ukdebates_root):
    splits = ukdebates(ukdebates_root).get_splits("mancini-et-al-2022")
    assert len(splits) == 2
    assert list(splits[0].train.labels) == [1, 0]
    assert list(splits[0].test.inputs) == ["good evening\n"]
    assert list(splits[1].test.inputs) == ["we will act\n"]


def test_ukdebates_builds_a_dataset_per_input_mode(ukdebates_root):
    audio = ukdebates(ukdebates_root, InputMode.AUDIO_ONLY)
    split = audio.get_splits("mancini-et-al-2022")[0]
    assert split.train.inputs[0].name == "1.wav"
    both = ukdebates(ukdebates_root, InputMode.TEXT_AUDIO)
    split = both.get_splits("mancini-et-al-2022")[0]
    assert isinstance(split.train, MultimodalDataset)


def test_ukdebates_default_splits_train_on_all_data(ukdebates_root):
    [split] = ukdebates(ukdebates_root).get_splits("default")
    assert len(split.train.inputs) == 4
    assert len(split.val.inputs) == 0 and len(split.test.inputs) == 0


def test_ukdebates_supports_only_asd(ukdebates_root):
    with pytest.raises(AssertionError):
        UKDebates(
            task_name="acc",
            input_mode=InputMode.TEXT_ONLY,
            base_data_path=ukdebates_root,
        )


def test_mmused_fallacy_mancini_splits_do_not_depend_on_hashing(monkeypatch, tmp_path):
    # Folds follow sorted dialogue ids, and each fold draws four distinct
    # validation dialogues, so a numpy seed fixes every split.
    monkeypatch.setattr(MMUSEDFallacy, "load", lambda self: None)
    loader = MMUSEDFallacy(
        task_name="afc", input_mode=InputMode.TEXT_ONLY, base_data_path=tmp_path
    )
    dialogues = [f"{index}_2020" for index in range(8)] + ["47_2024", "48_2024"]
    loader.__dict__["data"] = pd.DataFrame({"dialogue_id": dialogues})
    loader.build_info_from_splits = lambda train_df, val_df, test_df: (
        train_df,
        val_df,
        test_df,
    )
    np.random.seed(0)
    splits = list(loader.get_splits("mancini-et-al-2024"))
    assert [test.dialogue_id.item() for _, _, test in splits] == dialogues[:8]
    for train, val, test in splits:
        assert val.dialogue_id.nunique() == 4
        assert not set(train.dialogue_id) & set(val.dialogue_id)
