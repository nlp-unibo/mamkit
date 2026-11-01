"""Tokenizer, vocabulary, and GloVe reader that replace torchtext."""

import json
import zipfile

import pytest
from conftest import GOLDEN

from mamkit.data.processing import VocabBuilder
from mamkit.utility.text import (
    Vocabulary,
    basic_english,
    build_vocabulary,
    glove_vectors,
)

# Recorded with torchtext 0.18 before its removal.
REFERENCE = json.loads((GOLDEN / "basic_english.json").read_text())


def test_basic_english_matches_torchtext():
    assert [basic_english(text) for text in REFERENCE["corpus"]] == REFERENCE["tokens"]


@pytest.mark.parametrize("min_freq", [1, 2])
def test_build_vocabulary_matches_torchtext(min_freq):
    vocabulary = build_vocabulary(
        REFERENCE["vocabulary_corpus"], basic_english, min_freq=min_freq
    )
    assert vocabulary.get_itos() == REFERENCE[f"vocabulary_min_freq_{min_freq}"]


def test_vocabulary_maps_unknown_tokens_to_unk():
    vocabulary = Vocabulary(["<pad>", "<unk>", "cat"])
    assert vocabulary(["cat", "dog"]) == [2, 1]
    assert len(vocabulary) == 3


LINES = "the 1 2\ncat 3 4\nthe 5 6\n"


@pytest.fixture
def cache(tmp_path):
    with zipfile.ZipFile(tmp_path / "glove.6B.zip", "w") as files:
        files.writestr("glove.6B.50d.txt", LINES)
        files.writestr("glove.6B.100d.txt", "unused 0\n")
    return tmp_path


def test_glove_vectors_extracts_one_file_and_zeroes_unknown_tokens(cache):
    # A word listed twice keeps its last vector, as in torchtext.
    vectors = glove_vectors("glove.6B.50d", ["cat", "dog", "the"], cache=cache)
    assert vectors.tolist() == [[3.0, 4.0], [0.0, 0.0], [5.0, 6.0]]
    assert (cache / "glove.6B.50d.txt").exists()
    assert not (cache / "glove.6B.100d.txt").exists()


def test_glove_vectors_rejects_other_models(cache):
    with pytest.raises(ValueError, match="glove.6B"):
        glove_vectors("fasttext.en.300d", ["cat"], cache=cache)


def test_vocab_builder_builds_glove_embedding_matrix(cache, monkeypatch):
    (cache / ".vector_cache").mkdir()
    (cache / ".vector_cache" / "glove.6B.50d.txt").write_text(LINES)
    monkeypatch.chdir(cache)
    builder = VocabBuilder(
        basic_english, embedding_dim=50, embedding_model="glove.6B.50d"
    )
    builder.fit(inputs=["the cat", "a cat"])
    assert builder.vocab.get_itos() == ["<pad>", "<unk>", "cat", "a", "the"]
    assert builder.embedding_matrix.tolist()[2:] == [[3.0, 4.0], [0.0, 0.0], [5.0, 6.0]]
