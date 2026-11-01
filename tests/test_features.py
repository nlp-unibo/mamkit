"""Featurizers reproduce the processor and collator golden files."""

import numpy as np
import pytest
import torch as th
from conftest import BERT, HUBERT, WAV2VEC, stored, tensors

from mamkit.components import encoders, features
from mamkit.components.datasets import MAMDataset
from mamkit.components.models import Classifier
from mamkit.data import collators, processing
from mamkit.utility.text import basic_english


def same(actual, name, key=None):
    expected = stored(name)
    if key is not None:
        expected = expected[key]
    th.testing.assert_close(tensors(actual), expected, rtol=1e-4, atol=1e-5)


@pytest.fixture
def mfcc(tmp_path):
    return features.MFCCAudio(mfccs=13, cache_path=tmp_path / "mfccs.pkl")


def test_mfcc_audio_matches_golden(mfcc, audio_files):
    inputs = [audio_files["short"], [audio_files["short"], audio_files["long"]]]
    same(mfcc.transform(inputs), "processing/mfcc", "features")


def test_mfcc_audio_resamples_to_target_rate(mfcc, audio_files):
    same(mfcc.transform([audio_files["resampled"]]), "processing/mfcc_resampled")


def test_mfcc_audio_cache_is_reused_by_a_new_featurizer(mfcc, audio_files, tmp_path):
    [expected] = mfcc.transform([audio_files["short"]])
    reloaded = features.MFCCAudio(mfccs=13, cache_path=tmp_path / "mfccs.pkl")
    # The keys are paths, not hashes, so that a later process finds them.
    assert list(reloaded.cache) == [audio_files["short"].as_posix()]
    np.testing.assert_array_equal(
        reloaded.cache[audio_files["short"].as_posix()], expected
    )


def test_encoded_text_matches_golden():
    featurizer = features.EncodedText(model_card=BERT)
    same(
        featurizer.transform(["the cat sat", ""]),
        "processing/text_transformer",
        "features",
    )
    featurizer.clear()
    assert featurizer.model is None


def encoded_audio(**kwargs):
    return features.EncodedAudio(model_card=WAV2VEC, sampling_rate=16_000, **kwargs)


def test_encoded_audio_matches_golden(audio_files):
    inputs = [audio_files["short"], [audio_files["short"], audio_files["long"]]]
    same(encoded_audio().transform(inputs), "processing/audio_transformer", "features")


def test_encoded_audio_downsamples(audio_files):
    featurizer = encoded_audio(downsampling_factor=0.5)
    same(
        featurizer.transform([audio_files["long"]]),
        "processing/audio_transformer_downsampled",
    )


def test_encoded_audio_with_feature_extractor_matches_golden(audio_files):
    featurizer = features.EncodedAudio(
        model_card=HUBERT, sampling_rate=16_000, feature_extractor=True
    )
    same(
        featurizer.transform([audio_files["short"]]),
        "processing/audio_transformer_extractor",
    )


def test_tokenized_text_matches_golden():
    output = features.TokenizedText(model_card=BERT).collate(["the cat sat", None])
    expected = stored("collators/text_transformer")
    assert th.equal(output["inputs"], expected["inputs"])
    assert th.equal(output["input_mask"], expected["input_mask"])


TEXTS = ["The cat sat", "a cat", "the dog"]


def test_vocabulary_text_matches_vocab_builder_and_text_collator():
    old = processing.VocabBuilder(tokenizer=basic_english, embedding_dim=4)
    old.fit(inputs=TEXTS[:2], context=TEXTS[2:])
    new = features.VocabularyText()
    new.fit(TEXTS[:2], TEXTS[2:])
    assert new.vocabulary.get_itos() == old.vocab.get_itos()

    output = new.collate(TEXTS[:2] + [None])
    collator = collators.TextCollator(tokenizer=basic_english, vocab=old.vocab)
    assert th.equal(output["inputs"], collator(inputs=TEXTS[:2] + [None])["inputs"])
    assert output["input_mask"].tolist() == [[1, 1, 1], [1, 1, 0], [1, 0, 0]]
    assert new.model_inputs() == {"vocab_size": 7, "embedding_matrix": None}


def test_audio_collate_masks_padding_and_keeps_vectors_as_one_frame():
    output = features.Featurizer().collate([np.ones((2, 3)), np.ones(3)])
    assert output["inputs"].shape == (2, 2, 3)
    assert output["inputs"][1, 1].tolist() == [0.0, 0.0, 0.0]
    assert output["input_mask"].tolist() == [[1.0, 1.0], [1.0, 0.0]]


class Upper(features.Featurizer):
    """Records what it fits and upper-cases what it transforms."""

    def fit(self, inputs, context=None):
        self.fitted = (list(inputs), context)

    def transform(self, inputs):
        return [text.upper() for text in inputs]

    def collate(self, inputs):
        return list(inputs)


def pair_dataset():
    return MAMDataset(
        inputs={"a": {"text": ["a", "b"]}, "b": {"text": ["c", "d"]}},
        labels=[0, 1],
        context={"a": {"text": ["x", "y"]}, "b": {"text": ["z", "w"]}},
    )


def test_fit_reads_both_sides_of_a_pair_and_their_context():
    upper = Upper()
    features.fit(pair_dataset(), {"text": upper})
    assert upper.fitted == (["a", "b", "c", "d"], ["x", "y", "z", "w"])


def test_transform_and_collate_keep_the_pair_layout():
    featurizers = {"text": Upper()}
    dataset = features.transform(pair_dataset(), featurizers)
    assert dataset.inputs == {"a": {"text": ["A", "B"]}, "b": {"text": ["C", "D"]}}
    batch, labels = features.collate([dataset[0], dataset[1]], featurizers)
    assert batch == {"a": {"text": ["A", "B"]}, "b": {"text": ["C", "D"]}}
    assert labels.tolist() == [0, 1]


def test_padded_tokens_do_not_change_text_bilstm_logits():
    # A text gets the same logits alone and in a batch with a longer text.
    featurizers = {"text": features.VocabularyText()}
    dataset = MAMDataset({"text": ["the cat", "the cat sat on a mat"]}, [0, 1])
    features.fit(dataset, featurizers)
    th.manual_seed(0)
    model = Classifier(
        encoders.TextBiLSTM(
            embedding_dim=4, lstm_weights=[3], **featurizers["text"].model_inputs()
        ),
        head=[2],
    ).eval()
    alone, _ = features.collate([dataset[0]], featurizers)
    both, _ = features.collate([dataset[0], dataset[1]], featurizers)
    with th.no_grad():
        th.testing.assert_close(model(both)[:1], model(alone))
