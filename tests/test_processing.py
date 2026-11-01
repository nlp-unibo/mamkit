"""Processor outputs, including the defects the refactoring removes."""

import numpy as np
import pytest
import torch as th
from conftest import BERT, HUBERT, WAV2VEC, golden
from torchtext.data.utils import get_tokenizer

from mamkit.data import processing
from mamkit.data.datasets import (
    MultimodalDataset,
    PairMultimodalDataset,
    PairUnimodalDataset,
    UnimodalDataset,
)
from mamkit.utility.processing import parse_audio_nn

TOKENIZER = get_tokenizer("basic_english")


def vocabulary_builder(**tokenization_args):
    return processing.VocabBuilder(
        tokenizer=TOKENIZER, embedding_dim=4, tokenization_args=tokenization_args
    )


def test_vocab_builder_puts_specials_first():
    builder = vocabulary_builder()
    builder.fit(inputs=["the cat", "the dog"])
    tokens = builder.vocab.get_itos()
    assert tokens[:3] == ["<pad>", "<unk>", "the"]
    assert sorted(tokens[3:]) == ["cat", "dog"]
    assert builder.vocab["bird"] == builder.vocab["<unk>"]
    assert builder.embedding_matrix is None


def test_vocab_builder_counts_repeated_texts_without_context():
    builder = vocabulary_builder(min_freq=2)
    builder.fit(inputs=["cat", "cat"])
    assert "cat" in builder.vocab.get_itos()


def test_vocab_builder_counts_repeated_texts_once_with_context():
    # Defect: with a context, texts are deduplicated before counting.
    builder = vocabulary_builder(min_freq=2)
    builder.fit(inputs=["cat", "cat"], context=["dog"])
    assert "cat" not in builder.vocab.get_itos()


def test_pair_vocab_builder_counts_repeated_texts_once():
    # Defect: both sides are deduplicated before counting.
    builder = processing.PairVocabBuilder(
        tokenizer=TOKENIZER, embedding_dim=4, tokenization_args={"min_freq": 2}
    )
    builder.fit(a_inputs=["cat", "dog"], b_inputs=["cat", "bird"])
    assert "cat" not in builder.vocab.get_itos()
    builder.fit(a_inputs=["cat dog"], b_inputs=["cat bird"])
    assert "cat" in builder.vocab.get_itos()


def test_vocab_builder_leaves_texts_unchanged():
    builder = vocabulary_builder()
    builder.fit(inputs=["the cat"])
    assert builder(inputs=["the cat"], context=None) == (["the cat"], None)


@pytest.fixture
def mfcc(tmp_path):
    return processing.MFCCExtractor(mfccs=13, serialization_path=tmp_path / "m.pkl")


def test_mfcc_extractor_features(mfcc, audio_files):
    inputs = [audio_files["short"], [audio_files["short"], audio_files["long"]]]
    features, context = mfcc(inputs=inputs, context=[[audio_files["long"]], []])
    golden("processing/mfcc", {"features": features, "context": context})


def test_mfcc_extractor_resamples_to_target_rate(mfcc, audio_files):
    features, _ = mfcc(inputs=[audio_files["resampled"]])
    golden("processing/mfcc_resampled", features)


def test_mfcc_extractor_encodes_missing_audio_as_zero(mfcc, tmp_path):
    features, context = mfcc(inputs=[tmp_path / "missing.wav", None])
    assert [item.tolist() for item in features] == [[0.0], [0.0]]
    assert context is None


def test_mfcc_extractor_writes_its_cache(mfcc, audio_files):
    mfcc(inputs=[audio_files["short"]])
    assert mfcc.serialization_path.exists()


def test_pair_mfcc_extractor_matches_single_extractor(mfcc, audio_files, tmp_path):
    pair = processing.PairMFCCExtractor(
        mfccs=13, serialization_path=tmp_path / "pair.pkl"
    )
    a, b, _, _ = pair(a_inputs=[audio_files["short"]], b_inputs=[audio_files["long"]])
    single, _ = mfcc(inputs=[audio_files["short"], audio_files["long"]])
    np.testing.assert_allclose(a[0], single[0])
    np.testing.assert_allclose(b[0], single[1])


def text_transformer():
    return processing.TextTransformer(model_card=BERT)


def test_text_transformer_embeddings():
    features, context = text_transformer()(
        inputs=["the cat sat", ""], context=["a dog", "a cat"]
    )
    golden("processing/text_transformer", {"features": features, "context": context})


def test_text_transformer_clear_releases_model():
    processor = text_transformer()
    processor(inputs=["cat"])
    processor.clear()
    assert processor.model is None and processor.tokenizer is None


def test_pair_text_transformer_matches_single_transformer():
    pair = processing.PairTextTransformer(model_card=BERT)
    a, b, _, _ = pair(a_inputs=["the cat"], b_inputs=["a dog"])
    single, _ = text_transformer()(inputs=["the cat", "a dog"])
    np.testing.assert_allclose(a[0], single[0], rtol=1e-5)
    np.testing.assert_allclose(b[0], single[1], rtol=1e-5)


def audio_transformer(**kwargs):
    return processing.AudioTransformer(
        model_card=WAV2VEC, sampling_rate=16_000, **kwargs
    )


def test_audio_transformer_features(audio_files):
    features, context = audio_transformer()(
        inputs=[audio_files["short"], [audio_files["short"], audio_files["long"]]],
        context=[[audio_files["long"]], None],
    )
    golden("processing/audio_transformer", {"features": features, "context": context})


def test_audio_transformer_downsamples(audio_files):
    processor = audio_transformer(downsampling_factor=0.5)
    features, _ = processor(inputs=[audio_files["long"]])
    golden("processing/audio_transformer_downsampled", features)


def test_audio_transformer_aggregate_averages_frames(audio_files):
    features, _ = audio_transformer()(inputs=[audio_files["long"]])
    aggregated, _ = audio_transformer(aggregate=True)(inputs=[audio_files["long"]])
    assert features[0].shape[0] > 1
    np.testing.assert_allclose(
        aggregated[0], features[0].mean(axis=0, keepdims=True), rtol=1e-5
    )


def test_audio_transformer_extractor_features(audio_files):
    processor = processing.AudioTransformerExtractor(
        model_card=HUBERT, sampling_rate=16_000
    )
    features, _ = processor(inputs=[audio_files["short"]])
    golden("processing/audio_transformer_extractor", features)


def test_pair_audio_transformer_matches_single_transformer(audio_files):
    pair = processing.PairAudioTransformer(model_card=WAV2VEC, sampling_rate=16_000)
    a, b, a_context, b_context = pair(
        a_inputs=[audio_files["short"]],
        b_inputs=[audio_files["long"]],
        a_context=[[audio_files["long"]]],
        b_context=[[audio_files["short"]]],
    )
    single, context = audio_transformer()(
        inputs=[audio_files["short"], audio_files["long"]],
        context=[[audio_files["long"]], [audio_files["short"]]],
    )
    np.testing.assert_allclose(a[0], single[0], rtol=1e-5)
    np.testing.assert_allclose(b[0], single[1], rtol=1e-5)
    np.testing.assert_allclose(a_context[0], context[0], rtol=1e-5)
    np.testing.assert_allclose(b_context[0], context[1], rtol=1e-5)


def test_parse_audio_nn_resamples_to_target_rate(audio_files):
    # The clip lasts half a second at 22.05 kHz.
    audio = parse_audio_nn(audio_files["resampled"], sampling_rate=16_000)
    assert audio.shape == (1, 8_000)


class Upper:
    """A feature processor that fits nothing and upper-cases its inputs."""

    def __init__(self):
        self.fitted = None
        self.cleared = False

    def fit(self, inputs, context=None):
        self.fitted = list(inputs)

    def __call__(self, inputs, context=None):
        return [item.upper() for item in inputs], context

    def clear(self):
        self.cleared = True


class PairUpper(Upper):
    def fit(self, a_inputs, b_inputs, a_context=None, b_context=None):
        self.fitted = list(a_inputs) + list(b_inputs)

    def __call__(self, a_inputs, b_inputs, a_context=None, b_context=None):
        upper = super().__call__
        return upper(a_inputs)[0], upper(b_inputs)[0], a_context, b_context


def test_unimodal_processor_fits_and_transforms_features():
    features = Upper()
    processor = processing.UnimodalProcessor(features_processor=features)
    data = UnimodalDataset(inputs=["a", "b"], labels=[0, 1])
    processor.fit(data)
    assert features.fitted == ["a", "b"]
    assert processor(data).inputs == ["A", "B"]
    processor.clear()
    assert features.cleared


def test_multimodal_processor_routes_each_modality():
    text, audio = Upper(), Upper()
    processor = processing.MultimodalProcessor(
        text_processor=text, audio_processor=audio
    )
    data = MultimodalDataset(texts=["t"], audio=["x"], labels=[0])
    processor.fit(data)
    data = processor(data)
    assert (data.texts, data.audio) == (["T"], ["X"])
    processor.clear()
    assert text.cleared and audio.cleared


def test_pair_unimodal_processor_fits_on_both_sides():
    features = PairUpper()
    processor = processing.PairUnimodalProcessor(features_processor=features)
    data = PairUnimodalDataset(a_inputs=["a"], b_inputs=["b"], labels=[0])
    processor.fit(data)
    assert features.fitted == ["a", "b"]
    data = processor(data)
    assert (data.a_inputs, data.b_inputs) == (["A"], ["B"])


def test_pair_multimodal_processor_routes_each_modality_and_side():
    processor = processing.PairMultimodalProcessor(
        text_processor=PairUpper(), audio_processor=PairUpper()
    )
    data = PairMultimodalDataset(
        a_texts=["t"], b_texts=["u"], a_audio=["x"], b_audio=["y"], labels=[0]
    )
    processor.fit(data)
    data = processor(data)
    assert (data.a_texts, data.b_texts, data.a_audio, data.b_audio) == (
        ["T"],
        ["U"],
        ["X"],
        ["Y"],
    )


def test_processors_without_components_return_data_unchanged():
    data = UnimodalDataset(inputs=["a"], labels=[th.tensor(0)])
    processor = processing.UnimodalProcessor()
    processor.fit(data)
    assert processor(data).inputs == ["a"]
