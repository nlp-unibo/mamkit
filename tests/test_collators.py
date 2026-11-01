"""Collator outputs, including the defects the refactoring removes."""

import numpy as np
import torch as th
from conftest import BERT, golden

from mamkit.data import collators
from mamkit.utility.text import Vocabulary, basic_english

TOKENIZER = basic_english


def vocabulary():
    return Vocabulary(["<pad>", "<unk>", "the", "cat", "sat"])


def test_text_collator_pads_token_ids():
    collator = collators.TextCollator(tokenizer=TOKENIZER, vocab=vocabulary())
    output = collator(inputs=["The cat sat", "a cat"], context=["the dog"])
    assert output["inputs"].tolist() == [[2, 3, 4], [1, 3, 0]]
    assert output["context"].tolist() == [[2, 1]]


def test_text_collator_encodes_missing_text_as_padding():
    collator = collators.TextCollator(tokenizer=TOKENIZER, vocab=vocabulary())
    assert collator(inputs=[None, "cat"])["inputs"].tolist() == [[0], [3]]


def test_pair_text_collator_encodes_each_side():
    collator = collators.PairTextCollator(tokenizer=TOKENIZER, vocab=vocabulary())
    output = collator(a_inputs=["the cat"], b_inputs=["sat"])
    assert output["a_inputs"].tolist() == [[2, 3]]
    assert output["b_inputs"].tolist() == [[4]]


def test_text_transformer_collator_tokenizes():
    collator = collators.TextTransformerCollator(model_card=BERT)
    output = collator(inputs=["the cat sat", None], context=["a dog", "a cat"])
    golden("collators/text_transformer", output)


def test_pair_text_transformer_collator_matches_single_collator():
    pair = collators.PairTextTransformerCollator(model_card=BERT)
    output = pair(
        a_inputs=["the cat sat"],
        b_inputs=["something else entirely"],
        a_context=["a dog"],
        b_context=["a bird flew by"],
    )
    single = collators.TextTransformerCollator(model_card=BERT)
    for side, inputs, context in (
        ("a", ["the cat sat"], ["a dog"]),
        ("b", ["something else entirely"], ["a bird flew by"]),
    ):
        expected = single(inputs=inputs, context=context)
        for key, value in expected.items():
            assert th.equal(output[f"{side}_{key}"], value), f"{side}_{key}"


def test_text_transformer_output_collator_pads_embeddings():
    first = np.ones((2, 3), dtype=np.float32)
    second = np.full((1, 3), 2.0, dtype=np.float32)
    output = collators.TextTransformerOutputCollator()(inputs=[first, second])
    assert output["inputs"].shape == (2, 2, 3)
    assert output["input_mask"].tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert output["context"] is None


def test_pair_text_transformer_output_collator_pads_each_side():
    first = np.ones((2, 3), dtype=np.float32)
    second = np.ones((1, 3), dtype=np.float32)
    output = collators.PairTextTransformerOutputCollator()(
        a_inputs=[first, second], b_inputs=[second, second]
    )
    assert output["a_input_mask"].tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert output["b_input_mask"].tolist() == [[1.0], [1.0]]


def test_audio_collator_masks_padded_frames():
    first = np.ones((2, 3), dtype=np.float32)
    second = np.ones((1, 3), dtype=np.float32)
    output = collators.AudioCollatorOutput()(inputs=[first, second])
    assert output["inputs"][1, 1].tolist() == [0.0, 0.0, 0.0]
    assert output["input_mask"].tolist() == [[1.0, 1.0], [1.0, 0.0]]


def test_audio_collator_keeps_aggregated_vectors_as_one_frame():
    output = collators.AudioCollatorOutput()(
        inputs=[np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
    )
    assert output["inputs"].shape == (2, 1, 3)
    assert output["input_mask"].tolist() == [1.0, 1.0]


def test_pair_audio_collator_pads_each_side():
    first = np.ones((2, 3), dtype=np.float32)
    second = np.ones((1, 3), dtype=np.float32)
    output = collators.PairAudioOutputCollator()(
        a_inputs=[first, second], b_inputs=[second, second]
    )
    assert output["a_inputs"].shape == (2, 2, 3)
    assert output["b_inputs"].shape == (2, 1, 3)


def labels(values):
    return th.tensor(values)


def test_unimodal_collator_returns_features_and_labels():
    collator = collators.UnimodalCollator(
        features_collator=collators.TextCollator(TOKENIZER, vocabulary()),
        label_collator=labels,
    )
    features, targets = collator([("the cat", 1, None), ("sat", 0, None)])
    assert features["inputs"].tolist() == [[2, 3], [4, 0]]
    assert targets.tolist() == [1, 0]


def test_unimodal_collator_passes_raw_inputs_without_collators():
    collator = collators.UnimodalCollator(features_collator=None, label_collator=None)
    features, targets = collator([("the cat", 1, "ctx")])
    assert features == {"inputs": ("the cat",), "context": ("ctx",)}
    assert targets == (1,)


def test_multimodal_collator_prefixes_keys_by_modality():
    collator = collators.MultimodalCollator(
        text_collator=collators.TextCollator(TOKENIZER, vocabulary()),
        audio_collator=collators.AudioCollatorOutput(),
        label_collator=labels,
    )
    audio = np.ones((2, 3), dtype=np.float32)
    features, targets = collator([("the cat", audio, 1, None, None)])
    assert sorted(features) == [
        "audio_context",
        "audio_context_mask",
        "audio_input_mask",
        "audio_inputs",
        "text_context",
        "text_inputs",
    ]
    assert targets.tolist() == [1]


def test_pair_unimodal_collator_returns_both_sides():
    collator = collators.PairUnimodalCollator(
        features_collator=collators.PairTextCollator(TOKENIZER, vocabulary()),
        label_collator=labels,
    )
    features, targets = collator([("the", "cat", 2, None, None)])
    assert features["a_inputs"].tolist() == [[2]]
    assert features["b_inputs"].tolist() == [[3]]
    assert targets.tolist() == [2]


def test_pair_multimodal_collator_prefixes_keys_by_modality_and_side():
    collator = collators.PairMultimodalCollator(
        text_collator=collators.PairTextCollator(TOKENIZER, vocabulary()),
        audio_collator=collators.PairAudioOutputCollator(),
        label_collator=labels,
    )
    audio = np.ones((2, 3), dtype=np.float32)
    sample = ("the", "cat", audio, audio, 1, None, None, None, None)
    features, _ = collator([sample])
    assert features["text_a_inputs"].tolist() == [[2]]
    assert features["text_b_inputs"].tolist() == [[3]]
    assert features["audio_b_inputs"].shape == (1, 2, 3)
