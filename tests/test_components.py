"""Encoders, Classifier, and Ensemble reproduce the model golden files.

Each golden file was recorded from a model class in ``mamkit.models``. Its
state dict is loaded under the new parameter names, and its batch is converted
to the new layout, in which a sample maps a modality to its fields and a pair
batch maps ``a`` and ``b`` to samples.
"""

import copy

import pytest
import torch as th
from conftest import BERT, GOLDEN, stored

from mamkit.components import encoders
from mamkit.components.models import Classifier, Ensemble

DIM, VOCAB, HEAD = 8, 20, [6, 3]

TEXT_BILSTM = [
    ("embedding.", "encoder.encoders.0.embedding."),
    ("text_lstm.", "encoder.encoders.0.lstm."),
]
ENSEMBLE = [
    ("text_head.", "text.head."),
    ("audio_head.", "audio.head."),
    ("audio_encoder.", "audio.encoder.encoder."),
    ("positional_encoder.", "audio.encoder.pos_encoder."),
    ("layer_norm.", "audio.encoder.layer_norm."),
    ("weight", "weight"),
]


def text_bilstm():
    return encoders.TextBiLSTM(VOCAB, DIM, [6])


def audio_bilstm():
    return encoders.AudioBiLSTM(DIM, [6])


def audio_transformer():
    return encoders.AudioTransformer(DIM, n_heads=2, ffn_dim=16, n_layers=1)


def csa():
    return encoders.CSA(DIM, 16, 2, 1, encoder_dropout_rate=0.0)


def ensemble(pair=False):
    return Ensemble(
        Classifier(encoders.TextPool(DIM), HEAD, pair),
        Classifier(audio_transformer(), HEAD, pair),
    )


# Each golden file name maps to the default modality of its batch keys, a
# factory for the new model, and the renaming of its parameters. Parameters
# not renamed move under ``encoder.``, and the head keeps its name.
CASES = {
    "text_bilstm": ("text", lambda: Classifier(text_bilstm(), HEAD), []),
    "text_pair_bilstm": ("text", lambda: Classifier(text_bilstm(), HEAD, True), []),
    "text_transformer": (
        "text",
        lambda: Classifier(encoders.TextTransformer(BERT), HEAD),
        [],
    ),
    "text_pair_transformer": (
        "text",
        lambda: Classifier(encoders.TextTransformer(BERT), HEAD, True),
        [],
    ),
    "audio_bilstm": ("audio", lambda: Classifier(audio_bilstm(), HEAD), []),
    "audio_pair_bilstm": ("audio", lambda: Classifier(audio_bilstm(), HEAD, True), []),
    "audio_transformer": ("audio", lambda: Classifier(audio_transformer(), HEAD), []),
    "audio_pair_transformer": (
        "audio",
        lambda: Classifier(audio_transformer(), HEAD, True),
        [],
    ),
    "text_audio_bilstm": (
        None,
        lambda: Classifier(encoders.Concat([text_bilstm(), audio_bilstm()]), HEAD),
        TEXT_BILSTM + [("audio_lstm.", "encoder.encoders.1.lstm.")],
    ),
    "text_audio_pair_bilstm": (
        None,
        lambda: Classifier(
            encoders.Concat([text_bilstm(), audio_bilstm()]), HEAD, True
        ),
        TEXT_BILSTM + [("audio_lstm.", "encoder.encoders.1.lstm.")],
    ),
    "mmtransformer": (
        None,
        lambda: Classifier(
            encoders.Concat([encoders.TextTransformer(BERT), audio_bilstm()]), HEAD
        ),
        [
            ("model.", "encoder.encoders.0.model."),
            ("audio_lstm.", "encoder.encoders.1.lstm."),
        ],
    ),
    "pair_mmtransformer": (
        None,
        lambda: Classifier(
            encoders.Concat([encoders.TextTransformer(BERT), audio_bilstm()]),
            HEAD,
            True,
        ),
        [
            ("model.", "encoder.encoders.0.model."),
            ("audio_lstm.", "encoder.encoders.1.lstm."),
        ],
    ),
    "csa": (None, lambda: Classifier(csa(), HEAD), []),
    "pair_csa": (None, lambda: Classifier(csa(), HEAD, True), []),
    "ensemble": (None, ensemble, ENSEMBLE),
    "pair_ensemble": (None, lambda: ensemble(pair=True), ENSEMBLE),
    "multa": (None, lambda: Classifier(encoders.MulTA(DIM, 16, 1), HEAD), []),
    "pair_multa": (
        None,
        lambda: Classifier(encoders.MulTA(DIM, 16, 1), HEAD, True),
        [],
    ),
}


def rename(state: dict, rules: list) -> dict:
    renamed = {}
    for key, value in state.items():
        for old, new in rules + [("head.", "head."), ("", "encoder.")]:
            if key.startswith(old):
                renamed[new + key[len(old) :]] = value
                break
    return renamed


def convert(batch: dict, modality) -> dict:
    """Turns ``text_a_input_mask`` style keys into nested samples."""
    sides = {}
    for key, value in batch.items():
        parts = key.split("_")
        kind = parts.pop(0) if parts[0] in ("text", "audio") else modality
        side = parts.pop(0) if parts[0] in ("a", "b") else None
        sides.setdefault(side, {}).setdefault(kind, {})["_".join(parts)] = value
    return sides[None] if None in sides else sides


def build(name):
    """The new model with the weights of golden file ``name``, and the record."""
    th.manual_seed(0)
    modality, factory, rules = CASES[name]
    record = stored(f"models/{name}")
    model = factory().eval()
    model.load_state_dict(rename(record["state_dict"], rules))
    return model, convert(record["batch"], modality), record["logits"]


def samples(batch: dict) -> list:
    return [batch["a"], batch["b"]] if "a" in batch else [batch]


def test_cases_cover_every_golden_file():
    names = {path.stem for path in (GOLDEN / "models").glob("*.pt")}
    assert names == set(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_forward_matches_golden(name):
    model, batch, logits = build(name)
    if name == "mmtransformer":
        # The old model ignored the audio mask; the new encoder skips padding.
        del batch["audio"]["input_mask"]
    with th.no_grad():
        th.testing.assert_close(model(batch), logits, rtol=1e-4, atol=1e-5)


MASKS = {
    "text": th.tensor([[1.0] * 5, [1.0] * 4 + [0.0]]),
    "audio": th.tensor([[1.0] * 4, [1.0] * 3 + [0.0]]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_padding_does_not_change_logits(name):
    # Every encoder must ignore the values at padded positions. The LSTM
    # encoders read a mask only when the sample holds one, so one is added.
    model, batch, _ = build(name)
    for sample in samples(batch):
        for kind, fields in sample.items():
            fields.setdefault("input_mask", MASKS[kind])
    noisy = copy.deepcopy(batch)
    for sample in samples(noisy):
        for fields in sample.values():
            fields["inputs"][fields["input_mask"] == 0] = 7
    with th.no_grad():
        th.testing.assert_close(model(noisy), model(batch))


# The dropout that must drop all text features in each model.
TEXT_DROPOUT = {
    "csa": "encoder.text_dropout",
    "pair_csa": "encoder.text_dropout",
    "ensemble": "text.encoder.dropout",
    "pair_ensemble": "text.encoder.dropout",
}


@pytest.mark.parametrize("name", sorted(TEXT_DROPOUT))
def test_text_dropout_applies_in_training(name):
    # A training pass that drops all text equals an evaluation pass on zero
    # text features.
    model, batch, _ = build(name)
    for module_name, module in model.named_modules():
        if isinstance(module, th.nn.Dropout):
            module.p = 1.0 if module_name == TEXT_DROPOUT[name] else 0.0
        if isinstance(module, th.nn.MultiheadAttention):
            module.dropout = 0.0
    silent = copy.deepcopy(batch)
    for sample in samples(silent):
        sample["text"]["inputs"] = th.zeros_like(sample["text"]["inputs"])
    with th.no_grad():
        expected = model.eval()(silent)
        th.testing.assert_close(model.train()(batch), expected)
