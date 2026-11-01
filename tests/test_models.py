"""Model outputs for fixed weights and inputs.

Every model is checked in evaluation mode, where dropout is inactive. The
stored state dict lets a rewritten model load the same weights under its own
parameter names.
"""

import pytest
import torch as th
from conftest import BERT, GOLDEN, RECORD, stored

from mamkit.models import audio, text, text_audio
from mamkit.modules.transformer import CustomEncoder, PositionalEncoding

BATCH, TOKENS, FRAMES, DIM, VOCAB, CLASSES = 2, 5, 4, 8, 20, 3


def head(width):
    return lambda: th.nn.Sequential(
        th.nn.Linear(width, 6), th.nn.ReLU(), th.nn.Linear(6, CLASSES)
    )


def audio_encoder(dropout=0.1):
    layer = th.nn.TransformerEncoderLayer(
        d_model=DIM, nhead=2, dim_feedforward=16, dropout=dropout, batch_first=True
    )
    return th.nn.TransformerEncoder(layer, num_layers=1)


def positional():
    return PositionalEncoding(DIM, dual_modality=False)


def custom_encoder():
    return CustomEncoder(DIM, 16, 2, 1, 0.0)


KINDS = {
    "tokens": lambda: th.randint(1, VOCAB, (BATCH, TOKENS)),
    "ids": lambda: th.randint(1, 1000, (BATCH, TOKENS)),
    "text": lambda: th.randn(BATCH, TOKENS, DIM),
    "audio": lambda: th.randn(BATCH, FRAMES, DIM),
    # The second sample is padded by one position.
    "text_mask": lambda: th.tensor([[1.0] * TOKENS, [1.0] * (TOKENS - 1) + [0.0]]),
    "audio_mask": lambda: th.tensor([[1.0] * FRAMES, [1.0] * (FRAMES - 1) + [0.0]]),
}


def batch(**spec):
    """A batch builder: each keyword names a batch key and its kind."""
    return lambda: {key: KINDS[kind]() for key, kind in spec.items()}


TEXT_AUDIO = batch(
    text_inputs="text",
    text_input_mask="text_mask",
    audio_inputs="audio",
    audio_input_mask="audio_mask",
)
PAIR_TEXT_AUDIO = batch(
    text_a_inputs="text",
    text_a_input_mask="text_mask",
    audio_a_inputs="audio",
    audio_a_input_mask="audio_mask",
    text_b_inputs="text",
    text_b_input_mask="text_mask",
    audio_b_inputs="audio",
    audio_b_input_mask="audio_mask",
)

CASES = {
    "text_bilstm": (
        lambda: text.BiLSTM(VOCAB, DIM, [6], head(12)),
        batch(inputs="tokens"),
    ),
    "text_pair_bilstm": (
        lambda: text.PairBiLSTM(VOCAB, DIM, [6], head(24)),
        batch(a_inputs="tokens", b_inputs="tokens"),
    ),
    "text_transformer": (
        lambda: text.Transformer(BERT, head(32)),
        batch(inputs="ids", input_mask="text_mask"),
    ),
    "text_pair_transformer": (
        lambda: text.PairTransformer(BERT, head(64)),
        batch(
            a_inputs="ids",
            a_input_mask="text_mask",
            b_inputs="ids",
            b_input_mask="text_mask",
        ),
    ),
    "audio_bilstm": (
        lambda: audio.BiLSTM(DIM, [6], head(12), 0.0),
        batch(inputs="audio"),
    ),
    "audio_pair_bilstm": (
        lambda: audio.PairBiLSTM(DIM, [6], head(24), 0.0),
        batch(a_inputs="audio", b_inputs="audio"),
    ),
    "audio_transformer": (
        lambda: audio.TransformerEncoder(DIM, audio_encoder, head(DIM)),
        batch(inputs="audio", input_mask="audio_mask"),
    ),
    "audio_pair_transformer": (
        lambda: audio.PairTransformerEncoder(DIM, audio_encoder, head(2 * DIM)),
        batch(
            a_inputs="audio",
            a_input_mask="audio_mask",
            b_inputs="audio",
            b_input_mask="audio_mask",
        ),
    ),
    "text_audio_bilstm": (
        lambda: text_audio.BiLSTM(VOCAB, DIM, DIM, [6], [6], head(24)),
        batch(text_inputs="tokens", audio_inputs="audio"),
    ),
    "text_audio_pair_bilstm": (
        lambda: text_audio.PairBiLSTM(VOCAB, DIM, DIM, [6], [6], head(48)),
        batch(
            text_a_inputs="tokens",
            audio_a_inputs="audio",
            text_b_inputs="tokens",
            audio_b_inputs="audio",
        ),
    ),
    "mmtransformer": (
        lambda: text_audio.MMTransformer(BERT, head(44), DIM, [6]),
        batch(
            text_inputs="ids",
            text_input_mask="text_mask",
            audio_inputs="audio",
            audio_input_mask="audio_mask",
        ),
    ),
    "pair_mmtransformer": (
        lambda: text_audio.PairMMTransformer(BERT, head(88), DIM, [6]),
        batch(
            text_a_inputs="ids",
            text_a_input_mask="text_mask",
            audio_a_inputs="audio",
            text_b_inputs="ids",
            text_b_input_mask="text_mask",
            audio_b_inputs="audio",
        ),
    ),
    "csa": (
        lambda: text_audio.CSA(custom_encoder, head(DIM), positional),
        TEXT_AUDIO,
    ),
    "pair_csa": (
        lambda: text_audio.PairCSA(custom_encoder, head(2 * DIM), positional),
        PAIR_TEXT_AUDIO,
    ),
    "ensemble": (
        lambda: text_audio.Ensemble(
            head(DIM), head(DIM), audio_encoder, positional, DIM
        ),
        TEXT_AUDIO,
    ),
    "pair_ensemble": (
        lambda: text_audio.PairEnsemble(
            head(2 * DIM), head(2 * DIM), audio_encoder, positional, DIM
        ),
        PAIR_TEXT_AUDIO,
    ),
    "multa": (
        lambda: text_audio.MulTA(DIM, 16, 1, head(2 * DIM), positional),
        TEXT_AUDIO,
    ),
    "pair_multa": (
        lambda: text_audio.PairMulTA(DIM, 16, 1, head(4 * DIM), positional),
        PAIR_TEXT_AUDIO,
    ),
}


def build(name):
    """The model ``name`` with its golden weights, and its golden record."""
    th.manual_seed(0)
    factory, make_batch = CASES[name]
    model = factory().eval()
    if RECORD:
        inputs = make_batch()
        with th.no_grad():
            logits = model(inputs)
        path = GOLDEN / "models" / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"state_dict": model.state_dict(), "batch": inputs, "logits": logits}
        th.save(record, path)
    record = stored(f"models/{name}")
    model.load_state_dict(record["state_dict"])
    return model, record


@pytest.mark.parametrize("name", sorted(CASES))
def test_forward_matches_golden(name):
    model, record = build(name)
    with th.no_grad():
        logits = model(record["batch"])
    assert logits.shape == (BATCH, CLASSES)
    th.testing.assert_close(logits, record["logits"], rtol=1e-4, atol=1e-5)


def no_dropout_positional():
    return PositionalEncoding(DIM, dropout=0.0)


# Each model drops all text features and nothing else.
TEXT_DROPOUT = {
    "csa": lambda: text_audio.CSA(
        custom_encoder,
        head(DIM),
        no_dropout_positional,
        text_dropout_rate=1.0,
        audio_dropout_rate=0.0,
    ),
    "ensemble": lambda: text_audio.Ensemble(
        head(DIM),
        head(DIM),
        lambda: audio_encoder(dropout=0.0),
        no_dropout_positional,
        DIM,
        text_dropout_rate=1.0,
        audio_dropout_rate=0.0,
    ),
    "pair_csa": lambda: text_audio.PairCSA(
        custom_encoder,
        head(2 * DIM),
        no_dropout_positional,
        text_dropout_rate=1.0,
        audio_dropout_rate=0.0,
    ),
    "pair_ensemble": lambda: text_audio.PairEnsemble(
        head(2 * DIM),
        head(2 * DIM),
        lambda: audio_encoder(dropout=0.0),
        no_dropout_positional,
        DIM,
        text_dropout_rate=1.0,
        audio_dropout_rate=0.0,
    ),
}


@pytest.mark.filterwarnings("error::UserWarning")
@pytest.mark.parametrize("name", sorted(TEXT_DROPOUT))
def test_text_dropout_applies_in_training(name):
    # A training pass that drops all text equals an evaluation pass on zero
    # text features.
    _, record = build(name)
    model = TEXT_DROPOUT[name]()
    model.load_state_dict(record["state_dict"])
    silent = dict(record["batch"])
    for key in silent:
        if key.startswith("text_") and key.endswith("inputs"):
            silent[key] = th.zeros_like(silent[key])
    with th.no_grad():
        expected = model.eval()(silent)
        th.testing.assert_close(model.train()(record["batch"]), expected)


MASKED = [
    "audio_pair_transformer",
    "audio_transformer",
    "csa",
    "ensemble",
    "multa",
    "pair_csa",
    "pair_ensemble",
    "pair_multa",
    "text_pair_transformer",
    "text_transformer",
]


@pytest.mark.parametrize("name", MASKED)
def test_padding_does_not_change_logits(name):
    # Models that read a mask must ignore the values at padded positions.
    model, record = build(name)
    noisy = {key: value.clone() for key, value in record["batch"].items()}
    for key, mask in record["batch"].items():
        if key.endswith("input_mask"):
            noisy[key.replace("input_mask", "inputs")][mask == 0] = 100.0
    with th.no_grad():
        th.testing.assert_close(model(noisy), model(record["batch"]))
