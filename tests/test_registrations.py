"""The registered experiments reproduce the legacy configurations.

``tests/golden/registrations.json`` maps each legacy configuration key to the
tags of its task registration, and to the audio card of its variant when the
card is not the default one. Each task configuration and its children are read
back from the registry, described with the field names of the legacy
configuration, and compared with ``tests/golden/configs.json``.
"""

import json
import re
from pathlib import Path

import pytest
import torch as th
from cinnamon.registry import Registry
from conftest import GOLDEN

import mamkit
from mamkit.components.lightning import import_path
from mamkit.modules.transformer import CustomEncoder, PositionalEncoding

LEGACY = json.loads((GOLDEN / "configs.json").read_text())
MAPPING = json.loads((GOLDEN / "registrations.json").read_text())

#: Hidden size of the BERT and RoBERTa base cards, which the legacy heads read.
HIDDEN = {"bert-base-uncased": 768, "roberta-base": 768}

#: Buildable legacy keys that are not registered, with the reason.
SKIPPED = {
    # The head reads 256 features, but an M-Arg model reads a pair of sides
    # and gives the head 512, so the configuration cannot run.
    "text.BiLSTMConfig:marg:text-only:arc:mancini-et-al-2022",
}


@pytest.fixture(scope="module")
def registry():
    valid, invalid = Registry.build(directory=Path(mamkit.__file__).parent)
    assert not invalid
    return valid


def task_key(registry, label):
    entry = MAPPING[label]
    base = set(entry["task"][1:])
    card = entry["audio_model_card"]
    matches = [
        key
        for key in registry
        if key.name == "task"
        and (
            base < key.tags and f"audio.model_card={card}" in key.tags
            if card
            else key.tags == base
        )
    ]
    assert len(matches) == 1, (label, matches)
    return matches[0]


def config(key):
    return Registry.retrieve_configuration(key)


def output_dim(encoder) -> int:
    """The width of the vector that an encoder configuration returns."""
    fields = encoder.model_dump()
    if "encoders" in fields:
        return sum(output_dim(config(key)) for key in encoder.encoders)
    if "lstm_weights" in fields:
        return 2 * encoder.lstm_weights[-1]
    if "n_blocks" in fields:
        return 2 * encoder.embedding_dim
    if "model_card" in fields:
        return HIDDEN[encoder.model_card]
    return encoder.embedding_dim


def head(classifier) -> str:
    width = output_dim(config(classifier.encoder)) * (2 if classifier.pair else 1)
    layers = []
    for size in classifier.head:
        layers += [th.nn.Linear(width, size), th.nn.ReLU()]
        width = size
    return repr(th.nn.Sequential(*layers[:-1]))


def audio_encoder(encoder) -> str:
    layer = th.nn.TransformerEncoderLayer(
        d_model=encoder.embedding_dim,
        nhead=encoder.n_heads,
        dim_feedforward=encoder.ffn_dim,
        batch_first=True,
    )
    module = th.nn.TransformerEncoder(layer, num_layers=encoder.n_layers)
    return repr(module).replace(
        "elementwise_affine=True, bias=True)", "elementwise_affine=True)"
    )


def positional(dim) -> str:
    return repr(PositionalEncoding(dim))


def describe(label, key) -> dict:
    """The task of ``key`` described with the fields of the legacy ``label``."""
    task = config(key)
    model = config(task.model)
    text = config(task.text) if task.text else None
    audio = config(task.audio) if task.audio else None
    view = {
        "seeds": task.seeds,
        "batch_size": task.batch_size,
        "optimizer": task.optimizer,
        "optimizer_args": task.optimizer_args,
        "loss_function": {
            "loss": task.loss.rsplit(".", 1)[-1],
            "weight": task.class_weights,
        },
    }
    if text is not None and "embedding_model" in text.model_dump():
        view.update(
            embedding_model=text.embedding_model,
            tokenizer="mamkit.utility.text.basic_english",
            tokenization_args=None,
        )
    if audio is not None and "mfccs" in audio.model_dump():
        view.update(
            {
                name: getattr(audio, name)
                for name in (
                    "mfccs",
                    "sampling_rate",
                    "pooling_sizes",
                    "remove_energy",
                    "normalize",
                )
            }
        )
    if audio is not None and "model_card" in audio.model_dump():
        prefix = "" if label.startswith("audio.") else "audio_"
        view.update(
            {
                f"{prefix}model_card": audio.model_card,
                "sampling_rate": audio.sampling_rate,
                "downsampling_factor": audio.downsampling_factor,
                "aggregate": audio.aggregate,
                "processor_args": audio.processor_args,
                f"{prefix}model_args": audio.model_args,
            }
        )

    family = label.split(":")[0]
    if family == "text_audio.EnsembleConfig":
        text_model, audio_model = config(model.text), config(model.audio)
        pool, transformer = config(text_model.encoder), config(audio_model.encoder)
        view.update(
            text_head=head(text_model),
            audio_head=head(audio_model),
            num_classes=text_model.head[-1],
            audio_encoder=audio_encoder(transformer),
            audio_embedding_dim=transformer.embedding_dim,
            positional_encoder=positional(transformer.embedding_dim),
            text_dropout_rate=pool.dropout_rate,
            audio_dropout_rate=transformer.dropout_rate,
            lower_bound=model.lower_bound,
            upper_bound=model.upper_bound,
        )
    else:
        view.update(head=head(model), num_classes=model.head[-1])
        encoder = config(model.encoder)
        parts = (
            [config(k) for k in encoder.encoders]
            if family.startswith("text_audio.BiLSTM")
            or family == "text_audio.MMTransformerConfig"
            else [encoder]
        )
        describe_encoders(family, view, parts)
    if text is not None and "tokenizer_args" in text.model_dump():
        view["tokenizer_args"] = text.tokenizer_args
        if family == "text.TransformerConfig":
            view["model_card"] = text.model_card
        else:
            view["text_model_card"] = text.model_card
        if "model_args" in text.model_dump():
            view["text_model_args"] = text.model_args
    return view


def describe_encoders(family, view, parts):
    first = parts[0]
    if family in ("text.BiLSTMConfig", "audio.BiLSTMMFCCsConfig"):
        view.update(first.model_dump(include={"embedding_dim", "lstm_weights"}))
        view["dropout_rate"] = first.dropout_rate
    elif family == "audio.BiLSTMTransformerConfig":
        view.update(first.model_dump(include={"embedding_dim", "lstm_weights"}))
        view["dropout_rate"] = first.dropout_rate
    elif family == "text.TransformerConfig":
        view["dropout_rate"] = first.dropout_rate
        view["is_transformer_trainable"] = first.is_transformer_trainable
    elif family == "audio.TransformerEncoderConfig":
        view.update(
            embedding_dim=first.embedding_dim,
            dropout_rate=first.dropout_rate,
            encoder=audio_encoder(first),
        )
    elif family in (
        "text_audio.BiLSTMMFCCsConfig",
        "text_audio.BiLSTMTransformerConfig",
    ):
        text, audio = parts
        view.update(
            text_embedding_dim=text.embedding_dim,
            text_lstm_weights=text.lstm_weights,
            text_dropout_rate=text.dropout_rate,
            audio_embedding_dim=audio.embedding_dim,
            audio_lstm_weights=audio.lstm_weights,
            audio_dropout_rate=audio.dropout_rate,
        )
    elif family == "text_audio.MMTransformerConfig":
        text, audio = parts
        view.update(
            text_embedding_dim=HIDDEN[text.model_card],
            text_dropout_rate=text.dropout_rate,
            is_transformer_trainable=text.is_transformer_trainable,
            audio_embedding_dim=audio.embedding_dim,
            lstm_weights=audio.lstm_weights,
            audio_dropout_rate=audio.dropout_rate,
        )
    elif family == "text_audio.CSAConfig":
        module = CustomEncoder(
            first.embedding_dim,
            first.ffn_dim,
            first.n_heads,
            first.n_layers,
            first.encoder_dropout_rate,
        )
        view.update(
            transformer=repr(module),
            positional_encoder=positional(first.embedding_dim),
            text_dropout_rate=first.text_dropout_rate,
            audio_dropout_rate=first.audio_dropout_rate,
        )
    elif family == "text_audio.MulTAConfig":
        view.update(
            text_embedding_dim=first.embedding_dim,
            audio_embedding_dim=first.embedding_dim,
            d_ffn=first.ffn_dim,
            n_blocks=first.n_blocks,
            positional_encoder=positional(first.embedding_dim),
            text_dropout_rate=first.text_dropout_rate,
            audio_dropout_rate=first.audio_dropout_rate,
        )
    else:
        raise ValueError(family)


def test_every_buildable_legacy_key_is_registered():
    buildable = {label for label, value in LEGACY.items() if "error" not in value}
    assert set(MAPPING) == buildable - SKIPPED


@pytest.mark.parametrize("label", sorted(MAPPING))
def test_registration_matches_legacy_configuration(registry, label):
    view = describe(label, task_key(registry, label))
    legacy = dict(LEGACY[label])
    assert import_path(view.pop("optimizer")) is import_path(legacy.pop("optimizer"))
    # An empty argument dictionary and None both pass no arguments.
    view, legacy = (
        {k: None if k.endswith("_args") and not v else v for k, v in d.items()}
        for d in (view, legacy)
    )
    assert view == {name: legacy.get(name) for name in view}, label
    assert set(legacy) == set(view), label


def test_registered_tasks_run_their_seeds_and_epochs(registry):
    ukdebates = [
        config(key)
        for key in registry
        if key.name == "task" and {"ukdebates", "text-only", "bilstm"} <= key.tags
    ]
    assert {task.trainer_args["max_epochs"] for task in ukdebates} == {100}
    assert all(
        re.fullmatch(r"results/ukdebates/asd/text-only/bilstm/.+", task.save_path)
        for task in ukdebates
    )


def needs_download(key) -> bool:
    """Whether building ``key`` loads a pretrained model or a corpus."""
    component = Registry.retrieve_configuration_info(key).component
    if component.endswith((".TextTransformer", ".TokenizedText", "Loader")):
        return True
    if ".loaders." in component:
        return True
    children = []
    for value in Registry.retrieve_configuration(key).dependencies.values():
        children += value if isinstance(value, list) else [value]
    return any(needs_download(child) for child in children)


def test_registered_models_and_metrics_build(registry):
    built = 0
    for key in registry:
        if key.name in ("encoder", "model", "metric") and not needs_download(key):
            Registry.from_key(key)
            built += 1
    assert built > 100
