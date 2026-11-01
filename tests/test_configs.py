"""Every registered configuration, reduced to data.

The snapshot in ``tests/golden/configs.json`` is the reference that the Cinnamon
registrations must reproduce. Factories are called and described: a module by
its ``repr``, a loss by its class and class weights, a class or function by its
import path.
"""

import inspect
import json

import pytest
import torch as th
from conftest import GOLDEN, RECORD

from mamkit.configs import audio, base, text, text_audio

SNAPSHOT = GOLDEN / "configs.json"


def describe(value):
    if isinstance(value, th.nn.modules.loss._Loss):
        weight = getattr(value, "weight", None)
        return {
            "loss": type(value).__name__,
            "weight": None if weight is None else weight.tolist(),
        }
    if isinstance(value, th.nn.Module):
        return repr(value)
    if inspect.isclass(value) or inspect.isfunction(value):
        if value.__name__ == "<lambda>":
            return describe(value())
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, dict):
        return {key: describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    return value


def configurations():
    for module in (text, audio, text_audio):
        for name, cls in vars(module).items():
            if inspect.isclass(cls) and issubclass(cls, base.BaseConfig):
                if cls.__module__ != module.__name__:
                    continue
                for key in cls.configs:
                    label = ":".join(
                        [
                            f"{module.__name__.split('.')[-1]}.{name}",
                            key.dataset,
                            key.input_mode.value,
                            key.task_name,
                            ",".join(sorted(key.tags)),
                        ]
                    )
                    yield label, cls, key


def build(cls, key):
    # Six keys fail to build; the snapshot records their error instead.
    try:
        return describe(vars(cls.from_config(key)))
    except (AttributeError, TypeError) as error:
        return {"error": f"{type(error).__name__}: {error}"}


def snapshot():
    return {label: build(cls, key) for label, cls, key in configurations()}


@pytest.fixture(scope="module")
def current():
    value = snapshot()
    if RECORD:
        SNAPSHOT.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    return value


def test_every_configuration_key_is_recorded(current):
    assert sorted(current) == sorted(json.loads(SNAPSHOT.read_text()))
    assert len(current) == 165


def test_every_configuration_matches_snapshot(current):
    recorded = json.loads(SNAPSHOT.read_text())
    mismatched = [label for label in current if current[label] != recorded[label]]
    assert not mismatched


def test_six_configuration_keys_fail_to_build(current):
    # Defect: three methods omit seeds and three keys name missing methods.
    failing = sorted(label for label, value in current.items() if "error" in value)
    assert failing == [
        "audio.BiLSTMMFCCsConfig:marg:audio-only:arc:mancini-et-al-2022",
        "audio.BiLSTMTransformerConfig:marg:audio-only:arc:mancini-et-al-2022",
        "text.TransformerConfig:mmused-fallacy:text-only:afc:bert,mancini-et-al-2024",
        "text.TransformerConfig:mmused-fallacy:text-only:afc:mancini-et-al-2024,roberta",
        "text.TransformerConfig:ukdebates:text-only:asd:mancini-et-al-2022",
        "text_audio.MMTransformerConfig:ukdebates:text-audio:asd:anonymous,bert,clap",
    ]
