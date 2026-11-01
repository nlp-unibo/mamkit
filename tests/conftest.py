"""Shared fixtures for the characterization suite.

Golden files under ``tests/golden`` hold outputs recorded from the code as it
stood before the Cinnamon refactoring. A rewritten module must reproduce them
from the same weights and inputs. Re-record a file only when a change of
behavior is intended, by running ``MAMKIT_RECORD=1 pytest <test>``, and say so
in the pull request.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch as th

GOLDEN = Path(__file__).parent / "golden"
RECORD = os.environ.get("MAMKIT_RECORD") == "1"

#: Small pretrained models with random weights, published for testing.
BERT = "hf-internal-testing/tiny-random-BertModel"
ROBERTA = "hf-internal-testing/tiny-random-RobertaModel"
WAV2VEC = "hf-internal-testing/tiny-random-Wav2Vec2Model"
HUBERT = "hf-internal-testing/tiny-random-HubertModel"


def tensors(value):
    """Turns arrays nested in lists, tuples, and dicts into tensors."""
    if isinstance(value, np.ndarray):
        return th.as_tensor(value)
    if isinstance(value, dict):
        return {key: tensors(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensors(item) for item in value]
    return value


def golden(name: str, value):
    """Compares ``value`` with the golden file ``name``, or records it."""
    path = GOLDEN / f"{name}.pt"
    if RECORD:
        path.parent.mkdir(parents=True, exist_ok=True)
        th.save(tensors(value), path)
    th.testing.assert_close(tensors(value), th.load(path), rtol=1e-4, atol=1e-5)


def stored(name: str):
    return th.load(GOLDEN / f"{name}.pt")


@pytest.fixture(scope="session")
def audio_files(tmp_path_factory) -> dict:
    """Two short noise clips at 16 kHz and one at 22.05 kHz."""
    directory = tmp_path_factory.mktemp("audio")
    generator = np.random.default_rng(0)
    files = {}
    for name, rate, seconds in [
        ("short", 16_000, 0.5),
        ("long", 16_000, 0.8),
        ("resampled", 22_050, 0.5),
    ]:
        path = directory / f"{name}.wav"
        signal = generator.uniform(-0.5, 0.5, int(rate * seconds)).astype(np.float32)
        soundfile.write(path, signal, rate)
        files[name] = path
    return files
