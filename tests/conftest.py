"""Shared fixtures for the characterization suite.

Golden files under ``tests/golden`` hold outputs recorded from the code as it
stood before the Cinnamon refactoring. That code has been removed, so the files
are fixed references. A rewritten module must reproduce them from the same
weights and inputs.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch as th

GOLDEN = Path(__file__).parent / "golden"

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
