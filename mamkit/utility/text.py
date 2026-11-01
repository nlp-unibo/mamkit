"""Tokenization, vocabularies, and GloVe vectors for the BiLSTM text models."""

import logging
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, List, Union

import torch as th

# The rules of the basic_english tokenizer in torchtext 0.18, in their order.
_PATTERNS = [
    (re.compile(pattern), replacement)
    for pattern, replacement in [
        (r"\'", " '  "),
        (r"\"", ""),
        (r"\.", " . "),
        (r"<br \/>", " "),
        (r",", " , "),
        (r"\(", " ( "),
        (r"\)", " ) "),
        (r"\!", " ! "),
        (r"\?", " ? "),
        (r"\;", " "),
        (r"\:", " "),
        (r"\s+", " "),
    ]
]

SPECIALS = ["<pad>", "<unk>"]

GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
GLOVE_NAME = re.compile(r"glove\.6B\.(?:50|100|200|300)d")


def basic_english(line: str) -> List[str]:
    """Lowercases a line, separates punctuation, and splits it on whitespace."""
    line = line.lower()
    for pattern, replacement in _PATTERNS:
        line = pattern.sub(replacement, line)
    return line.split()


class Vocabulary:
    """A token index in which every unknown token maps to ``<unk>``."""

    def __init__(self, tokens: Iterable[str]):
        self.itos = list(tokens)
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.unknown = self.stoi["<unk>"]

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.unknown)

    def __call__(self, tokens: Iterable[str]) -> List[int]:
        return [self[token] for token in tokens]

    def __len__(self) -> int:
        return len(self.itos)

    def get_itos(self) -> List[str]:
        return self.itos


def build_vocabulary(
    texts: Iterable[str],
    tokenizer: Callable[[str], List[str]],
    min_freq: int = 1,
) -> Vocabulary:
    """Builds a vocabulary with the special tokens first.

    The other tokens appear at least ``min_freq`` times. They are ordered by
    descending frequency, then lexicographically, as in torchtext 0.18.
    """
    counts = Counter(token for text in texts for token in tokenizer(text))
    tokens = sorted(
        (token for token, count in counts.items() if count >= min_freq),
        key=lambda token: (-counts[token], token),
    )
    return Vocabulary(SPECIALS + [token for token in tokens if token not in SPECIALS])


def glove_vectors(
    name: str,
    tokens: List[str],
    cache: Union[str, Path] = ".vector_cache",
) -> th.Tensor:
    """Returns one GloVe vector per token, and zeros for tokens without one.

    ``name`` has the form ``glove.6B.<dim>d``. The archive is downloaded into
    ``cache`` once, and only the requested vector file is extracted.
    """
    match = GLOVE_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported embedding model {name!r}: use glove.6B.<dim>d")
    cache = Path(cache)
    path = cache / f"{name}.txt"
    if not path.exists():
        archive = cache / "glove.6B.zip"
        if not archive.exists():
            logging.info(f"Downloading {GLOVE_URL}...")
            cache.mkdir(parents=True, exist_ok=True)
            partial = archive.with_suffix(".part")
            urllib.request.urlretrieve(GLOVE_URL, partial)
            partial.rename(archive)
        with zipfile.ZipFile(archive) as files:
            files.extract(path.name, cache)

    wanted = {token: index for index, token in enumerate(tokens)}
    vectors = None
    with path.open("rb") as lines:
        for line in lines:
            word, _, values = line.rstrip().partition(b" ")
            values = values.split(b" ")
            if vectors is None:
                vectors = th.zeros(len(tokens), len(values))
            index = wanted.get(word.decode("utf-8", errors="replace"))
            if index is not None:
                vectors[index] = th.tensor([float(value) for value in values])
    return vectors
