"""Models built from the encoders in ``mamkit.components.encoders``.

A pair batch maps ``"a"`` and ``"b"`` to two samples. Only ``Classifier``
knows whether a batch is a pair.
"""

from typing import List

import torch as th


class Classifier(th.nn.Module):
    """An encoder followed by a head of linear layers separated by ReLU.

    ``head`` lists the output width of each linear layer; the last width is the
    number of classes. With ``pair``, the encoder encodes side ``a`` and side
    ``b``, and the head reads both vectors concatenated.
    """

    def __init__(self, encoder: th.nn.Module, head: List[int], pair: bool = False):
        super().__init__()
        self.encoder = encoder
        self.pair = pair
        width = encoder.output_dim * (2 if pair else 1)
        layers = []
        for size in head:
            layers += [th.nn.Linear(width, size), th.nn.ReLU()]
            width = size
        self.head = th.nn.Sequential(*layers[:-1])

    def forward(self, batch: dict) -> th.Tensor:
        if self.pair:
            features = th.cat((self.encoder(batch["a"]), self.encoder(batch["b"])), -1)
        else:
            features = self.encoder(batch)
        return self.head(features)


class Ensemble(th.nn.Module):
    """Late fusion of a text classifier and an audio classifier.

    A learned weight mixes the two class distributions with a text coefficient
    between ``lower_bound`` and ``upper_bound``. The model returns log
    probabilities.
    """

    def __init__(
        self,
        text: th.nn.Module,
        audio: th.nn.Module,
        lower_bound: float = 0.3,
        upper_bound: float = 0.7,
    ):
        super().__init__()
        self.text = text
        self.audio = audio
        # tanh(0) gives both classifiers the same weight at the start.
        self.weight = th.nn.Parameter(th.tensor(0.0))
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, batch: dict) -> th.Tensor:
        coefficient = (th.tanh(self.weight) + 1) / 2
        coefficient = (
            coefficient * (self.upper_bound - self.lower_bound) + self.lower_bound
        )
        probabilities = coefficient * self.text(batch).softmax(dim=-1) + (
            1 - coefficient
        ) * self.audio(batch).softmax(dim=-1)
        return th.log(probabilities + 1e-08)
