"""Encoders that turn one sample into one vector.

A sample maps a modality to its fields::

    {"text": {"inputs": ..., "input_mask": ...},
     "audio": {"inputs": ..., "input_mask": ...}}

A mask holds 1 for valid positions and 0 for padding. Every encoder exposes
``output_dim``, the width of the vector it returns. Encoders know nothing about
pairs: ``mamkit.components.models.Classifier`` applies one encoder to each side.

An argument that holds another component also accepts its Cinnamon
registration key, which ``Registry.from_keys`` builds.
"""

from typing import List, Optional, Union

import torch as th
from cinnamon.registry import RegistrationKey, Registry
from transformers import AutoModel

from mamkit.modules.rnn import LSTMStack
from mamkit.modules.transformer import (
    CustomEncoder,
    MulTACrossAttentionBlock,
    PositionalEncoding,
)

#: A child component, or the registration key that builds it.
Child = Union[RegistrationKey, th.nn.Module]


def masked_mean(values: th.Tensor, mask: th.Tensor) -> th.Tensor:
    """Averages ``values`` of shape [bs, T, d] over the valid positions of ``mask``."""
    return (values * mask[:, :, None]).sum(dim=1) / mask.sum(dim=1)[:, None]


def lengths(fields: dict) -> Optional[th.Tensor]:
    """The sequence lengths given by the mask in ``fields``, if it has one."""
    mask = fields.get("input_mask")
    return None if mask is None else mask.sum(dim=1).long()


class TextBiLSTM(th.nn.Module):
    """Embeds token indices and returns the last hidden states of a BiLSTM stack.

    The embedding exists once ``load_vocabulary`` receives the vocabulary that a
    featurizer fitted. When the sample holds a text mask, the stack skips the
    padded tokens.
    """

    def __init__(
        self, embedding_dim: int, lstm_weights: List[int], dropout_rate: float = 0.0
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = None
        self.lstm = LSTMStack(input_size=embedding_dim, lstm_weights=lstm_weights)
        self.dropout = th.nn.Dropout(p=dropout_rate)
        self.output_dim = 2 * lstm_weights[-1]

    def load_vocabulary(
        self, vocab_size: int, embedding_matrix: Optional[th.Tensor] = None
    ):
        """Creates the embedding, initialized from ``embedding_matrix`` if given."""
        self.embedding = th.nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data = embedding_matrix

    def forward(self, sample: dict) -> th.Tensor:
        text = sample["text"]
        tokens = self.dropout(self.embedding(text["inputs"]))
        return self.lstm(tokens, lengths(text))


class TextTransformer(th.nn.Module):
    """Averages the last hidden states of a pretrained model over valid tokens."""

    def __init__(
        self,
        model_card: str,
        dropout_rate: float = 0.0,
        is_transformer_trainable: bool = False,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_card)
        self.model.requires_grad_(is_transformer_trainable)
        self.dropout = th.nn.Dropout(p=dropout_rate)
        self.output_dim = self.model.config.hidden_size

    def forward(self, sample: dict) -> th.Tensor:
        text = sample["text"]
        tokens = self.model(
            input_ids=text["inputs"], attention_mask=text["input_mask"]
        ).last_hidden_state
        return masked_mean(self.dropout(tokens), text["input_mask"])


class TextPool(th.nn.Module):
    """Averages precomputed token embeddings over valid tokens."""

    def __init__(self, embedding_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.dropout = th.nn.Dropout(p=dropout_rate)
        self.output_dim = embedding_dim

    def forward(self, sample: dict) -> th.Tensor:
        text = sample["text"]
        return masked_mean(self.dropout(text["inputs"]), text["input_mask"])


class AudioBiLSTM(th.nn.Module):
    """Returns the last hidden states of a BiLSTM stack over audio frames.

    When the sample holds an audio mask, the stack skips the padded frames.
    """

    def __init__(
        self, embedding_dim: int, lstm_weights: List[int], dropout_rate: float = 0.0
    ):
        super().__init__()
        self.lstm = LSTMStack(input_size=embedding_dim, lstm_weights=lstm_weights)
        self.dropout = th.nn.Dropout(p=dropout_rate)
        self.output_dim = 2 * lstm_weights[-1]

    def forward(self, sample: dict) -> th.Tensor:
        audio = sample["audio"]
        return self.lstm(self.dropout(audio["inputs"]), lengths(audio))


class AudioTransformer(th.nn.Module):
    """A Transformer encoder over audio frames, averaged over valid frames.

    ``encoder_dropout_rate`` applies inside the encoder layers,
    ``positional_dropout_rate`` after the positional encodings, and
    ``dropout_rate`` to the encoder output before the residual connection.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        ffn_dim: int,
        n_layers: int,
        dropout_rate: float = 0.0,
        encoder_dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
    ):
        super().__init__()
        layer = th.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=encoder_dropout_rate,
            batch_first=True,
        )
        self.encoder = th.nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.pos_encoder = PositionalEncoding(
            embedding_dim, dropout=positional_dropout_rate
        )
        self.layer_norm = th.nn.LayerNorm(embedding_dim)
        self.dropout = th.nn.Dropout(p=dropout_rate)
        self.output_dim = embedding_dim

    def forward(self, sample: dict) -> th.Tensor:
        audio = sample["audio"]
        mask = audio["input_mask"]
        features = self.pos_encoder(audio["inputs"])
        output = self.encoder(features, src_key_padding_mask=~mask.to(th.bool))
        output = self.layer_norm(features + self.dropout(output))
        return masked_mean(output, mask)


class CSA(th.nn.Module):
    """Self-attention over the concatenation of text embeddings and audio frames.

    ``positional_dropout_rate`` applies after the positional encodings of the
    audio frames.
    """

    def __init__(
        self,
        embedding_dim: int,
        ffn_dim: int,
        n_heads: int,
        n_layers: int,
        encoder_dropout_rate: float = 0.1,
        text_dropout_rate: float = 0.1,
        audio_dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.transformer = CustomEncoder(
            embedding_dim, ffn_dim, n_heads, n_layers, encoder_dropout_rate
        )
        self.positional_encoder = PositionalEncoding(
            embedding_dim, dropout=positional_dropout_rate
        )
        self.text_dropout = th.nn.Dropout(p=text_dropout_rate)
        # Applied to the encoder output, as in the original CSA.
        self.audio_dropout = th.nn.Dropout(p=audio_dropout_rate)
        self.output_dim = embedding_dim

    def forward(self, sample: dict) -> th.Tensor:
        text, audio = sample["text"], sample["audio"]
        tokens = self.text_dropout(text["inputs"])
        frames = self.positional_encoder(audio["inputs"])
        output = self.transformer(
            th.cat((tokens, frames), dim=1), text["input_mask"], audio["input_mask"]
        )
        mask = th.cat((text["input_mask"], audio["input_mask"].float()), dim=1)
        return masked_mean(self.audio_dropout(output), mask)


class MulTA(th.nn.Module):
    """Cross-modal attention from text to audio and from audio to text.

    Each of the ``n_blocks`` blocks per direction attends with ``n_heads``
    heads. ``text_dropout_rate`` and ``audio_dropout_rate`` apply to the
    attention weights of each direction, ``ffn_dropout_rate`` inside the
    feed-forward layer of every block, and ``positional_dropout_rate`` after
    the positional encodings.
    """

    def __init__(
        self,
        embedding_dim: int,
        ffn_dim: int,
        n_blocks: int,
        n_heads: int = 4,
        text_dropout_rate: float = 0.1,
        audio_dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        ffn_dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.text_crossmodal_blocks = th.nn.ModuleList(
            MulTACrossAttentionBlock(
                embedding_dim,
                ffn_dim,
                n_heads,
                dropout_prob=text_dropout_rate,
                ffn_dropout_prob=ffn_dropout_rate,
            )
            for _ in range(n_blocks)
        )
        self.audio_crossmodal_blocks = th.nn.ModuleList(
            MulTACrossAttentionBlock(
                embedding_dim,
                ffn_dim,
                n_heads,
                dropout_prob=audio_dropout_rate,
                ffn_dropout_prob=ffn_dropout_rate,
            )
            for _ in range(n_blocks)
        )
        self.positional_encoder = PositionalEncoding(
            embedding_dim, dropout=positional_dropout_rate
        )
        self.output_dim = 2 * embedding_dim

    def forward(self, sample: dict) -> th.Tensor:
        text_mask = sample["text"]["input_mask"]
        audio_mask = sample["audio"]["input_mask"]
        text = self.positional_encoder(sample["text"]["inputs"])
        audio = self.positional_encoder(sample["audio"]["inputs"])

        # Text queries attend to audio, and audio queries attend to text.
        text_out = text
        for block in self.text_crossmodal_blocks:
            text_out = block(text_out, audio, audio_mask)
        audio_out = audio
        for block in self.audio_crossmodal_blocks:
            audio_out = block(audio_out, text, text_mask)

        return th.cat(
            (masked_mean(text_out, text_mask), masked_mean(audio_out, audio_mask)),
            dim=-1,
        )


class Concat(th.nn.Module):
    """Concatenates the outputs of several encoders on the same sample."""

    def __init__(self, encoders: List[Child]):
        super().__init__()
        self.encoders = th.nn.ModuleList(Registry.from_keys(encoders))
        self.output_dim = sum(encoder.output_dim for encoder in self.encoders)

    def forward(self, sample: dict) -> th.Tensor:
        return th.cat([encoder(sample) for encoder in self.encoders], dim=-1)
