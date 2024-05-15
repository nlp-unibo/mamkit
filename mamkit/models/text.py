import torch as th

from mamkit.modules.rnn import LSTMStack
from transformers import AutoModel, AutoConfig


class TextOnlyModel(th.nn.Module):

    def forward(
            self,
            text
    ):
        pass


class BiLSTM(TextOnlyModel):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            lstm_weights,
            head: th.nn.Module,
            dropout_rate=0.0,
            embedding_matrix=None
    ):
        super().__init__()

        self.embedding = th.nn.Embedding(num_embeddings=vocab_size,
                                         embedding_dim=embedding_dim,
                                         padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data = embedding_matrix

        self.lstm = LSTMStack(input_size=embedding_dim,
                              lstm_weigths=lstm_weights)
        self.head = head

        self.dropout = th.nn.Dropout(p=dropout_rate)

    def forward(
            self,
            text
    ):
        # [bs, N, d]
        tokens_emb = self.embedding(text)

        tokens_emb = self.dropout(tokens_emb)

        # [bs, d']
        text_emb = self.lstm(tokens_emb)

        logits = self.head(text_emb)

        # [bs, #classes]
        return logits


class Transformer(TextOnlyModel):

    def __init__(
            self,
            model_card,
            head: th.nn.Module,
            dropout_rate=0.0,
            is_transformer_trainable: bool = False,
    ):
        super().__init__()

        self.model_card = model_card
        self.model_config = AutoConfig.from_pretrained(model_card)
        self.model = AutoModel.from_pretrained(model_card)

        if not is_transformer_trainable:
            for module in self.model.modules():
                for param in module.parameters():
                    param.requires_grad = False

        self.head = head
        self.dropout = th.nn.Dropout(p=dropout_rate)

    def forward(
            self,
            text
    ):
        input_ids, attention_mask = text

        tokens_emb = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        tokens_emb = self.dropout(tokens_emb)
        text_emb = (tokens_emb * attention_mask[:, :, None]).sum(dim=1)
        text_emb = text_emb / attention_mask.sum(dim=1)[:, None]

        logits = self.head(text_emb)
        return logits


class TransformerHead(TextOnlyModel):
    """
    Class for the text-only model
    """

    def __init__(
            self,
            head: th.nn.Module,
            dropout_rate=0.0
    ):
        super().__init__()
        self.head = head
        self.dropout = th.nn.Dropout(dropout_rate)

    def forward(
            self,
            text
    ):
        # tokens_emb     -> [bs, N, d]
        # attention_mask -> [bs, N]
        tokens_emb, attention_mask = text

        tokens_emb = self.dropout(tokens_emb)

        # [bs, d]
        text_emb = (tokens_emb * attention_mask[:, :, None]).sum(dim=1)
        text_emb = text_emb / attention_mask.sum(dim=1)[:, None]

        logits = self.head(text_emb)
        return logits
