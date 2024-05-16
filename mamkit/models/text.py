import torch as th
from transformers import AutoModel, AutoConfig

from mamkit.modules.rnn import LSTMStack


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
            head,
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
        self.head = head()

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


class PairBiLSTM(BiLSTM):

    def forward(
            self,
            text
    ):
        a_text, b_text = text

        # A input
        # [bs, T, d_t]
        a_tokens_emb = self.embedding(a_text)
        a_tokens_emb = self.dropout(a_tokens_emb)

        # [bs, d']
        a_text_emb = self.lstm(a_tokens_emb)

        # B input
        # [bs, T, d_t]
        b_tokens_emb = self.embedding(b_text)
        b_tokens_emb = self.dropout(b_tokens_emb)

        # [bs, d']
        b_text_emb = self.lstm(b_tokens_emb)

        concat_emb = th.concat((a_text_emb, b_text_emb), dim=-1)
        logits = self.head(concat_emb)

        # [bs, #classes]
        return logits


class Transformer(TextOnlyModel):

    def __init__(
            self,
            model_card,
            head,
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

        self.head = head()
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


class PairTransformer(Transformer):

    def encode_text(
            self,
            text
    ):
        input_ids, attention_mask = text

        tokens_emb = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        tokens_emb = self.dropout(tokens_emb)
        text_emb = (tokens_emb * attention_mask[:, :, None]).sum(dim=1)
        text_emb = text_emb / attention_mask.sum(dim=1)[:, None]

        return text_emb

    def forward(
            self,
            text
    ):
        a_text, b_text = text

        a_text_emb = self.encode_text(text=a_text)
        b_text_emb = self.encode_text(text=b_text)

        concat_emb = th.concat((a_text_emb, b_text_emb), dim=-1)
        logits = self.head(concat_emb)
        return logits
