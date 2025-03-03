import torch as th
from transformers import AutoModel, AutoConfig

from mamkit.modules.rnn import LSTMStack


class TextOnlyModel(th.nn.Module):

    def forward(
            self,
            inputs
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
            inputs
    ):

        # [bs, N, d]
        tokens_emb = self.embedding(inputs['inputs'])

        tokens_emb = self.dropout(tokens_emb)

        # [bs, d']
        text_emb = self.lstm(tokens_emb)

        logits = self.head(text_emb)

        # [bs, #classes]
        return logits


class PairBiLSTM(BiLSTM):

    def forward(
            self,
            inputs
    ):
        # A input
        # [bs, T, d_t]
        a_tokens_emb = self.embedding(inputs['a_inputs'])
        a_tokens_emb = self.dropout(a_tokens_emb)

        # [bs, d']
        a_text_emb = self.lstm(a_tokens_emb)

        # B input
        # [bs, T, d_t]
        b_tokens_emb = self.embedding(inputs['b_inputs'])
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
            inputs
    ):
        input_ids = inputs['inputs']
        attention_mask = inputs['input_mask']

        tokens_emb = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        tokens_emb = self.dropout(tokens_emb)
        text_emb = (tokens_emb * attention_mask[:, :, None]).sum(dim=1)
        text_emb = text_emb / attention_mask.sum(dim=1)[:, None]

        logits = self.head(text_emb)
        return logits


class PairTransformer(Transformer):

    def encode_text(
            self,
            input_ids,
            attention_mask
    ):
        tokens_emb = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        tokens_emb = self.dropout(tokens_emb)
        text_emb = (tokens_emb * attention_mask[:, :, None]).sum(dim=1)
        text_emb = text_emb / attention_mask.sum(dim=1)[:, None]

        return text_emb

    def forward(
            self,
            inputs
    ):
        a_text_emb = self.encode_text(input_ids=inputs['a_inputs'], attention_mask=inputs['a_input_mask'])
        b_text_emb = self.encode_text(input_ids=inputs['b_inputs'], attention_mask=inputs['b_input_mask'])

        concat_emb = th.concat((a_text_emb, b_text_emb), dim=-1)
        logits = self.head(concat_emb)
        return logits
