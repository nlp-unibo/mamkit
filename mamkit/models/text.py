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
            mlp_weights,
            dropout_rate,
            num_classes,
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

        self.pre_classifier = th.nn.Sequential()
        input_size = lstm_weights[-1] * 2
        for weight in mlp_weights:
            self.pre_classifier.append(th.nn.Linear(in_features=input_size,
                                                    out_features=weight))
            self.pre_classifier.append(th.nn.LeakyReLU())
            self.pre_classifier.append(th.nn.Dropout(p=dropout_rate))
            input_size = weight

        self.classifier = th.nn.Linear(in_features=mlp_weights[-1], out_features=num_classes)

    def forward(
            self,
            text
    ):
        # [bs, N, d]
        tokens_emb = self.embedding(text)

        # [bs, d']
        text_emb = self.lstm(tokens_emb)

        logits = self.pre_classifier(text_emb)
        logits = self.classifier(logits)

        # [bs, #classes]
        return logits


class Transformer(TextOnlyModel):

    def __init__(
            self,
            model_card,
            num_classes
    ):
        super().__init__()

        self.model_card = model_card
        self.model_config = AutoConfig.from_pretrained(model_card)
        self.model = AutoModel.from_pretrained(model_card)
        self.classifier = th.nn.Linear(in_features=self.model_config.hidden_dim,
                                       out_features=num_classes)

    def forward(
            self,
            text
    ):
        input_ids, attention_mask = text

        tokens_emb = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_emb = th.mean(tokens_emb, dim=1)

        logits = self.classifier(text_emb)

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
