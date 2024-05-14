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
            mlp_weights,
            num_classes,
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

        self.pre_classifier = th.nn.Sequential()
        in_features = self.model_config.hidden_size
        for weight in mlp_weights:
            self.pre_classifier.append(th.nn.Linear(in_features=in_features,
                                                    out_features=weight))
            in_features = weight

        self.classifier = th.nn.Sequential()
        self.classifier.append(th.nn.Linear(in_features=in_features,
                                            out_features=num_classes))
        self.classifier.append(th.nn.ReLU())

        self.dropout = th.nn.Dropout(p=dropout_rate)
        self.bn = th.nn.BatchNorm1d(num_features=mlp_weights[-1])

    def forward(
            self,
            text
    ):
        input_ids, attention_mask = text

        tokens_emb = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_emb = th.mean(tokens_emb, dim=1)

        pre_logits = self.pre_classifier(text_emb)
        pre_logits = self.bn(pre_logits)
        pre_logits = self.dropout(pre_logits)
        logits = self.classifier(pre_logits)

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
