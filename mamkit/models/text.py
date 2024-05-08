import torch as th


class MAMKitTextOnly(th.nn.Module):

    def forward(
            self,
            text
    ):
        pass


class BiLSTMBaseline(MAMKitTextOnly):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            lstm_weights,
            mlp_weights,
            dropout_rate,
            num_classes
    ):
        super().__init__()

        self.embedding = th.nn.Embedding(num_embeddings=vocab_size,
                                         embedding_dim=embedding_dim,
                                         padding_idx=0)

        self.lstm = th.nn.Sequential()
        input_size = embedding_dim
        for weight in lstm_weights:
            self.lstm.append(th.nn.LSTM(input_size=input_size,
                                        hidden_size=weight,
                                        batch_first=True,
                                        bidirectional=True))
            input_size = weight

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
        _, (text_emb, _) = self.lstm(tokens_emb)
        text_emb = text_emb.permute(1, 0, 2).reshape(tokens_emb.shape[0], -1)

        logits = self.pre_classifier(text_emb)
        logits = self.classifier(logits)

        # [bs, #classes]
        return logits


class TransformerHead(MAMKitTextOnly):
    """
    Class for the text-only model
    """

    def __init__(
            self,
            head: th.nn.Module
    ):
        super().__init__()
        self.head = head

    def forward(
            self,
            text
    ):
        # tokens_emb     -> [bs, N, d]
        # attention_mask -> [bs, N]
        tokens_emb, attention_mask = text

        # [bs, d]
        text_emb = (tokens_emb * attention_mask[:, :, None]).sum(dim=1)
        text_emb = text_emb / attention_mask.sum(dim=1)[:, None]

        logits = self.head(text_emb)
        return logits
