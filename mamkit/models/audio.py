import torch as th


class AudioOnlyModel(th.nn.Module):

    def forward(
            self,
            audio
    ):
        pass


class BiLSTM(AudioOnlyModel):

    def __init__(
            self,
            embedding_dim,
            lstm_weights,
            mlp_weights,
            dropout_rate,
            num_classes
    ):
        super().__init__()

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
            audio
    ):
        # audio -> [bs, N, d]

        # [bs, d']
        _, (audio_emb, _) = self.lstm(audio)
        audio_emb = audio_emb.permute(1, 0, 2).reshape(audio_emb.shape[0], -1)

        logits = self.pre_classifier(audio_emb)
        logits = self.classifier(logits)

        # [bs, #classes]
        return logits


# TODO: fix cnn blocks
class BiLSTMCNN(AudioOnlyModel):

    def __init__(
            self,
            embedding_dim,
            lstm_weights,
            audio_shape,
            cnn_info,
            mlp_weights,
            dropout_rate,
            num_classes
    ):
        super().__init__()

        self.audio_shape = audio_shape

        self.lstm = th.nn.Sequential()
        input_size = embedding_dim
        for weight in lstm_weights:
            self.lstm.append(th.nn.LSTM(input_size=input_size,
                                        hidden_size=weight,
                                        batch_first=True,
                                        bidirectional=True))
            input_size = weight

        self.dropout = th.nn.Dropout(dropout_rate)

        self.cnn = th.nn.Sequential()
        in_channels = lstm_weights[-1] * 2
        for info in cnn_info:
            self.cnn.append(th.nn.Conv2d(in_channels=in_channels,
                                         out_channels=info['out_channels'],
                                         kernel_size=info['kernel_size'],
                                         stride=info['stride'],
                                         padding='valid'))
            self.cnn.append(th.nn.BatchNorm2d(num_features=info['out_channels']))
            self.cnn.append(th.nn.ReLU())
            self.cnn.append(th.nn.Dropout(dropout_rate))
            self.cnn.append(th.nn.MaxPool2d(stride=info['pool_stride'],
                                            kernel_size=info['pool_size']))
            self.cnn.append(th.nn.Flatten())

            in_channels = info['out_channels']

        self.pre_classifier = th.nn.Sequential()
        input_size = lstm_weights[-1] * 2
        for weight in mlp_weights:
            self.pre_classifier.append(th.nn.Linear(in_features=input_size,
                                                    out_features=weight))
            self.pre_classifier.append(th.nn.BatchNorm1d(num_features=weight))
            self.pre_classifier.append(th.nn.LeakyReLU())
            self.pre_classifier.append(th.nn.Dropout(p=dropout_rate))
            input_size = weight

        self.classifier = th.nn.Linear(in_features=mlp_weights[-1], out_features=num_classes)


class TransformerHead(AudioOnlyModel):

    def __init__(
            self,
            head: th.nn.Module
    ):
        super().__init__()
        self.head = head

    def forward(
            self,
            audio
    ):
        # audio_features     -> [bs, N, d]
        # attention_mask     -> [bs, N]
        audio_features, attention_mask = audio

        # [bs, d]
        text_emb = (audio_features * attention_mask[:, :, None]).sum(dim=1)
        text_emb = text_emb / attention_mask.sum(dim=1)[:, None]

        logits = self.head(text_emb)
        return logits
