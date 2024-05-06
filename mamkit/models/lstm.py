from mamkit.models.core import MAMKitBase
import torch as th


class BiLSTM(MAMKitBase):

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

        self.text_lstm = th.nn.Sequential()
        input_size = embedding_dim
        for weight in lstm_weights:
            self.text_lstm.append(th.nn.LSTM(input_size=input_size,
                                             hidden_size=weight,
                                             batch_first=True,
                                             bidirectional=True))
            input_size = weight

        self.audio_lstm = th.nn.Sequential()
        input_size = embedding_dim
        for weight in lstm_weights:
            self.audio_lstm.append(th.nn.LSTM(input_size=input_size,
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
            data,
            **kwargs
    ):
        text, audio = data
        text_features, text_attentions = text
        audio_features, audio_attentions = audio
