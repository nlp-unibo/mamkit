import torch as th
from transformers import AutoModel, AutoConfig

from mamkit.modules.rnn import LSTMStack
from mamkit.modules.transformer import PositionalEncoding


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
            audio
    ):
        # audio_features -> [bs, N, d]
        audio_features, audio_attention = audio

        # [bs, d']
        audio_emb = self.lstm(audio_features)

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


class Transformer(AudioOnlyModel):

    def __init__(
            self,
            model_card,
            num_classes
    ):
        super().__init__()

        self.model_card = model_card
        self.model_config = AutoConfig.from_pretrained(model_card)
        self.model = AutoModel.from_pretrained(model_card)
        self.classifier = th.nn.Linear(in_features=self.model_config.hidden_size,
                                       out_features=num_classes)

    def forward(
            self,
            audio
    ):
        input_ids, attention_mask = audio

        tokens_emb = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_emb = th.mean(tokens_emb, dim=1)

        logits = self.classifier(text_emb)

        return logits


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


class TransformerEncoder(AudioOnlyModel):

    def __init__(
            self,
            embedding_dim,
            encoder: th.nn.Module,
            head: th.nn.Module,
            dropout_rate=0.0
    ):
        super().__init__()

        self.encoder = encoder
        self.head = head
        self.pos_encoder = PositionalEncoding(embedding_dim, dual_modality=False)
        self.layer_norm = th.nn.LayerNorm(embedding_dim)
        self.dropout = th.nn.Dropout(p=dropout_rate)

    def forward(
            self,
            audio
    ):
        audio_features, attention_mask = audio

        padding_mask = ~attention_mask.to(th.bool)
        full_attention_mask = th.zeros((audio_features.shape[1], audio_features.shape[1]), dtype=th.bool).to(
            audio_features.device)

        audio_features = self.pos_encoder(audio_features)

        transformer_output = self.transformer(audio_features, mask=full_attention_mask,
                                              src_key_padding_mask=padding_mask)

        # Dropout and LayerNorm to help training phase
        transformer_output = self.dropout(transformer_output)
        transformer_output = self.ln(audio_features + transformer_output)

        transformer_output_sum = (transformer_output * attention_mask.unsqueeze(-1)).sum(dim=1)
        transformer_output_pooled = transformer_output_sum / attention_mask.sum(dim=1).unsqueeze(-1)

        return self.head(transformer_output_pooled)
