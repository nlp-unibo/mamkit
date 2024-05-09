import torch as th
from transformers import AutoModel
from mamkit.modules.transformer_modules import MulTA_CrossAttentionBlock, PositionalEncoding


class TextAudioModel(th.nn.Module):

    def forward(
            self,
            text,
            audio
    ):
        pass


class BiLSTM(TextAudioModel):

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
        input_size = lstm_weights[-1] * 4
        for weight in mlp_weights:
            self.pre_classifier.append(th.nn.Linear(in_features=input_size,
                                                    out_features=weight))
            self.pre_classifier.append(th.nn.LeakyReLU())
            self.pre_classifier.append(th.nn.Dropout(p=dropout_rate))
            input_size = weight

        self.classifier = th.nn.Linear(in_features=mlp_weights[-1], out_features=num_classes)

    def forward(
            self,
            text,
            audio,
    ):

        # [bs, N, d]
        tokens_emb = self.embedding(text)

        # [bs, d']
        _, (text_emb, _) = self.text_lstm(tokens_emb)
        text_emb = text_emb.permute(1, 0, 2).reshape(tokens_emb.shape[0], -1)

        audio_emb = self.audio_lstm(audio)

        concat_emb = th.concat((text_emb, audio_emb), dim=-1)

        logits = self.pre_classifier(concat_emb)
        logits = self.classifier(logits)

        # [bs, #classes]
        return logits


# TODO: fix cnn
class MArgNet(TextAudioModel):

    def __init__(
            self,
            transformer_model_card,
            embedding_dim,
            lstm_weights,
            cnn_info,
            mlp_weights,
            dropout_rate,
            num_classes
    ):
        super().__init__()
        
        # Text
        self.transformer = AutoModel.from_pretrained(transformer_model_card)

        # Audio
        self.lstm = th.nn.Sequential()
        input_size = embedding_dim
        for weight in lstm_weights:
            self.lstm.append(th.nn.LSTM(input_size=input_size,
                                        hidden_size=weight,
                                        batch_first=True,
                                        bidirectional=True))
            input_size = weight

        self.dropout = th.nn.Dropout(dropout_rate)

        self.pre_classifier = th.nn.Sequential()
        input_size = lstm_weights[-1] * 2
        for weight in mlp_weights:
            self.pre_classifier.append(th.nn.Linear(in_features=input_size,
                                                    out_features=weight))
            self.pre_classifier.append(th.nn.LeakyReLU())
            self.pre_classifier.append(th.nn.Dropout(p=dropout_rate))
            input_size = weight

        self.classifier = th.nn.Linear(in_features=mlp_weights[-1], out_features=num_classes)


class CSA(TextAudioModel):
    def __init__(
            self,
            transformer,
            head,
            positional_encoder
    ):
        """
        Args:
            transformer: transformer to use
            head: head to use
            positional_encoder: positional encoder to use
        """
        super().__init__()
        self.transformer = transformer
        self.head = head
        self.pos_encoder = positional_encoder

    def forward(
            self,
            text,
            audio,
            **kwargs
    ):
        # tokens_emb        -> [bs, N_t, d]
        # text_attentions   -> [bs, N]
        # audio_features    -> [bs, N_a, d]
        # audio_attentions  -> [bs, N]
        tokens_emb, text_attentions = text
        audio_features, audio_attentions = audio

        concatenated_attentions = th.cat((text_attentions, audio_attentions.float()), dim=1)

        audio_features = self.pos_encoder(audio_features)

        concatenated_features = th.cat((tokens_emb, audio_features), dim=1)

        transformer_output = self.transformer(concatenated_features, text_attentions, audio_attentions)

        # pooling of transformer output
        transformer_output_sum = (transformer_output * concatenated_attentions.unsqueeze(-1)).sum(dim=1)
        transformer_output_pooled = transformer_output_sum / concatenated_attentions.sum(dim=1).unsqueeze(-1)

        logits = self.head(transformer_output_pooled)

        return logits


class Ensemble(TextAudioModel):
    def __init__(
            self,
            text_model,
            audio_model,
            lower_bound=0.3,
            upper_bound=0.7
    ):
        """
        Args:
            text_model: text model to use
            audio_model: audio model to use
            lower_bound: lower bound for the weight
            upper_bound: upper bound for the weight
        """
        super().__init__()
        self.text_model = text_model
        self.audio_model = audio_model
        # weight to balance the two models, 0 because (tanh(0)+1)/2 = 0.5 => equal weight to both models
        self.weight = th.nn.Parameter(th.tensor(0.0))
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(
            self,
            text,
            audio
    ):
        text_logits = self.text_model(text)
        audio_logits = self.audio_model(audio)

        text_probabilities = th.nn.functional.softmax(text_logits)
        audio_probabilities = th.nn.functional.softmax(audio_logits)

        # coefficient to balance the two models based on weight learned
        # (tanh + 1) / 2 to have values in [0,1]
        coefficient = (th.tanh(self.weight) + 1) / 2
        # next step is to have values in [lower_bound, upper_bound] to avoid too much imbalance
        coefficient = coefficient * (self.upper_bound - self.lower_bound) + self.lower_bound

        return coefficient * text_probabilities + (1 - coefficient) * audio_probabilities


class MulTA(TextAudioModel):
    """
    Class for the unaligned multimodal model
    """

    def __init__(
            self,
            embedding_dim,
            d_ffn,
            n_blocks,
            head,
            dropout_prob=0.1
    ):
        """
        Args:
            embedding_dim: dimension of the embedding
            d_ffn: dimension of the feed forward layer
            n_blocks: number of blocks to use
            head: head to use
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.d_ffn = d_ffn
        self.n_blocks = n_blocks
        self.head = head
        self.dropout_prob = dropout_prob
        self.text_crossmodal_blocks = th.nn.ModuleList([
            MulTA_CrossAttentionBlock(self.embedding_dim, self.d_ffn, dropout_prob=self.dropout_prob) for _ in
            range(self.n_blocks)
        ])
        self.audio_crossmodal_blocks = th.nn.ModuleList([
            MulTA_CrossAttentionBlock(self.embedding_dim, self.d_ffn, dropout_prob=self.dropout_prob) for _ in
            range(self.n_blocks)
        ])
        self.pos_encoder = PositionalEncoding(embedding_dim, dual_modality=False)

    def forward(
            self,
            text,
            audio
    ):
        text_features, text_attentions = text
        audio_features, audio_attentions = audio

        text_features = self.pos_encoder(text_features)
        audio_features = self.pos_encoder(audio_features)

        # cross modal attention blocks for text
        # using audio features as key and value and text features as query
        text_crossmodal_out = text_features
        for cm_block in self.text_crossmodal_blocks:
            text_crossmodal_out = cm_block(text_crossmodal_out, audio_features, audio_attentions)

        # cross modal attention blocks for audio
        # using text features as key and value and audio features as query
        audio_crossmodal_out = audio_features
        for cm_block in self.audio_crossmodal_blocks:
            audio_crossmodal_out = cm_block(audio_crossmodal_out, text_features, text_attentions)

        # pooling of transformer output
        text_crossmodal_out_mean = th.mean(text_crossmodal_out, dim=1)
        audio_crossmodal_out_mean = th.mean(audio_crossmodal_out, dim=1)

        # concatenate text and audio features
        text_audio = th.cat((text_crossmodal_out_mean, audio_crossmodal_out_mean), dim=-1)

        logits = self.head(text_audio)
        return logits
