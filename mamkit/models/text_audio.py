import torch as th
from transformers import AutoModel, AutoConfig

from mamkit.modules.rnn import LSTMStack
from mamkit.modules.transformer import MulTA_CrossAttentionBlock


class TextAudioModel(th.nn.Module):

    def forward(
            self,
            inputs
    ):
        pass


class BiLSTM(TextAudioModel):

    def __init__(
            self,
            vocab_size,
            text_embedding_dim,
            audio_embedding_dim,
            text_lstm_weights,
            audio_lstm_weights,
            head,
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            embedding_matrix=None
    ):
        super().__init__()

        self.embedding = th.nn.Embedding(num_embeddings=vocab_size,
                                         embedding_dim=text_embedding_dim,
                                         padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data = embedding_matrix

        self.text_lstm = LSTMStack(input_size=text_embedding_dim,
                                   lstm_weigths=text_lstm_weights)
        self.audio_lstm = LSTMStack(input_size=audio_embedding_dim,
                                    lstm_weigths=audio_lstm_weights)
        self.head = head()

        self.text_dropout = th.nn.Dropout(p=text_dropout_rate)
        self.audio_dropout = th.nn.Dropout(p=audio_dropout_rate)

    def forward(
            self,
            inputs
    ):
        # [bs, T, d_t]
        tokens_emb = self.embedding(inputs['text_inputs'])
        tokens_emb = self.text_dropout(tokens_emb)

        # [bs, d'_t]
        text_emb = self.text_lstm(tokens_emb)

        # [bs, A, d_a]
        audio_features = inputs['audio_inputs']
        audio_features = self.audio_dropout(audio_features)

        # [bs, d'_a]
        audio_emb = self.audio_lstm(audio_features)

        # [bs, d'_t + d'_a]
        concat_emb = th.concat((text_emb, audio_emb), dim=-1)

        logits = self.head(concat_emb)

        # [bs, #classes]
        return logits


class PairBiLSTM(BiLSTM):

    def encode_input(
            self,
            text,
            audio_features,
    ):
        # [bs, T, d_t]
        tokens_emb = self.embedding(text)
        tokens_emb = self.text_dropout(tokens_emb)

        # [bs, d'_t]
        text_emb = self.text_lstm(tokens_emb)

        # [bs, A, d_a]
        audio_features = self.audio_dropout(audio_features)

        # [bs, d'_a]
        audio_emb = self.audio_lstm(audio_features)

        # [bs, d'_t + d'_a]
        concat_emb = th.concat((text_emb, audio_emb), dim=-1)

        return concat_emb

    def forward(
            self,
            inputs
    ):
        a_concat_emb = self.encode_input(text=inputs['text_a_inputs'], audio_features=inputs['audio_a_inputs'])
        b_concat_emb = self.encode_input(text=inputs['text_b_inputs'], audio_features=inputs['audio_b_inputs'])

        concat_emb = th.concat((a_concat_emb, b_concat_emb), dim=-1)

        logits = self.head(concat_emb)

        # [bs, #classes]
        return logits


class MMTransformer(TextAudioModel):

    def __init__(
            self,
            model_card,
            head,
            audio_embedding_dim,
            lstm_weights,
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
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

        self.audio_lstm = LSTMStack(input_size=audio_embedding_dim,
                                    lstm_weigths=lstm_weights)

        self.head = head()
        self.text_dropout = th.nn.Dropout(p=text_dropout_rate)
        self.audio_dropout = th.nn.Dropout(p=audio_dropout_rate)

    def forward(
            self,
            inputs
    ):
        text_ids, text_mask = inputs['text_inputs'], inputs['text_input_mask']
        audio, audio_mask = inputs['audio_inputs'], inputs['audio_input_mask']

        # [bs, T, d_t]
        tokens_emb = self.model(input_ids=text_ids, attention_mask=text_mask).last_hidden_state
        tokens_emb = self.text_dropout(tokens_emb)

        # [bs, T, d'_t]
        text_emb = (tokens_emb * text_mask[:, :, None]).sum(dim=1)
        text_emb = text_emb / text_mask.sum(dim=1)[:, None]

        # [bs, A, d_a]
        audio_features = self.audio_dropout(audio)

        # [bs, d'_a]
        audio_emb = self.audio_lstm(audio_features)

        # [bs, d'_t + d'_a]
        concat_emb = th.concat((text_emb, audio_emb), dim=-1)

        logits = self.head(concat_emb)

        # [bs, #classes]
        return logits


class PairMMTransformer(MMTransformer):

    def encode_inputs(
            self,
            text_ids,
            text_mask,
            audio_features,
            audio_mask
    ):
        # [bs, T, d_t]
        tokens_emb = self.model(input_ids=text_ids, attention_mask=text_mask).last_hidden_state
        tokens_emb = self.text_dropout(tokens_emb)

        # [bs, T, d'_t]
        text_emb = (tokens_emb * text_mask[:, :, None]).sum(dim=1)
        text_emb = text_emb / text_mask.sum(dim=1)[:, None]

        # [bs, A, d_a]
        audio_features = self.audio_dropout(audio_features)

        # [bs, d'_a]
        audio_emb = self.audio_lstm(audio_features)

        # [bs, d'_t + d'_a]
        concat_emb = th.concat((text_emb, audio_emb), dim=-1)

        return concat_emb

    def forward(
            self,
            inputs
    ):
        a_concat_emb = self.encode_inputs(text_ids=inputs['text_a_inputs'], text_mask=inputs['text_a_mask'],
                                          audio_features=inputs['audio_a_inputs'], audio_mask=inputs['audio_a_inputs'])
        b_concat_emb = self.encode_inputs(text_ids=inputs['text_b_inputs'], text_mask=inputs['text_b_mask'],
                                          audio_features=inputs['audio_b_inputs'], audio_mask=inputs['audio_b_inputs'])

        concat_emb = th.concat((a_concat_emb, b_concat_emb), dim=-1)

        logits = self.head(concat_emb)

        # [bs, #classes]
        return logits


class CSA(TextAudioModel):
    def __init__(
            self,
            transformer,
            head,
            positional_encoder,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1
    ):
        """
        Args:
            transformer: transformer to use
            head: head to use
            positional_encoder: positional encoder to use
        """
        super().__init__()
        self.transformer = transformer()
        self.head = head()
        self.positional_encoder = positional_encoder()
        self.text_dropout = th.nn.Dropout(p=text_dropout_rate)
        self.audio_dropout = th.nn.Dropout(p=audio_dropout_rate)

    def forward(
            self,
            inputs
    ):
        # tokens_emb        -> [bs, N_t, d]
        # text_attentions   -> [bs, N]
        # audio_features    -> [bs, N_a, d]
        # audio_attentions  -> [bs, N]
        tokens_emb, text_attentions = inputs['text_inputs'], inputs['text_input_mask']
        audio_features, audio_attentions = inputs['audio_inputs'], inputs['audio_input_mask']

        concatenated_attentions = th.cat((text_attentions, audio_attentions.float()), dim=1)

        audio_features = self.positional_encoder(audio_features)

        concatenated_features = th.cat((tokens_emb, audio_features), dim=1)

        transformer_output = self.transformer(concatenated_features, text_attentions, audio_attentions)

        # Dropout and LayerNorm to help training phase
        transformer_output = self.audio_dropout(transformer_output)

        # pooling of transformer output
        transformer_output_sum = (transformer_output * concatenated_attentions.unsqueeze(-1)).sum(dim=1)
        transformer_output_pooled = transformer_output_sum / concatenated_attentions.sum(dim=1).unsqueeze(-1)

        logits = self.head(transformer_output_pooled)

        return logits


class PairCSA(CSA):

    def encode_inputs(
            self,
            tokens_emb,
            text_attentions,
            audio_features,
            audio_attentions
    ):
        # tokens_emb        -> [bs, N_t, d]
        # text_attentions   -> [bs, N]
        # audio_features    -> [bs, N_a, d]
        # audio_attentions  -> [bs, N]

        concatenated_attentions = th.cat((text_attentions, audio_attentions.float()), dim=1)

        audio_features = self.positional_encoder(audio_features)

        concatenated_features = th.cat((tokens_emb, audio_features), dim=1)

        transformer_output = self.transformer(concatenated_features, text_attentions, audio_attentions)

        # Dropout and LayerNorm to help training phase
        transformer_output = self.audio_dropout(transformer_output)

        # pooling of transformer output
        transformer_output_sum = (transformer_output * concatenated_attentions.unsqueeze(-1)).sum(dim=1)
        transformer_output_pooled = transformer_output_sum / concatenated_attentions.sum(dim=1).unsqueeze(-1)

        return transformer_output_pooled

    def forward(
            self,
            inputs
    ):
        a_emb = self.encode_inputs(tokens_emb=inputs['text_a_inputs'], text_attentions=inputs['text_a_mask'],
                                   audio_features=inputs['audio_a_inputs'], audio_attentions=inputs['audio_a_mask'])
        b_emb = self.encode_inputs(tokens_emb=inputs['text_b_inputs'], text_attentions=inputs['text_b_mask'],
                                   audio_features=inputs['audio_b_inputs'], audio_attentions=inputs['audio_b_mask'])

        concat_emb = th.concat((a_emb, b_emb), dim=-1)

        logits = self.head(concat_emb)

        return logits


class Ensemble(TextAudioModel):
    def __init__(
            self,
            text_head,
            audio_head,
            audio_encoder,
            positional_encoder,
            audio_embedding_dim,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            lower_bound=0.3,
            upper_bound=0.7
    ):
        super().__init__()
        self.text_head = text_head()
        self.audio_head = audio_head()
        self.audio_encoder = audio_encoder()
        self.positional_encoder = positional_encoder()
        self.layer_norm = th.nn.LayerNorm(audio_embedding_dim)
        self.audio_dropout = th.nn.Dropout(p=audio_dropout_rate)
        self.text_dropout = th.nn.Dropout(p=text_dropout_rate)

        # weight to balance the two models, 0 because (tanh(0)+1)/2 = 0.5 => equal weight to both models
        self.weight = th.nn.Parameter(th.tensor(0.0))
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(
            self,
            inputs
    ):
        # Text
        tokens_emb, text_attentions = inputs['text_inputs'], inputs['text_input_mask']

        tokens_emb = self.text_dropout(tokens_emb)

        # [bs, d_t]
        text_emb = (tokens_emb * text_attentions[:, :, None]).sum(dim=1)
        text_emb = text_emb / text_attentions.sum(dim=1)[:, None]

        text_logits = self.text_head(text_emb)

        # Audio
        audio_features, audio_attentions = inputs['audio_inputs'], inputs['audio_input_mask']

        padding_mask = ~audio_attentions.to(th.bool)
        full_attention_mask = th.zeros((audio_features.shape[1], audio_features.shape[1]), dtype=th.bool).to(
            audio_features.device)

        audio_features = self.positional_encoder(audio_features)

        transformer_output = self.audio_encoder(audio_features,
                                                mask=full_attention_mask,
                                                src_key_padding_mask=padding_mask)

        # Dropout and LayerNorm to help training phase
        transformer_output = self.audio_dropout(transformer_output)
        transformer_output = self.layer_norm(audio_features + transformer_output)

        transformer_output_sum = (transformer_output * audio_attentions.unsqueeze(-1)).sum(dim=1)
        transformer_output_pooled = transformer_output_sum / audio_attentions.sum(dim=1).unsqueeze(-1)

        audio_logits = self.audio_head(transformer_output_pooled)

        # Ensemble
        text_probabilities = th.nn.functional.softmax(text_logits, dim=-1)
        audio_probabilities = th.nn.functional.softmax(audio_logits, dim=-1)

        # coefficient to balance the two models based on weight learned
        # (tanh + 1) / 2 to have values in [0,1]
        coefficient = (th.tanh(self.weight) + 1) / 2

        # next step is to have values in [lower_bound, upper_bound] to avoid too much imbalance
        coefficient = coefficient * (self.upper_bound - self.lower_bound) + self.lower_bound

        final_prob = coefficient * text_probabilities + (1 - coefficient) * audio_probabilities

        return th.log(final_prob + 1e-08)


class PairEnsemble(Ensemble):

    def encode_text(
            self,
            tokens_emb,
            text_attentions
    ):
        text_emb = (tokens_emb * text_attentions[:, :, None]).sum(dim=1)
        text_emb = text_emb / text_attentions.sum(dim=1)[:, None]

        return text_emb

    def encode_audio(
            self,
            audio_features,
            audio_attentions
    ):
        padding_mask = ~audio_attentions.to(th.bool)
        full_attention_mask = th.zeros((audio_features.shape[1], audio_features.shape[1]), dtype=th.bool).to(
            audio_features.device)

        audio_features = self.positional_encoder(audio_features)

        transformer_output = self.audio_encoder(audio_features,
                                                mask=full_attention_mask,
                                                src_key_padding_mask=padding_mask)

        # Dropout and LayerNorm to help training phase
        transformer_output = self.audio_dropout(transformer_output)
        transformer_output = self.layer_norm(audio_features + transformer_output)

        transformer_output_sum = (transformer_output * audio_attentions.unsqueeze(-1)).sum(dim=1)
        transformer_output_pooled = transformer_output_sum / audio_attentions.sum(dim=1).unsqueeze(-1)

        return transformer_output_pooled

    def forward(
            self,
            inputs
    ):
        a_text_emb = self.encode_text(tokens_emb=inputs['text_a_inputs'], text_attentions=inputs['text_a_mask'])
        b_text_emb = self.encode_text(tokens_emb=inputs['text_b_inputs'], text_attentions=inputs['text_b_mask'])
        concat_text_emb = th.concat((a_text_emb, b_text_emb), dim=-1)
        text_logits = self.text_head(concat_text_emb)

        # Audio
        a_audio_emb = self.encode_audio(audio_features=inputs['audio_a_inputs'],
                                        audio_attentions=inputs['audio_a_input_mask'])
        b_audio_emb = self.encode_audio(audio_features=inputs['audio_b_inputs'],
                                        audio_attentions=inputs['audio_b_input_mask'])
        concat_audio_emb = th.concat((a_audio_emb, b_audio_emb), dim=-1)
        audio_logits = self.audio_head(concat_audio_emb)

        # Ensemble
        text_probabilities = th.nn.functional.softmax(text_logits)
        audio_probabilities = th.nn.functional.softmax(audio_logits)

        # coefficient to balance the two models based on weight learned
        # (tanh + 1) / 2 to have values in [0,1]
        coefficient = (th.tanh(self.weight) + 1) / 2

        # next step is to have values in [lower_bound, upper_bound] to avoid too much imbalance
        coefficient = coefficient * (self.upper_bound - self.lower_bound) + self.lower_bound

        final_prob = coefficient * text_probabilities + (1 - coefficient) * audio_probabilities

        return th.log(final_prob + 1e-08)


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
            positional_encoder,
            audio_dropout_rate=0.1,
            text_dropout_rate=0.1
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
        self.head = head()
        self.text_crossmodal_blocks = th.nn.ModuleList([
            MulTA_CrossAttentionBlock(self.embedding_dim, self.d_ffn, dropout_prob=text_dropout_rate) for _ in
            range(self.n_blocks)
        ])
        self.audio_crossmodal_blocks = th.nn.ModuleList([
            MulTA_CrossAttentionBlock(self.embedding_dim, self.d_ffn, dropout_prob=audio_dropout_rate) for _ in
            range(self.n_blocks)
        ])
        self.positional_encoder = positional_encoder()

    def forward(
            self,
            inputs
    ):
        text_features, text_attentions = inputs['text_inputs'], inputs['text_input_mask']
        audio_features, audio_attentions = inputs['audio_inputs'], inputs['audio_input_mask']

        text_features = self.positional_encoder(text_features)
        audio_features = self.positional_encoder(audio_features)

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


class PairMulTA(MulTA):

    def encode_inputs(
            self,
            text_features,
            text_attentions,
            audio_features,
            audio_attentions
    ):
        text_features = self.positional_encoder(text_features)
        audio_features = self.positional_encoder(audio_features)

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

        return text_audio

    def forward(
            self,
            inputs
    ):
        a_emb = self.encode_inputs(text_features=inputs['text_a_inputs'],
                                   text_attentions=inputs['text_a_input_mask'],
                                   audio_features=inputs['audio_a_inputs'],
                                   audio_attentions=inputs['audio_a_input_mask'])
        b_emb = self.encode_inputs(text_features=inputs['text_b_inputs'],
                                   text_attentions=inputs['text_b_input_mask'],
                                   audio_features=inputs['audio_b_inputs'],
                                   audio_attentions=inputs['audio_b_input_mask'])

        concat_emb = th.concat((a_emb, b_emb), dim=-1)

        logits = self.head(concat_emb)
        return logits
