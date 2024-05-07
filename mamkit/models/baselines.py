from .core import MAMKitBase


class MAMKitTextOnly(MAMKitBase):
    """
    Class for the text-only model
    """

    def __init__(self, head):
        """
        Args:
            head: head to use
        """
        super().__init__()
        self.head = head

    def forward(self, data):
        """
        Forward pass of the model
        Args:
            data: data to use
        """
        text_features, text_attentions = data
        # pooling transformer output
        text_features_sum = (text_features * text_attentions.unsqueeze(-1)).sum(axis=1)
        text_features_pooled = text_features_sum / text_attentions.sum(axis=1).unsqueeze(-1)
        return self.head(text_features_pooled)


class MAMKitAudioOnly(MAMKitBase):
    """
    Class for the audio-only model
    """

    def __init__(self, head, dropout=0.1):
        """
        Args:
            transformer: transformer to use
            head: head to use
        """
        super().__init__()
        self.head = head
        # self.positional_encoding = PositionalEncoding(d_model=transformer.d_model, dropout=dropout)
        # self.transformer = transformer
        # self.layer_norm = LayerNorm(d_model=transformer.d_model)
        # self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        """
        Forward pass of the model
        Args:
            data: audio to use
        """
        audio_features, audio_attention = data

        # padding_mask = ~audio_attention.to(torch.bool)        
        # full_attention_mask = torch.zeros((audio_features.shape[1],audio_features.shape[1]), dtype=torch.bool).to(audio_features.device)

        # audio_features = self.positional_encoding(audio_features)
        # transformer_output = self.transformer(audio_features, mask=full_attention_mask, src_key_padding_mask=padding_mask)
        # transformer_output = self.dropout(transformer_output)
        # transformer_output = self.layer_norm(audio_features + transformer_output)

        # pooling transformer output
        audio_features_sum = (audio_features * audio_attention.unsqueeze(-1)).sum(axis=1)
        audio_features_pooled = audio_features_sum / audio_attention.sum(axis=1).unsqueeze(-1)

        return self.head(audio_features_pooled)
