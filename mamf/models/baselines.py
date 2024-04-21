import torch
from core import MAMFBase


class MAMFTextOnly(MAMFBase):
    """
    Class for the text-only model
    """
    def __init__(self, head):
        """
        Args:
            tokenizer: tokenizer to use
            embedder: embedder to use
            head: head to use
        """
        super().__init__()
        self.head = head

    def forward(self, text_features, text_attentions, audio_features, audio_attention):
        """
        Forward pass of the model
        Args:
            text_features: texts to use
            text_attentions: text attentions to use
            audio_features: audio features to use
            audio_attentions: audio attentions to use
        """

        # pooling transformer output
        text_features_sum = (text_features * text_attentions.unsqueeze(-1)).sum(axis=1)
        text_features_pooled = text_features_sum / text_attentions.sum(axis=1).unsqueeze(-1)
        return self.head(text_features_pooled)


class MAMFAudioOnly(MAMFBase):
    """
    Class for the audio-only model
    """
    def __init__(self, head):
        """
        Args:
            tokenizer: tokenizer to use
            embedder: embedder to use
            head: head to use
        """
        super().__init__()
        self.head = head

    def forward(self, text_features, text_attentions, audio_features, audio_attention):
        """
        Forward pass of the model
        Args:
            text_features: texts to use
            text_attentions: text attentions to use
            audio_features: audio features to use
            audio_attentions: audio attentions to use
        """

        # pooling transformer output
        audio_features_sum = (audio_features * audio_attention.unsqueeze(-1)).sum(axis=1)
        audio_features_pooled = audio_features_sum / audio_attention.sum(axis=1).unsqueeze(-1)
        return self.head(audio_features_pooled)