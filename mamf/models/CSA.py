import torch
from core import MAMFBase

class MAMFCSA(MAMFBase):
    def __init__(self, transformer, head, positional_encoder):
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

    
    def forward(self, text_features, text_attentions, audio_features, audio_attentions, **kwargs):
        """
        Forward pass of the model
        Args:
            text_features: texts to use
            text_attentions: text attentions to use
            audio_features: audio features to use
            audio_attentions: audio attentions to use
        """

        concatenated_attentions = torch.cat((text_attentions, audio_attentions.float()), dim=1)
        
        audio_features = self.pos_encoder(audio_features)
        
        concatenated_features = torch.cat((text_features, audio_features), dim=1)

        transformer_output = self.transformer(concatenated_features, text_attentions, audio_attentions)

        # pooling of transformer output        
        transformer_output_sum = (transformer_output * concatenated_attentions.unsqueeze(-1)).sum(axis=1)
        transformer_output_pooled = transformer_output_sum / concatenated_attentions.sum(axis=1).unsqueeze(-1)
        return self.head(transformer_output_pooled)


        