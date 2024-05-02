import torch
from .core import MAMKitBase

class MAMKitCSA(MAMKitBase):
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

    
    def forward(self, data, **kwargs):
        """
        Forward pass of the model
        Args:
            text_data: texts to use
            audio_data: audio to use
        """
        text, audio = data
        text_features, text_attentions = text
        audio_features, audio_attentions = audio

        concatenated_attentions = torch.cat((text_attentions, audio_attentions.float()), dim=1)
        
        audio_features = self.pos_encoder(audio_features)
        
        concatenated_features = torch.cat((text_features, audio_features), dim=1)

        transformer_output = self.transformer(concatenated_features, text_attentions, audio_attentions)

        # pooling of transformer output        
        transformer_output_sum = (transformer_output * concatenated_attentions.unsqueeze(-1)).sum(axis=1)
        transformer_output_pooled = transformer_output_sum / concatenated_attentions.sum(axis=1).unsqueeze(-1)
        return self.head(transformer_output_pooled)


        