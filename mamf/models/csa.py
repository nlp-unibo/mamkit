import torch
import torch.nn as nn

from core import MAMFBase
from common import PositionalEncoding

class CSA(MAMFBase):
    def __init__(self, tokenizer, embedder, transformer, head, embedding_dim, max_length):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim, dual_modality=False, max_len=max_length)
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.transformer = transformer
        self.head = head

    def forward(self, text_features, text_attentions, audio_features, audio_attentions):
        concatenated_attentions = torch.cat((text_attentions, audio_attentions.float()), dim=1)
        
        audio_features = self.pos_encoder(audio_features)
        
        concatenated_features = torch.cat((text_features, audio_features), dim=1)

        transformer_output = self.transformer(concatenated_features, text_attentions, audio_attentions)

        # pooling of transformer output        
        transformer_output_sum = (transformer_output * concatenated_attentions.unsqueeze(-1)).sum(axis=1)
        transformer_output_pooled = transformer_output_sum / concatenated_attentions.sum(axis=1).unsqueeze(-1)
        return self.head(transformer_output_pooled)