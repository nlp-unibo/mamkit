import torch

class MAMKitBase(torch.nn.Module):
    def forward(self, text_features, text_attentions, audio_features, audio_attentions, **kwargs):
        raise NotImplementedError