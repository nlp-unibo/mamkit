import torch

class MAMKitBase(torch.nn.Module):
    def forward(self, text_data, audio_data, **kwargs):
        raise NotImplementedError