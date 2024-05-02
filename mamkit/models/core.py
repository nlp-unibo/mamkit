import torch

class MAMKitBase(torch.nn.Module):
    def forward(self, data, **kwargs):
        raise NotImplementedError