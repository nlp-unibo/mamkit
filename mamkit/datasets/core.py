import torch
import torch.utils

from typing import Tuple

class MAMKitDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def download_audio(self):
        raise NotImplementedError
        
    def precompute_text(self):
        raise NotImplementedError

    def precompute_audio(self):
        raise NotImplementedError

class MAMKitPrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, text_features, audio_features, labels):
        self.text_features = text_features
        self.audio_features = audio_features
        self.labels = labels
    
    def __getitem__(self, idx):
        return self.text_features[idx], self.audio_features[idx], self.labels[idx]

class MAMKitMonomodalDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]