import torch
import torch.utils

from typing import Tuple

# TODO: Implement this function
# def prepare_for_task(train, val, test, task_name: str):
#     if task_name == 'ACC':
#         train = [x for x in train if x[-1] in ['Claim', 'Premise']]
#         val = [x for x in val if x[-1] in ['Claim', 'Premise']]
#         test = [x for x in test if x[-1] in ['Claim', 'Premise']]
#     elif task_name == 'ASD':
#         train = []
#         for x in train:
#             if x[-1] in ['Claim', 'Premise']:
#                 train.append(*x[:-1], 'Arg')
#             else:
#                 train.append(*x[:-1], 'Non-Arg')

#         val = []
#         for x in val:
#             if x[-1] in ['Claim', 'Premise']:
#                 val.append(*x[:-1], 'Arg')
#             else:
#                 val.append(*x[:-1], 'Non-Arg')

#         test = []
#         for x in test:
#             if x[-1] in ['Claim', 'Premise']:
#                 test.append(*x[:-1], 'Arg')
#             else:
#                 test.append(*x[:-1], 'Non-Arg')

#     return train, val, test

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

    def __len__(self):
        return len(self.labels)

class MAMKitMonomodalDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)