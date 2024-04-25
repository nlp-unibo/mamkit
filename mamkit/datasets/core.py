import torch
import torch.utils

from typing import Tuple

class MAMKitDataset(torch.utils.data.Dataset):
    # returns [text_features, text_attentions, audio_features, audio_attentions, target]
    def __getitem__(self, idx) -> Tuple[list, list, list, list, list]:
        raise NotImplementedError


class MAMKitModularDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, text_preprocessor, audio_preprocessor, target_preprocessor):
        super().__init__()
        self.dataset = dataset
        self.text_preprocessor = text_preprocessor
        self.audio_preprocessor = audio_preprocessor
        self.target_preprocessor = target_preprocessor


    def __getitem__(self, idx):
        text, audio, target = self.dataset[idx]
        
        text_data = self.text_preprocessor(text) # epected to return a list of features or a list of features and attentions
        audio_data = self.audio_preprocessor(audio) # epected to return a list of features or a list of features and attentions

        target = self.target_preprocessor(target)

        return text_data, audio_data, target

    def __len__(self):
        return len(self.text_dataset)



class MAMKitListDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        assert all(len(datasets[0]) == len(dataset) for dataset in datasets)
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return tuple(dataset[idx] for dataset in self.datasets)