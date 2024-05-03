import torch
import torch.utils
from itertools import compress

from typing import Tuple


# We want to return a boolean array to filter labels depending on the task
def get_task_map(labels, taskname):

    if taskname.lower() == 'asd':
        return [True if label in ['Claim', 'Premise', 'O'] else False for label in labels]
    
    if taskname.lower() == 'acc':
        return [True if label in ['Claim', 'Premise'] else False for label in labels]
    
    raise ValueError(f'Taskname {taskname} not supported. Supported tasks: ["ASD", "ACC"]')


def get_task_labels(labels, taskname):
    
    # In ASD we want to merge Caim and Premise labels into "ARG", while the rest should be named "Not-ARG"
    if taskname.lower() == 'asd':
        return ['ARG' if label in ['Claim', 'Premise'] else 'Not-ARG' for label in labels]

    # In ACC we want only Claim and Premise labels, 
    if taskname.lower() == 'acc':
        return labels
    
    raise ValueError(f'Taskname {taskname} not supported. Supported tasks: ["ASD", "ACC"]')




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
    def __init__(self, text_features, audio_features, labels, taskname):
        self.task_map = get_task_map(labels, taskname)
        self.text_features = list(compress(text_features, self.task_map))
        self.audio_features = list(compress(audio_features, self.task_map))
        self.labels = get_task_labels(list(compress(labels, self.task_map)), taskname)
    
    def __getitem__(self, idx):
        return self.text_features[idx], self.audio_features[idx], self.labels[idx]

class MAMKitMonomodalDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, taskname):
        self.task_map = get_task_map(labels, taskname)
        self.features = list(compress(features, self.task_map))
        self.labels = get_task_labels(list(compress(labels, self.task_map)), taskname)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]