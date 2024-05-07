import abc
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel


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


@dataclass
class DataInfo:
    train: Dataset
    val: Optional[Dataset]
    test: Optional[Dataset]


class Loader(abc.ABC):

    def __init__(
            self,
            task_name: str,
            input_mode: str
    ):
        self.task_name = task_name
        self.input_mode = input_mode
        self.texts = None
        self.audio = None
        self.labels = None

    @abc.abstractmethod
    def get_splits(
            self,
    ) -> DataInfo:
        pass


class UnimodalDataset(Dataset):

    def __init__(
            self,
            inputs,
            labels
    ):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(
            self,
            idx
    ):
        return self.inputs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class MultimodalDataset(Dataset):

    def __init__(
            self,
            texts,
            audio,
            labels
    ):
        self.texts = texts
        self.audio = audio
        self.labels = labels

    def __getitem__(
            self,
            idx
    ):
        return self.texts[idx], self.audio[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class UnimodalCollator:
    def __init__(self, features_collator, label_collator):
        self.features_collator = features_collator
        self.label_collator = label_collator

    def __call__(self, batch):
        features_raw, labels = zip(*batch)
        if self.features_collator is None:
            features_collated = features_raw
        else:
            features_collated = self.features_collator(features_raw)

        if self.label_collator is None:
            labels_collated = labels
        else:
            labels_collated = self.label_collator(labels)

        return (*features_collated, labels_collated)


class MultimodalCollator:
    def __init__(self, text_collator, audio_collator, label_collator):
        self.text_collator = text_collator
        self.audio_collator = audio_collator
        self.label_collator = label_collator

    def __call__(self, batch):
        text_raw, audio_raw, labels = zip(*batch)
        if self.text_collator is None:
            text_collated = text_raw
        else:
            text_collated = self.text_collator(text_raw)

        if self.audio_collator is None:
            audio_collated = audio_raw
        else:
            audio_collated = self.audio_collator(audio_raw)

        if self.label_collator is None:
            labels_collated = labels
        else:
            labels_collated = self.label_collator(labels)

        return text_collated, audio_collated, labels_collated


class BERT_Collator:
    def __init__(self, model_card='bert-base-uncased'):
        self.model_card = model_card
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_card)
        self.model = BertModel.from_pretrained(model_card).to(self.device)

    def __call__(self, text_raw):
        tokenized = self.tokenizer(text_raw, padding=True, truncation=True, return_tensors='pt').to(self.device)
        text_features = self.model(**tokenized).last_hidden_state
        text_attentions = tokenized.attention_mask
        return text_features, text_attentions
