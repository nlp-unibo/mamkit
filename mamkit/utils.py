import lightning as L
import os
from mamkit.datasets import MAMKitPrecomputedDataset, MAMKitMonomodalDataset
import pandas as pd
from mamkit.datasets import prepare_for_task
import torch

SUPPORTED_DATASETS = {
    'usdbelec' : {
        'audio' : {
            'wav2vec2-single': 'https://huggingface.co/datasets/andreazecca3/wav2vec2-single/resolve/main/wav2vec2Single.zip',
            'wavlm-downsampled': ...,
        },
        'text' : {
            'bert' : ...
        }
    }
}

class MAMKitLightingModel(L.LightningModule):
    def __init__(self, model, loss_function, optimizer_class, **optimizer_kwargs):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch[:-1], batch[-1]        
        y_hat = self.model(inputs)
        loss = self.loss_function(y_hat, targets)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch[:-1], batch[-1]
        y_hat = self.model(inputs)
        loss = self.loss_function(y_hat, targets)
        return loss

    def test_step(self, batch, batch_idx):
        # compute accuracy
        inputs, targets = batch[:-1], batch[-1]
        y_hat = self.model(inputs)
        loss = self.loss_function(y_hat, targets)
        acc = (y_hat.argmax(dim=1) == targets).float().mean()
        self.log_dict({'loss': loss, 'acc': acc})

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

def to_lighting_model(model, loss_function, optimizer_class, **optimizer_kwargs):
    return MAMKitLightingModel(model, loss_function, optimizer_class, **optimizer_kwargs)



def load_text(dataset_name):
    if dataset_name.lower() not in SUPPORTED_DATASETS.keys():
        raise ValueError(f'Dataset {dataset_name} not supported. Supported datasets: {SUPPORTED_DATASETS.keys()}')
    if dataset_name.lower() == 'usdbelec':
        path = './data/MM-USElecDeb60to16.csv'
        text_column = 'Text'
        set_column = 'Set'
        splits = ['TRAIN', 'VALIDATION', 'TEST']
    df = pd.read_csv(path)
    train = df[df[set_column] == splits[0]][text_column].tolist()
    val = df[df[set_column] == splits[1]][text_column].tolist()
    test = df[df[set_column] == splits[2]][text_column].tolist()
    return train, val, test



def load_labels(dataset_name):
    if dataset_name.lower() not in SUPPORTED_DATASETS.keys():
        raise ValueError(f'Dataset {dataset_name} not supported. Supported datasets: {SUPPORTED_DATASETS.keys()}')
    if dataset_name.lower() == 'usdbelec':
        path = './data/MM-USElecDeb60to16.csv'
        label_column = 'Component'
        set_column = 'Set'
        splits = ['TRAIN', 'VALIDATION', 'TEST']
    df = pd.read_csv(path)
    train = df[df[set_column] == splits[0]][label_column].tolist()
    val = df[df[set_column] == splits[1]][label_column].tolist()
    test = df[df[set_column] == splits[2]][label_column].tolist()
    return train, val, test



def load_precomputed_audio(dataset_name, audio_preprocessing, download_dir='./data'):
    if audio_preprocessing.lower() not in SUPPORTED_DATASETS[dataset_name]['audio'].keys():
        raise ValueError(f'Audio preprocessing {audio_preprocessing} not supported. Supported audio preprocessing: {SUPPORTED_DATASETS[dataset_name]["audio"].keys()}')
    download_link = SUPPORTED_DATASETS[dataset_name]['audio'][audio_preprocessing.lower()]
    dataset_dir = os.path.join(download_dir, dataset_name, audio_preprocessing)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, 'dataset.zip')
    if not os.path.exists(dataset_path):
        print(f'Downloading dataset {audio_preprocessing}...')
        os.system(f'wget {download_link} -O {dataset_path} >/dev/null 2>&1')
        print(f'Dataset {audio_preprocessing} downloaded.')
        print(f'Extracting dataset {audio_preprocessing}...')
        os.system(f'unzip {dataset_path} -d {dataset_dir} >/dev/null 2>&1')
        print(f'Dataset {audio_preprocessing} extracted.')
    audio_train, audio_val, audio_test = torch.load(os.path.join(dataset_dir, 'train.pkl')), torch.load(os.path.join(dataset_dir, 'val.pkl')), torch.load(os.path.join(dataset_dir, 'test.pkl'))
    os.system(f'rm -rf {os.path.join(download_dir, dataset_name)} >/dev/null 2>&1')
    return audio_train, audio_val, audio_test


def get_audio_dataset(dataset_name, audio_preprocessing=None, taskname='acc', download_dir='./data'):
    if dataset_name.lower() not in SUPPORTED_DATASETS.keys():
        raise ValueError(f'Dataset {dataset_name} not supported. Supported datasets: {SUPPORTED_DATASETS.keys()}')
    
    if isinstance(audio_preprocessing, str):
        audio_train, audio_val, audio_test = load_precomputed_audio(dataset_name, audio_preprocessing, download_dir)
        labels_train, labels_val, labels_test = load_labels(dataset_name)

        train_dataset = MAMKitMonomodalDataset(audio_train, labels_train, taskname)
        val_dataset = MAMKitMonomodalDataset(audio_val, labels_val, taskname)
        test_dataset = MAMKitMonomodalDataset(audio_test, labels_test, taskname)
        return train_dataset, val_dataset, test_dataset
    elif callable(audio_preprocessing) or audio_preprocessing is None:
        raise NotImplementedError('Audio preprocessing is not implemented yet.')
    else:
        raise ValueError(f'Audio preprocessing must be a string or a callable. Received: {audio_preprocessing}')


def get_text_dataset(dataset_name, text_preprocessing=None, taskname='acc'):
    if dataset_name.lower() not in SUPPORTED_DATASETS.keys():
        raise ValueError(f'Dataset {dataset_name} not supported. Supported datasets: {SUPPORTED_DATASETS.keys()}')
    
    if not callable(text_preprocessing):
        raise ValueError(f'Text preprocessing must be a callable. Received: {text_preprocessing}')
    else:
        raw_text_train, raw_text_val, raw_text_test = load_text(dataset_name)
        text_train = list(map(text_preprocessing, raw_text_train))
        text_val = list(map(text_preprocessing, raw_text_val))
        text_test = list(map(text_preprocessing, raw_text_test))
        labels_train, labels_val, labels_test = load_labels(dataset_name)
        train_dataset = MAMKitMonomodalDataset(text_train, labels_train, taskname)
        val_dataset = MAMKitMonomodalDataset(text_val, labels_val, taskname)
        test_dataset = MAMKitMonomodalDataset(text_test, labels_test, taskname)
        return train_dataset, val_dataset, test_dataset
    


def get_multimodal_dataset(dataset_name, text_preprocessing=None, audio_preprocessing=None,  taskname='acc', download_dir='./data',):
    if dataset_name.lower() not in SUPPORTED_DATASETS.keys():
        raise ValueError(f'Dataset {dataset_name} not supported. Supported datasets: {SUPPORTED_DATASETS.keys()}')
    
    ### Text preprocessing
    if not callable(text_preprocessing):
        raise ValueError(f'Text preprocessing must be a callable. Received: {text_preprocessing}')
    else:
        raw_text_train, raw_text_val, raw_text_test = load_text(dataset_name)
        text_train = list(map(text_preprocessing, raw_text_train))
        text_val = list(map(text_preprocessing, raw_text_val))
        text_test = list(map(text_preprocessing, raw_text_test))

    ### Audio preprocessing
    if isinstance(audio_preprocessing, str):
        audio_train, audio_val, audio_test = load_precomputed_audio(dataset_name, audio_preprocessing, download_dir)
        
    elif callable(audio_preprocessing) or audio_preprocessing is None:
        # Load the raw version of the dataset
        # Optionally apply the audio preprocessing
        raise NotImplementedError('Audio preprocessing is not implemented yet.')
    else:
        raise ValueError(f'Audio preprocessing must be a string or a callable. Received: {audio_preprocessing}')

    ### Labels
    labels_train, labels_val, labels_test = load_labels(dataset_name)

    train = MAMKitPrecomputedDataset(text_train, audio_train, labels_train, taskname)
    val = MAMKitPrecomputedDataset(text_val, audio_val, labels_val, taskname)
    test = MAMKitPrecomputedDataset(text_test, audio_test, labels_test, taskname)

    return train, val, test