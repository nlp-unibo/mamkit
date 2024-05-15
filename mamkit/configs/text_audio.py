import torch as th
from torchtext.data.utils import get_tokenizer

from mamkit.configs.base import BaseConfig, ConfigKey
from mamkit.data.datasets import InputMode


class BiLSTMConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous'}): 'ukdebates_anonymous',
    }

    def __init__(
            self,
            text_embedding_dim,
            text_lstm_weights,
            audio_embedding_dim,
            audio_lstm_weights,
            head: th.nn.Module,
            num_classes,
            tokenizer,
            audio_model_card,
            sampling_rate,
            downsampling_factor=None,
            audio_model_args=None,
            aggregate=False,
            processor_args=None,
            tokenization_args=None,
            embedding_model=None,
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.text_embedding_dim = text_embedding_dim
        self.embedding_model = embedding_model
        self.text_lstm_weights = text_lstm_weights
        self.audio_embedding_dim = audio_embedding_dim
        self.audio_lstm_weights = audio_lstm_weights
        self.head = head
        self.text_dropout_rate = text_dropout_rate
        self.audio_dropout_rate = audio_dropout_rate
        self.num_classes = num_classes
        self.audio_model_card = audio_model_card
        self.sampling_rate = sampling_rate
        self.downsampling_factor = downsampling_factor
        self.aggregate = aggregate
        self.audio_model_args = audio_model_args
        self.processor_args = processor_args
        self.tokenizer = tokenizer
        self.tokenization_args = tokenization_args

    @classmethod
    def ukdebates_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.0,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42, 2024, 666, 11, 1492],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=th.nn.CrossEntropyLoss(),
            batch_size=8,
            num_classes=2,
        )