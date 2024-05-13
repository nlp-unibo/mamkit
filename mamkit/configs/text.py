import torch as th
from torchtext.data.utils import get_tokenizer

from mamkit.configs.base import BaseConfig, ConfigKey
from mamkit.data.datasets import InputMode


class BiLSTMConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_ONLY, task_name='asd',
                  tags={'mancini-et-al-2022'}): 'ukdebates_mancini_2022',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_ONLY, task_name='arc',
                  tags={'mancini-et-al-2022'}): 'marg_mancini_2022',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_ONLY, task_name='asd',
                  tags={'mancini-et-al-2022'}): 'mmused_asd_mancini_2022',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_ONLY, task_name='acd',
                  tags={'mancini-et-al-2022'}): 'mmused_acd_mancini_2022'
    }

    def __init__(
            self,
            embedding_dim,
            lstm_weights,
            mlp_weights,
            dropout_rate,
            num_classes,
            tokenizer,
            tokenization_args=None,
            embedding_model=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.lstm_weights = lstm_weights
        self.mlp_weights = mlp_weights
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.tokenization_args = tokenization_args

    @classmethod
    def ukdebates_mancini_2022(
            cls
    ):
        return cls(
            mlp_weights=[128],
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            optimizer=th.optim.Adam,
            lstm_weights=[128, 32],
            dropout_rate=0.5,
            embedding_dim=200,
            embedding_model='glove.6B.200d',
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            num_classes=2,
            seeds=[15371, 15372, 15373]
        )

    @classmethod
    def marg_mancini_2022(
            cls
    ):
        return cls(
            mlp_weights=[64],
            dropout_rate=0.4,
            embedding_dim=100,
            embedding_model='glove.6B.100d',
            lstm_weights=[128],
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.0001
            },
            num_classes=2,
            seeds=[15371, 15372, 15373]
        )

    @classmethod
    def mmused_asd_mancini_2022(
            cls
    ):
        return cls(
            mlp_weights=[256],
            dropout_rate=0.0,
            embedding_dim=100,
            embedding_model='glove.6B.100d',
            lstm_weights=[64, 64],
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            optimizer=th.optim.Adam,
            optimizer_args={
                'l2': 0.0002,
                'weight_decay': 0.0001
            },
            num_classes=2,
            seeds=[15371, 15372, 15373]
        )

    @classmethod
    def mmused_acd_mancini_2022(
            cls
    ):
        return cls(
            mlp_weights=[64],
            dropout_rate=0.3,
            embedding_dim=100,
            embedding_model='glove.6B.100d',
            lstm_weights=[64, 32],
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            optimizer=th.optim.Adam,
            optimizer_args={
                'l2': 0.001,
                'weight_decay': 0.0005
            },
            num_classes=2,
            seeds=[15371, 15372, 15373]
        )


class TransformerConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY, task_name='afc',
                  tags={'mancini-et-al-2024', 'bert'}): 'mmused_fallacy_afc_bert_mancini_2024',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY, task_name='afc',
                  tags={'mancini-et-al-2024', 'roberta'}): 'mmused_fallacy_afc_roberta_mancini_2024'
    }

    def __init__(
            self,
            model_card,
            mlp_weights,
            num_classes,
            dropout_rate=0.0,
            is_transformer_trainable: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.model_card = model_card
        self.mlp_weights = mlp_weights
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.is_transformer_trainable = is_transformer_trainable

    @classmethod
    def mmused_fallacy_afc_bert_mancini_2024(
            cls
    ):
        return cls(
            model_card='bert-base-uncased',
            mlp_weights=[100, 50],
            num_classes=2,
            dropout_rate=0.1,
            is_transformer_trainable=True
        )

    @classmethod
    def mmused_fallacy_afc_roberta_mancini_2024(
            cls
    ):
        return cls(
            model_card='roberta-base',
            mlp_weights=[100, 50],
            num_classes=2,
            dropout_rate=0.1,
            is_transformer_trainable=True
        )