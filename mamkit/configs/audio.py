import torch as th

from mamkit.configs.base import BaseConfig, ConfigKey
from mamkit.data.datasets import InputMode


class BiLSTMMFCCsConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'mancini-et-al-2022'}): 'ukdebates_mancini_2022',
        ConfigKey(dataset='marg', input_mode=InputMode.AUDIO_ONLY, task_name='arc',
                  tags={'mancini-et-al-2022'}): 'marg_mancini_2022',
        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'mancini-et-al-2022'}): 'mmused_asd_mancini_2022',
        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='acd',
                  tags={'mancini-et-al-2022'}): 'mmused_acd_mancini_2022'
    }

    def __init__(
            self,
            mfccs,
            lstm_weights,
            mlp_weights,
            dropout_rate,
            num_classes,
            pooling_sizes=None,
            normalize=True,
            remove_energy=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.mfccs = mfccs
        self.lstm_weights = lstm_weights
        self.mlp_weights = mlp_weights
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.pooling_sizes = pooling_sizes
        self.normalize = normalize
        self.remove_energy = remove_energy
        self.embedding_dim = mfccs + 19

    @classmethod
    def ukdebates_mancini_2022(
            cls
    ):
        return cls(
            mlp_weights=[256],
            optimizer_args={
                'lr': 0.001,
                'weight_decay': 0.005
            },
            optimizer=th.optim.Adam,
            lstm_weights=[64, 32],
            dropout_rate=0.2,
            mfccs=25,
            pooling_sizes=[5, 5, 5],
            normalize=True,
            remove_energy=True,
            num_classes=2,
            seeds=[15371, 15372, 15373]
        )

    @classmethod
    def marg_mancini_2022(
            cls
    ):
        return cls(
            mlp_weights=[256],
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0001
            },
            optimizer=th.optim.Adam,
            lstm_weights=[128, 32],
            dropout_rate=0.3,
            mfccs=25,
            pooling_sizes=[10],
            normalize=True,
            remove_energy=True,
            num_classes=2,
            seeds=[15371, 15372, 15373]
        )

    @classmethod
    def mmused_asd_mancini_2022(
            cls
    ):
        return cls(
            mlp_weights=[64],
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.005
            },
            optimizer=th.optim.Adam,
            lstm_weights=[64, 32],
            dropout_rate=0.3,
            mfccs=25,
            pooling_sizes=[5],
            normalize=True,
            remove_energy=True,
            num_classes=2,
            seeds=[15371, 15372, 15373]
        )

    @classmethod
    def mmused_acd_mancini_2022(
            cls
    ):
        return cls(
            mlp_weights=[64],
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.0001
            },
            optimizer=th.optim.Adam,
            lstm_weights=[128, 32],
            dropout_rate=0.1,
            mfccs=25,
            pooling_sizes=[10],
            normalize=True,
            remove_energy=True,
            num_classes=2,
            seeds=[15371, 15372, 15373]
        )


class BiLSTMTransformerConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'mancini-et-al-2022'}): 'ukdebates_mancini_2022',
        ConfigKey(dataset='marg', input_mode=InputMode.AUDIO_ONLY, task_name='arc',
                  tags={'mancini-et-al-2022'}): 'marg_mancini_2022',
        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'mancini-et-al-2022'}): 'mmused_asd_mancini_2022',
        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='acd',
                  tags={'mancini-et-al-2022'}): 'mmused_acd_mancini_2022'
    }

    def __init__(
            self,
            model_card,
            sampling_rate,
            lstm_weights,
            mlp_weights,
            dropout_rate,
            num_classes,
            aggregate: bool = False,
            processor_args=None,
            model_args=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.model_card = model_card
        self.sampling_rate = sampling_rate
        self.lstm_weights = lstm_weights
        self.mlp_weights = mlp_weights
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.aggregate = aggregate
        self.processor_args = processor_args if processor_args is not None else {}
        self.model_args = model_args if model_args is not None else {}

    @classmethod
    def ukdebates_mancini_2022(
            cls
    ):
        return cls(
            model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            mlp_weights=[32, 32],
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.001
            },
            optimizer=th.optim.Adam,
            lstm_weights=[64, 32],
            dropout_rate=0.5,
            aggregate=True,
            num_classes=2,
            seeds=[15371, 15372, 15373]
        )

    @classmethod
    def marg_mancini_2022(
            cls
    ):
        return cls(
            model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            mlp_weights=[256],
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.001
            },
            optimizer=th.optim.Adam,
            lstm_weights=[64],
            dropout_rate=0.3,
            aggregate=True,
            num_classes=2,
            seeds=[15371, 15372, 15373]
        )

    @classmethod
    def mmused_asd_mancini_2022(
            cls
    ):
        return cls(
            model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            mlp_weights=[128],
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.0001
            },
            optimizer=th.optim.Adam,
            lstm_weights=[32, 32],
            dropout_rate=0.0,
            aggregate=True,
            num_classes=2,
            seeds=[15371, 15372, 15373]
        )

    @classmethod
    def mmused_acd_mancini_2022(
            cls
    ):
        return cls(
            model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            mlp_weights=[256],
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            optimizer=th.optim.Adam,
            lstm_weights=[128],
            dropout_rate=0.0,
            aggregate=True,
            num_classes=2,
            seeds=[15371, 15372, 15373]
        )
