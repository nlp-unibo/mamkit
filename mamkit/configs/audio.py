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
                  tags={'anonymous'}): 'ukdebates_asd_anonymous',
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
            embedding_dim,
            sampling_rate,
            lstm_weights,
            head: th.nn.Module,
            dropout_rate,
            num_classes,
            aggregate: bool = False,
            downsampling_factor=None,
            processor_args=None,
            model_args=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.model_card = model_card
        self.embedding_dim = embedding_dim
        self.sampling_rate = sampling_rate
        self.lstm_weights = lstm_weights
        self.head = head
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.aggregate = aggregate
        self.downsampling_factor = downsampling_factor
        self.processor_args = processor_args if processor_args is not None else {}
        self.model_args = model_args if model_args is not None else {}

    @classmethod
    def ukdebates_asd_anonymous(
            cls
    ):
        return cls(
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            sampling_rate=16000,
            head=th.nn.Sequential(
                th.nn.Linear(64, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.001
            },
            optimizer=th.optim.Adam,
            lstm_weights=[64, 32],
            dropout_rate=0.1,
            aggregate=False,
            downsampling_factor=None,
            num_classes=2,
            seeds=[42, 2024, 666, 11, 1492],
            batch_size=8,
            loss_function=th.nn.CrossEntropyLoss()
        )

    @classmethod
    def ukdebates_mancini_2022(
            cls
    ):
        return cls(
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            sampling_rate=16000,
            head=th.nn.Sequential(
                th.nn.Linear(64, 32),
                th.nn.ReLU(),
                th.nn.Linear(32, 32),
                th.nn.ReLU(),
                th.nn.Linear(32, 2)
            ),
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
            embedding_dim=768,
            sampling_rate=16000,
            head=th.nn.Sequential(
                th.nn.Linear(128, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2),
            ),
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
            embedding_dim=768,
            sampling_rate=16000,
            head=th.nn.Sequential(
                th.nn.Linear(64, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2),
            ),
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
            embedding_dim=768,
            sampling_rate=16000,
            head=th.nn.Sequential(
                th.nn.Linear(256, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2),
            ),
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


class TransformerEncoderConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous'}): 'ukdebates_asd_anonymous',
    }

    def __init__(
            self,
            model_card,
            embedding_dim,
            encoder: th.nn.Module,
            head: th.nn.Module,
            num_classes,
            dropout_rate=0.0,
            processor_args=None,
            model_args=None,
            downsampling_factor=None,
            aggregate=False,
            sampling_rate=16000,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.model_card = model_card
        self.embedding_dim = embedding_dim
        self.encoder = encoder
        self.head = head
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.processor_args = processor_args
        self.model_args = model_args
        self.aggregate = aggregate
        self.downsampling_factor = downsampling_factor
        self.sampling_rate = sampling_rate

    @classmethod
    def ukdebates_asd_anonymous(
            cls
    ):
        return cls(
            head=th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            encoder=th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=2,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=None,
            sampling_rate=16000,
            batch_size=8,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.0,
            loss_function=th.nn.CrossEntropyLoss(),
            seeds=[42, 2024, 666, 11, 1492],
        )
