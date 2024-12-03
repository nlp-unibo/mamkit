from typing import Type, List, Callable

import torch as th
from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method

from mamkit.components.modeling.audio import BiLSTM, PairBiLSTM, Transformer, PairTransformer


class BiLSTMConfig(Configuration):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='lstm_weights',
                   type_hint=List[int],
                   description='LSTM stack units',
                   is_required=True)
        config.add(name='head',
                   type_hint=Callable[[], th.nn.Module],
                   description='Classification head',
                   is_required=True)
        config.add(name='dropout_rate',
                   type_hint=float,
                   description='Dropout rate',
                   is_required=True)

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mode:audio-only', 'bilstm', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def ukdebates_asd_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.lstm_weights = [64, 32]
        config.dropout_rate = 0.2

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mode:audio-only', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def ukdebates_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.lstm_weights = [64, 32]
        config.dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mode:audio-only', 'bilstm', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=PairBiLSTM)
    def marg_arc_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.lstm_weights = [128, 32]
        config.dropout_rate = 0.3

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mode:audio-only', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=PairBiLSTM)
    def marg_arc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 3)
        )
        config.lstm_weights = [64, 32]
        config.dropout_rate = 0.1

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mode:audio-only', 'bilstm', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_asd_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 2)
        )
        config.lstm_weights = [64, 32]
        config.dropout_rate = 0.1

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mode:audio-only', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.lstm_weights = [64, 32]
        config.dropout_rate = 0.1

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mode:audio-only', 'bilstm', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_acc_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 2)
        )
        config.lstm_weights = [128, 32]
        config.dropout_rate = 0.1

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mode:audio-only', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_acc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.lstm_weights = [64, 32]
        config.dropout_rate = 0.1

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused-fallacy', 'task:afc', 'mode:audio-only', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_acc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 6)
        )
        config.lstm_weights = [64, 32]
        config.dropout_rate = 0.1

        return config


class TransformerConfig(Configuration):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='encoder',
                   type_hint=Callable[[], th.nn.Module],
                   description='Encoder module',
                   is_required=True)
        config.add(name='head',
                   type_hint=Callable[[], th.nn.Module],
                   description='Classification head',
                   is_required=True)
        config.add(name='dropout_rate',
                   type_hint=float,
                   description='Dropout rate',
                   is_required=True)

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mode:audio-only', 'transformer', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=Transformer)
    def ukdebates_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.encoder = lambda: th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
            num_layers=1
        )
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mode:audio-only', 'transformer', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=Transformer)
    def mmused_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.encoder = lambda: th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
            num_layers=1
        )
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.dropout_rate = 0.2

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mode:audio-only', 'transformer', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=Transformer)
    def mmused_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.encoder = lambda: th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
            num_layers=1
        )
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.dropout_rate = 0.2

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mode:audio-only', 'transformer', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=PairTransformer)
    def mmused_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.encoder = lambda: th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
            num_layers=1
        )
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.dropout_rate = 0.2

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused-fallacy', 'task:afc', 'mode:audio-only', 'transformer', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=PairTransformer)
    def mmused_fallacy_afc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.encoder = lambda: th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
            num_layers=1
        )
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 6)
        )
        config.dropout_rate = 0.2

        return config
