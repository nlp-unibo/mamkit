from typing import Type, List, Callable

import torch as th
from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method

from mamkit.components.modeling.audio import BiLSTM, PairBiLSTM


class BiLSTMMFCCConfig(Configuration):

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
                     tags={'data:ukdebates', 'task:asd', 'bilstm', 'mfcc', 'mancini-2022-argmining'},
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
                     tags={'data:ukdebates', 'task:asd', 'bilstm', 'mfcc', 'mancini-2024-mamkit'},
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
                     tags={'data:marg', 'task:arc', 'bilstm', 'mfcc', 'mancini-2022-argmining'},
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
                     tags={'data:marg', 'task:arc', 'bilstm', 'mfcc', 'mancini-2024-mamkit'},
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
                     tags={'data:mmused', 'task:asd', 'bilstm', 'mfcc', 'mancini-2022-argmining'},
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
                     tags={'data:mmused', 'task:asd', 'bilstm', 'mfcc', 'mancini-2024-mamkit'},
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
                     tags={'data:mmused', 'task:acc', 'bilstm', 'mfcc', 'mancini-2022-argmining'},
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
                     tags={'data:mmused', 'task:acc', 'bilstm', 'mfcc', 'mancini-2024-mamkit'},
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
                     tags={'data:mmused-fallacy', 'task:afc', 'bilstm', 'mfcc', 'mancini-2024-mamkit'},
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
