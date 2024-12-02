from typing import Type, List, Callable

import torch as th
from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method

from mamkit.components.modeling.text import BiLSTM


class BiLSTMConfig(Configuration):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='embedding_dim',
                   type_hint=int,
                   description='Embedding size',
                   is_required=True)
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
                     tags={'data:ukdebates', 'task:asd', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def ukdebates_asd_mancini_2024(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.embedding_dim = 200
        config.lstm_weights = [128, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def ukdebates_asd_mancini_2022(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.embedding_dim = 200
        config.lstm_weights = [128, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.dropout_rate = 0.5

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def marg_arc_mancini_2022(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.embedding_dim = 100
        config.lstm_weights = [128]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(256, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 2)
        )
        config.dropout_rate = 0.4

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def marg_arc_mancini_2024(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.embedding_dim = 200
        config.lstm_weights = [128, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 3)
        )
        config.dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_asd_mancini_2022(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.embedding_dim = 100
        config.lstm_weights = [64, 64]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(128, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_asd_mancini_2024(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.embedding_dim = 200
        config.lstm_weights = [128, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_acc_mancini_2022(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.embedding_dim = 100
        config.lstm_weights = [64, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 2)
        )
        config.dropout_rate = 0.3

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_acc_mancini_2024(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.embedding_dim = 200
        config.lstm_weights = [128, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused-fallacy', 'task:afc', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_fallacy_afc_mancini_2024(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.embedding_dim = 200
        config.lstm_weights = [128, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 6)
        )
        config.dropout_rate = 0.0

        return config
