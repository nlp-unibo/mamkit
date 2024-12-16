from typing import Type, List, Callable

import torch as th
from cinnamon.configuration import C
from cinnamon.registry import RegistrationKey, register_method
from torchmetrics.classification.f_beta import F1Score

from mamkit.components.text.model import BiLSTM, PairBiLSTM, Transformer, PairTransformer
from mamkit.configurations.model import MAMKitModelConfig
from mamkit.utility.metrics import ClassSubsetMulticlassF1Score

__all__ = [
    'BiLSTMConfig',
    'TransformerConfig'
]


class BiLSTMConfig(MAMKitModelConfig):

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
                     tags={'data:ukdebates', 'task:asd', 'mode:text-only', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def ukdebates_asd_mancini_2024_mamkit(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.processor = RegistrationKey(name='processor',
                                           tags={'data:ukdebates', 'task:asd', 'mode:text-only', 'vocab-builder', 'mancini-2024-mamkit'},
                                           namespace='mamkit')
        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684]))
        config.val_metrics = {'val_f1': F1Score(task='binary')}
        config.test_metrics = {'test_f1': F1Score(task='binary')}
        config.optimizer_args = {
            'lr': 0.0001,
            'weight_decay': 0.0005
        }
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
                     tags={'data:ukdebates', 'task:asd', 'mode:text-only', 'bilstm', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def ukdebates_asd_mancini_2022_argmining(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.processor = RegistrationKey(name='processor',
                                           tags={'data:ukdebates', 'task:asd', 'mode:text-only', 'vocab-builder',
                                                 'mancini-2022-argmining'},
                                           namespace='mamkit')
        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684]))
        config.val_metrics = {'val_f1': F1Score(task='binary')}
        config.test_metrics = {'test_f1': F1Score(task='binary')}
        config.optimizer_args = {
            'lr': 0.0001,
            'weight_decay': 0.0005
        }
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
                     tags={'data:mmused', 'task:asd', 'mode:text-only', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_asd_mancini_2024_mamkit(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.processor = RegistrationKey(name='processor',
                                           tags={'data:mmused', 'task:asd', 'mode:text-only', 'vocab-builder', 'mancini-2024-mamkit'},
                                           namespace='mamkit')
        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
            'lr': 0.0002,
            'weight_decay': 0.0001
        }
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
                     tags={'data:mmused', 'task:asd', 'mode:text-only', 'bilstm', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_asd_mancini_2022_argmining(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.processor = RegistrationKey(name='processor',
                                           tags={'data:mmused', 'task:asd', 'mode:text-only', 'vocab-builder', 'mancini-2022-argmining'},
                                           namespace='mamkit')
        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
            'lr': 0.0002,
            'weight_decay': 0.0001
        }
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
                     tags={'data:mmused', 'task:acc', 'mode:text-only', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_acc_mancini_2024_mamkit(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.processor = RegistrationKey(name='processor',
                                           tags={'data:mmused', 'task:acc', 'mode:text-only', 'vocab-builder', 'mancini-2024-mamkit'},
                                           namespace='mamkit')
        config.loss_function = lambda: th.nn.CrossEntropyLoss()
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
            'lr': 0.0002,
            'weight_decay': 0.0001
        }
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
                     tags={'data:mmused', 'task:acc', 'mode:text-only', 'bilstm', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_acc_mancini_2022_argmining(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.processor = RegistrationKey(name='processor',
                                           tags={'data:mmused', 'task:acc', 'mode:text-only', 'vocab-builder', 'mancini-2022-argmining'},
                                           namespace='mamkit')
        config.loss_function = lambda: th.nn.CrossEntropyLoss()
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
            'lr': 0.001,
            'weight_decay': 0.0005
        }
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
                     tags={'data:mmused-fallacy', 'task:afc', 'mode:text-only', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_fallacy_afc_mancini_2024_mamkit(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.processor = RegistrationKey(name='processor',
                                           tags={'data:mmused-fallacy', 'task:afc', 'mode:text-only', 'vocab-builder',
                                                 'mancini-2024-mamkit'},
                                           namespace='mamkit')
        config.loss_function = lambda: th.nn.CrossEntropyLoss(
            weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=6)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=6)}
        config.optimizer_args = {
            'lr': 0.0002,
            'weight_decay': 0.0001
        }
        config.embedding_dim = 200
        config.lstm_weights = [128, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(64, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 6)
        )
        config.dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mode:text-only', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=PairBiLSTM)
    def marg_arc_mancini_2024_mamkit(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.processor = RegistrationKey(name='processor',
                                           tags={'data:marg', 'task:arc', 'mode:text-only', 'vocab-builder', 'mancini-2024-mamkit'},
                                           namespace='mamkit')
        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977]))
        config.val_metrics = {
            'val_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.test_metrics = {
            'test_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.optimizer_args = {
            'lr': 0.0002,
            'weight_decay': 0.0001
        }
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
                     tags={'data:marg', 'task:arc', 'mode:text-only', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=PairBiLSTM)
    def marg_arc_mancini_2022(
            cls: Type[C]
    ) -> C:
        config = cls.default()

        config.processor = RegistrationKey(name='processor',
                                           tags={'data:mmused-fallacy', 'task:afc', 'mode:text-only', 'vocab-builder',
                                                 'mancini-2024-mamkit'},
                                           namespace='mamkit')
        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977]))
        config.val_metrics = {
            'val_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.test_metrics = {
            'test_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.optimizer_args = {
            'lr': 0.0002,
            'weight_decay': 0.0001
        }
        config.embedding_dim = 100
        config.lstm_weights = [128]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(256, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 2)
        )
        config.dropout_rate = 0.4

        return config


# TODO: add processor and collator
class TransformerConfig(MAMKitModelConfig):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='model_card',
                   type_hint=str,
                   is_required=True,
                   description='Transformer model card from HuggingFace.',
                   variants=['bert-base-uncased', 'roberta-base'])
        config.add(name='head',
                   type_hint=Callable[[], th.nn.Module],
                   description='Classification head',
                   is_required=True)
        config.add(name='dropout_rate',
                   type_hint=float,
                   description='Dropout rate',
                   is_required=True)
        config.add(name='is_transformer_trainable',
                   type_hint=bool,
                   value=False,
                   description='Whether the transformer is fully trainable or not.')

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mode:text-only', 'transformer', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=Transformer)
    def ukdebates_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684]))
        config.val_metrics = {'val_f1': F1Score(task='binary')}
        config.test_metrics = {'test_f1': F1Score(task='binary')}
        config.optimizer_args = {'lr': 1e-03, 'weight_decay': 1e-05}
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.dropout_rate = 0.0
        config.is_transformer_trainable = False

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mode:text-only', 'transformer', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=Transformer)
    def ukdebates_asd_mancini_2022(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684]))
        config.val_metrics = {'val_f1': F1Score(task='binary')}
        config.test_metrics = {'test_f1': F1Score(task='binary')}
        config.optimizer_args = {}
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.dropout_rate = 0.0
        config.is_transformer_trainable = True

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mode:text-only', 'transformer', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=Transformer)
    def mmused_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {'lr': 1e-03, 'weight_decay': 1e-05}
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.dropout_rate = 0.2
        config.is_transformer_trainable = False

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mode:text-only', 'transformer', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=Transformer)
    def mmused_acc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss()
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {'lr': 1e-03, 'weight_decay': 1e-05}
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.dropout_rate = 0.2
        config.is_transformer_trainable = False

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused-fallacy', 'task:afc', 'mode:text-only', 'transformer', 'mancini-2024-eacl'},
                     namespace='mamkit',
                     component_class=Transformer)
    def mmused_fallacy_afc_mancini_2024_eacl(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(
            weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=6)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=6)}
        config.optimizer_args = {}
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 100),
            th.nn.ReLU(),
            th.nn.Linear(100, 50),
            th.nn.ReLU(),
            th.nn.Linear(50, 6)
        ),
        config.dropout_rate = 0.1
        config.is_transformer_trainable = True

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused-fallacy', 'task:afc', 'mode:text-only', 'transformer', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=Transformer)
    def mmused_fallacy_afc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(
            weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=6)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=6)}
        config.optimizer_args = {'lr': 1e-03, 'weight_decay': 1e-05}
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 6)
        ),
        config.dropout_rate = 0.2
        config.is_transformer_trainable = False

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mode:text-only', 'transformer', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=PairTransformer)
    def marg_arc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977]))
        config.val_metrics = {
            'val_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.test_metrics = {
            'test_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.optimizer_args = {'lr': 1e-03, 'weight_decay': 1e-05}
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768 * 2, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 3)
        ),
        config.dropout_rate = 0.2
        config.is_transformer_trainable = False

        return config
