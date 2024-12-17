from typing import Type, List, Callable

import torch as th
from cinnamon.configuration import C
from cinnamon.registry import register_method

from mamkit.components.text_audio.model import (
    BiLSTM,
    PairBiLSTM,
    MMTransformer,
    PairMMTransformer,
    CSA,
    PairCSA,
    Ensemble,
    PairEnsemble,
    MulTA,
    PairMulTA
)
from mamkit.configurations.model import MAMKitModelConfig
from mamkit.modules.transformer import PositionalEncoding, CustomEncoder
from mamkit.utility.metrics import ClassSubsetMulticlassF1Score
from torchmetrics.classification.f_beta import F1Score

__all__ = [
    'BiLSTMConfig',
    'MMTransformerConfig',
    'CSAConfig',
    'EnsembleConfig',
    'MulTAConfig'
]


class BiLSTMConfig(MAMKitModelConfig):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='text_lstm_weights',
                   type_hint=List[int],
                   description='LSTM stack units for text module.',
                   is_required=True)
        config.add(name='audio_lstm_weights',
                   type_hint=List[int],
                   description='LSTM stack units for audio module.',
                   is_required=True)
        config.add(name='head',
                   type_hint=Callable[[], th.nn.Module],
                   description='Classification head',
                   is_required=True)
        config.add(name='text_dropout_rate',
                   type_hint=float,
                   description='Dropout rate for text module',
                   is_required=True)
        config.add(name='audio_dropout_rate',
                   type_hint=float,
                   description='Dropout rate for audio module',
                   is_required=True)

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mode:text-audio', 'bilstm', 'mfcc', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def ukdebates_asd_mancini_2024_mamkit_mfcc(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684]))
        config.val_metrics = {'val_f1': F1Score(task='binary')}
        config.test_metrics = {'test_f1': F1Score(task='binary')}
        config.optimizer_args = {
            'lr': 0.0001,
            'weight_decay': 0.0005
        }
        config.text_lstm_weights = [128, 32]
        config.audio_lstm_weights = [64, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.text_dropout_rate = 0.0
        config.audio_dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mode:text-audio', 'bilstm', 'transformer',
                           'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def ukdebates_asd_mancini_2024_mamkit_transformer(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684]))
        config.val_metrics = {'val_f1': F1Score(task='binary')}
        config.test_metrics = {'test_f1': F1Score(task='binary')}
        config.optimizer_args = {
            'lr': 0.0001,
            'weight_decay': 0.0005
        }
        config.text_lstm_weights = [128, 32]
        config.audio_lstm_weights = [64, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.text_dropout_rate = 0.0
        config.audio_dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mode:text-audio', 'bilstm', 'mfcc', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_asd_mancini_2024_mamkit_mfcc(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
            'lr': 0.0002,
            'weight_decay': 0.001
        }
        config.text_lstm_weights = [128, 32]
        config.audio_lstm_weights = [64, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.text_dropout_rate = 0.0
        config.audio_dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mode:text-audio', 'bilstm', 'transformer',
                           'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_asd_mancini_2024_mamkit_transformer(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
            'lr': 0.0001,
            'weight_decay': 0.0005
        }
        config.text_lstm_weights = [128, 32]
        config.audio_lstm_weights = [64, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.text_dropout_rate = 0.0
        config.audio_dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mode:text-audio', 'bilstm', 'mfcc', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_acc_mancini_2024_mamkit_mfcc(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss()
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
            'lr': 0.0002,
            'weight_decay': 0.001
        }
        config.text_lstm_weights = [128, 32]
        config.audio_lstm_weights = [64, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.text_dropout_rate = 0.0
        config.audio_dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mode:text-audio', 'bilstm', 'transformer',
                           'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_acc_mancini_2024_mamkit_transformer(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss()
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
            'lr': 0.0001,
            'weight_decay': 0.0005
        }
        config.text_lstm_weights = [128, 32]
        config.audio_lstm_weights = [64, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.text_dropout_rate = 0.0
        config.audio_dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused-fallacy', 'task:afc', 'mode:text-audio', 'bilstm', 'mfcc',
                           'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_fallacy_afc_mancini_2024_mamkit_mfcc(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(
            weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=6)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=6)}
        config.optimizer_args = {
            'lr': 0.0002,
            'weight_decay': 0.001
        }
        config.text_lstm_weights = [128, 32]
        config.audio_lstm_weights = [64, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.text_dropout_rate = 0.0
        config.audio_dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused-fallacy', 'task:afc', 'mode:text-audio', 'bilstm', 'transformer',
                           'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=BiLSTM)
    def mmused_fallacy_afc_mancini_2024_mamkit_transformer(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(
            weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=6)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=6)}
        config.optimizer_args = {
            'lr': 0.0001,
            'weight_decay': 0.0005
        }
        config.text_lstm_weights = [128, 32]
        config.audio_lstm_weights = [64, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.text_dropout_rate = 0.0
        config.audio_dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mode:text-audio', 'bilstm', 'mfcc', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=PairBiLSTM)
    def marg_arc_mancini_2024_mamkit_mfcc(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977]))
        config.val_metrics = {
            'val_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.test_metrics = {
            'test_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.optimizer_args = {
            'lr': 0.0002,
            'weight_decay': 0.001
        }
        config.text_lstm_weights = [128, 32]
        config.audio_lstm_weights = [64, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(256, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 3)
        )
        config.text_dropout_rate = 0.0
        config.audio_dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mode:text-audio', 'bilstm', 'transformer', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=PairBiLSTM)
    def marg_arc_mancini_2024_mamkit_transformer(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977]))
        config.val_metrics = {
            'val_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.test_metrics = {
            'test_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.optimizer_args = {
            'lr': 0.0001,
            'weight_decay': 0.0005
        }
        config.text_lstm_weights = [128, 32]
        config.audio_lstm_weights = [64, 32]
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(256, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 3)
        )
        config.text_dropout_rate = 0.0
        config.audio_dropout_rate = 0.0

        return config


class MMTransformerConfig(MAMKitModelConfig):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='model_card',
                   type_hint=str,
                   is_required=True,
                   description="Huggingface model card for transformer model.",
                   variants=['bert-base-uncased', 'roberta-base'])
        config.add(name='head',
                   type_hint=Callable[[], th.nn.Module],
                   description='Classification head',
                   is_required=True)
        config.add(name='text_dropout_rate',
                   type_hint=float,
                   description='Dropout rate for text module',
                   is_required=True)
        config.add(name='audio_dropout_rate',
                   type_hint=float,
                   description='Dropout rate for audio module',
                   is_required=True)
        config.add(name='is_transformer_trainable',
                   type_hint=bool,
                   value=False,
                   description='Whether the transformer is fully trainable or not.')
        config.add(name='lstm_weights',
                   type_hint=List[int],
                   description='LSTM stack units',
                   is_required=True)

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mode:text-audio', 'transformer', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MMTransformer)
    def ukdebates_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684]))
        config.val_metrics = {'val_f1': F1Score(task='binary')}
        config.test_metrics = {'test_f1': F1Score(task='binary')}
        config.optimizer_args = {
                'lr': 1e-03,
                'weight_decay': 0.0005
            }
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(832, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.lstm_weights = [64, 32]
        config.text_dropout_rate = 0.0
        config.audio_dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mode:text-audio', 'transformer', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MMTransformer)
    def mmused_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
                'lr': 1e-03,
                'weight_decay': 0.0005
            }
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(832, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.lstm_weights = [64, 32]
        config.text_dropout_rate = 0.2
        config.audio_dropout_rate = 0.2

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mode:text-audio', 'transformer', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MMTransformer)
    def mmused_acc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss()
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
            'lr': 1e-03,
            'weight_decay': 0.0005
        }
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(832, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 2)
        )
        config.lstm_weights = [64, 32]
        config.text_dropout_rate = 0.2
        config.audio_dropout_rate = 0.2

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused-fallacy', 'task:afc', 'mode:text-audio', 'transformer', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MMTransformer)
    def mmused_fallacy_afc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(
            weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=6)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=6)}
        config.optimizer_args = {
            'lr': 1e-03,
            'weight_decay': 0.0005
        }
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(832, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 6)
        )
        config.lstm_weights = [64, 32]
        config.text_dropout_rate = 0.2
        config.audio_dropout_rate = 0.2

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mode:text-audio', 'transformer', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=PairMMTransformer)
    def marg_arc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977]))
        config.val_metrics = {
            'val_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.test_metrics = {
            'test_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.optimizer_args = {
            'lr': 1e-03,
            'weight_decay': 0.0005
        }
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(832 * 2, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 3)
        )
        config.lstm_weights = [64, 32]
        config.text_dropout_rate = 0.2
        config.audio_dropout_rate = 0.2

        return config


class CSAConfig(MAMKitModelConfig):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='transformer',
                   is_required=True,
                   type_hint=Callable[[], th.nn.Module],
                   description='Transformer module used to encode multi-modalities.'
                   )
        config.add(name='head',
                   type_hint=Callable[[], th.nn.Module],
                   description='Classification head',
                   is_required=True)
        config.add(name='positional_encoder',
                   type_hint=Callable[[], th.nn.Module],
                   description='Positional encoding module.',
                   is_required=True)
        config.add(name='text_dropout_rate',
                   type_hint=float,
                   description='Dropout rate for text module',
                   is_required=True)
        config.add(name='audio_dropout_rate',
                   type_hint=float,
                   description='Dropout rate for audio module',
                   is_required=True)

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mode:text-audio', 'transformer', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=CSA)
    def ukdebates_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684]))
        config.val_metrics = {'val_f1': F1Score(task='binary')}
        config.test_metrics = {'test_f1': F1Score(task='binary')}
        config.optimizer_args = {
                'lr': 1e-04,
                'weight_decay': 1e-03
            }
        config.transformer = lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1)
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.positional_encoder = lambda: PositionalEncoding(768, dual_modality=False)
        config.text_dropout_rate = 0.0
        config.audio_dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mode:text-audio', 'transformer', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=CSA)
    def mmused_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
                'lr': 1e-04,
                'weight_decay': 1e-03
            }
        config.transformer = lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1)
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.positional_encoder = lambda: PositionalEncoding(768, dual_modality=False)
        config.text_dropout_rate = 0.1
        config.audio_dropout_rate = 0.1

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mode:text-audio', 'transformer', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=CSA)
    def mmused_acc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss()
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
            'lr': 1e-04,
            'weight_decay': 1e-03
        }
        config.transformer = lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1)
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.positional_encoder = lambda: PositionalEncoding(768, dual_modality=False)
        config.text_dropout_rate = 0.1
        config.audio_dropout_rate = 0.1

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused-fallacy', 'task:afc', 'mode:text-audio', 'transformer', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=CSA)
    def mmused_fallacy_afc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(
            weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=6)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=6)}
        config.optimizer_args = {
            'lr': 1e-04,
            'weight_decay': 1e-03
        }
        config.transformer = lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1)
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 6)
        )
        config.positional_encoder = lambda: PositionalEncoding(768, dual_modality=False)
        config.text_dropout_rate = 0.1
        config.audio_dropout_rate = 0.1

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:margs', 'task:arc', 'mode:text-audio', 'transformer', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=PairCSA)
    def marg_arc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977]))
        config.val_metrics = {
            'val_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.test_metrics = {
            'test_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.optimizer_args = {
            'lr': 1e-04,
            'weight_decay': 1e-03
        }
        config.transformer = lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1)
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768 * 2, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 3)
        )
        config.positional_encoder = lambda: PositionalEncoding(768, dual_modality=False)
        config.text_dropout_rate = 0.1
        config.audio_dropout_rate = 0.1

        return config


class EnsembleConfig(MAMKitModelConfig):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='audio_encoder',
                   type_hint=Callable[[], th.nn.Module],
                   description='Encoder for audio modality',
                   is_required=True)
        config.add(name='text_head',
                   type_hint=Callable[[], th.nn.Module],
                   description='Classification head for text modality',
                   is_required=True)
        config.add(name='audio_head',
                   type_hint=Callable[[], th.nn.Module],
                   description='Classification head for audio modality',
                   is_required=True)
        config.add(name='positional_encoder',
                   type_hint=Callable[[], th.nn.Module],
                   description='Positional encoding module.',
                   is_required=True)
        config.add(name='text_dropout_rate',
                   type_hint=float,
                   description='Dropout rate for text module',
                   is_required=True)
        config.add(name='audio_dropout_rate',
                   type_hint=float,
                   description='Dropout rate for audio module',
                   is_required=True)
        config.add(name='lower_bound',
                   type_hint=float,
                   description='Lower bound coefficient',
                   is_required=True)
        config.add(name='upper_bound',
                   type_hint=float,
                   description='Upper bound coefficient',
                   is_required=True)

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mode:text-audio', 'ensemble', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=Ensemble)
    def ukdebates_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684]))
        config.val_metrics = {'val_f1': F1Score(task='binary')}
        config.test_metrics = {'test_f1': F1Score(task='binary')}
        config.optimizer_args = {
            'lr': 1e-04,
            'weight_decay': 1e-03
        }
        config.audio_encoder = lambda: th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
            num_layers=1
        )
        config.text_head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.audio_head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.positional_encoder = lambda: PositionalEncoding(d_model=768, dual_modality=False)
        config.text_dropout_rate = 0.0
        config.audio_dropout_rate = 0.0
        config.lower_bound = 0.3
        config.upper_bound = 0.7

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mode:text-audio', 'ensemble', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=Ensemble)
    def mmused_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
            'lr': 1e-04,
            'weight_decay': 1e-03
        }
        config.audio_encoder = lambda: th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
            num_layers=1
        )
        config.text_head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.audio_head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.positional_encoder = lambda: PositionalEncoding(d_model=768, dual_modality=False)
        config.text_dropout_rate = 0.1
        config.audio_dropout_rate = 0.1
        config.lower_bound = 0.3
        config.upper_bound = 0.7

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mode:text-audio', 'ensemble', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=Ensemble)
    def mmused_acc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss()
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
            'lr': 1e-04,
            'weight_decay': 1e-03
        }
        config.audio_encoder = lambda: th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
            num_layers=1
        )
        config.text_head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.audio_head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.positional_encoder = lambda: PositionalEncoding(d_model=768, dual_modality=False)
        config.text_dropout_rate = 0.1
        config.audio_dropout_rate = 0.1
        config.lower_bound = 0.3
        config.upper_bound = 0.7

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused-fallacy', 'task:afc', 'mode:text-audio', 'ensemble', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=Ensemble)
    def mmused_fallacy_afc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(
            weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=6)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=6)}
        config.optimizer_args = {
            'lr': 1e-04,
            'weight_decay': 1e-03
        }
        config.audio_encoder = lambda: th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
            num_layers=1
        )
        config.text_head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 6)
        )
        config.audio_head = lambda: th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 6)
        )
        config.positional_encoder = lambda: PositionalEncoding(d_model=768, dual_modality=False)
        config.text_dropout_rate = 0.1
        config.audio_dropout_rate = 0.1
        config.lower_bound = 0.3
        config.upper_bound = 0.7

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mode:text-audio', 'ensemble', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=PairEnsemble)
    def marg_arc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977]))
        config.val_metrics = {
            'val_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.test_metrics = {
            'test_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.optimizer_args = {
            'lr': 1e-04,
            'weight_decay': 1e-03
        }
        config.audio_encoder = lambda: th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
            num_layers=1
        )
        config.text_head = lambda: th.nn.Sequential(
            th.nn.Linear(768 * 2, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 3)
        )
        config.audio_head = lambda: th.nn.Sequential(
            th.nn.Linear(768 * 2, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 3)
        )
        config.positional_encoder = lambda: PositionalEncoding(d_model=768, dual_modality=False)
        config.text_dropout_rate = 0.1
        config.audio_dropout_rate = 0.1
        config.lower_bound = 0.3
        config.upper_bound = 0.7

        return config


class MulTAConfig(MAMKitModelConfig):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='d_fnn',
                   type_hint=int,
                   is_required=True,
                   description='Cross-attention units')
        config.add(name='n_blocks',
                   type_hint=int,
                   is_required=True,
                   description='Number of cross-attention blocks')
        config.add(name='head',
                   type_hint=Callable[[], th.nn.Module],
                   description='Classification head',
                   is_required=True)
        config.add(name='positional_encoder',
                   type_hint=Callable[[], th.nn.Module],
                   description='Positional encoding module.',
                   is_required=True)
        config.add(name='text_dropout_rate',
                   type_hint=float,
                   description='Dropout rate for text module',
                   is_required=True)
        config.add(name='audio_dropout_rate',
                   type_hint=float,
                   description='Dropout rate for audio module',
                   is_required=True)

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mode:text-audio', 'multa', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MulTA)
    def ukdebates_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684]))
        config.val_metrics = {'val_f1': F1Score(task='binary')}
        config.test_metrics = {'test_f1': F1Score(task='binary')}
        config.optimizer_args = {
            'lr': 1e-04,
            'weight_decay': 1e-03
        }
        config.d_ffn = 2048
        config.n_blocks = 4
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768 * 2, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.positional_encoder = lambda: PositionalEncoding(d_model=768, dual_modality=False)
        config.text_dropout_rate = 0.0
        config.audio_dropout_rate = 0.0

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mode:text-audio', 'multa', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MulTA)
    def mmused_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
            'lr': 1e-04,
            'weight_decay': 1e-03
        }
        config.d_ffn = 2048
        config.n_blocks = 4
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768 * 2, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.positional_encoder = lambda: PositionalEncoding(d_model=768, dual_modality=False)
        config.text_dropout_rate = 0.1
        config.audio_dropout_rate = 0.1

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mode:text-audio', 'multa', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MulTA)
    def mmused_acc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss()
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=2)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=2)}
        config.optimizer_args = {
            'lr': 1e-04,
            'weight_decay': 1e-03
        }
        config.d_ffn = 2048
        config.n_blocks = 2
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768 * 2, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.positional_encoder = lambda: PositionalEncoding(d_model=768, dual_modality=False)
        config.text_dropout_rate = 0.1
        config.audio_dropout_rate = 0.1

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused-fallacy', 'task:afc', 'mode:text-audio', 'multa', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MulTA)
    def mmused_fallacy_afc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(
            weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255]))
        config.val_metrics = {'val_f1': F1Score(task='multiclass', num_classes=6)}
        config.test_metrics = {'test_f1': F1Score(task='multiclass', num_classes=6)}
        config.optimizer_args = {
            'lr': 1e-04,
            'weight_decay': 1e-03
        }
        config.d_ffn = 2048
        config.n_blocks = 4
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768 * 2, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        )
        config.positional_encoder = lambda: PositionalEncoding(d_model=768, dual_modality=False)
        config.text_dropout_rate = 0.1
        config.audio_dropout_rate = 0.1

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mode:text-audio', 'multa', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=PairMulTA)
    def marg_arc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.loss_function = lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977]))
        config.val_metrics = {
            'val_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.test_metrics = {
            'test_f1': ClassSubsetMulticlassF1Score(task='multiclass', num_classes=3, class_subset=[1, 2])}
        config.optimizer_args = {
            'lr': 1e-04,
            'weight_decay': 1e-03
        }
        config.d_ffn = 2048
        config.n_blocks = 4
        config.head = lambda: th.nn.Sequential(
            th.nn.Linear(768 * 4, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 3)
        )
        config.positional_encoder = lambda: PositionalEncoding(d_model=768, dual_modality=False)
        config.text_dropout_rate = 0.1
        config.audio_dropout_rate = 0.1

        return config
