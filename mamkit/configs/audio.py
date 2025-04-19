import torch as th

from mamkit.configs.base import BaseConfig, ConfigKey
from mamkit.data.datasets import InputMode


class BiLSTMMFCCsConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'mancini-et-al-2022'}): 'ukdebates_asd_mancini_2022',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous'}): 'ukdebates_asd_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'mancini-et-al-2022'}): 'mmused_asd_mancini_2022',
        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous'}): 'mmused_asd_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='acc',
                  tags={'mancini-et-al-2022'}): 'mmused_acc_mancini_2022',
        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='acc',
                  tags={'anonymous'}): 'mmused_acc_anonymous',

        ConfigKey(dataset='marg', input_mode=InputMode.AUDIO_ONLY, task_name='arc',
                  tags={'mancini-et-al-2022'}): 'marg_arc_mancini_2022',
        ConfigKey(dataset='marg', input_mode=InputMode.AUDIO_ONLY, task_name='arc',
                  tags={'anonymous'}): 'marg_arc_anonymous',

        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.AUDIO_ONLY, task_name='afc',
                  tags={'anonymous'}): 'mmused_fallacy_afc_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.AUDIO_ONLY, task_name='afd',
                  tags={'anonymous'}): 'mmused_fallacy_afd_anonymous'
    }

    def __init__(
            self,
            mfccs,
            lstm_weights,
            head,
            num_classes,
            sampling_rate=16000,
            dropout_rate=0.1,
            pooling_sizes=None,
            normalize=True,
            remove_energy=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.mfccs = mfccs
        self.lstm_weights = lstm_weights
        self.head = head
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.pooling_sizes = pooling_sizes
        self.normalize = normalize
        self.remove_energy = remove_energy
        self.embedding_dim = mfccs + 19

    @classmethod
    def ukdebates_asd_mancini_2022(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
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
            seeds=[15371, 15372, 15373],
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
        )

    @classmethod
    def ukdebates_asd_anonymous(
            cls
    ):
        return cls(
            sampling_rate=16000,
            mfccs=25,
            head=lambda: th.nn.Sequential(
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
            dropout_rate=0.,
            pooling_sizes=None,
            normalize=True,
            remove_energy=True,
            num_classes=2,
            seeds=[42, 2024, 666],
            batch_size=16,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684]))
        )

    @classmethod
    def marg_mancini_2022(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
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
    def marg_arc_anonymous(
            cls
    ):
        return cls(
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
            ),
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.001
            },
            optimizer=th.optim.Adam,
            lstm_weights=[64, 32],
            dropout_rate=0.1,
            mfccs=25,
            normalize=True,
            remove_energy=True,
            num_classes=3,
            seeds=[42, 2024, 666],
            batch_size=8,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
        )

    @classmethod
    def mmused_asd_mancini_2022(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 64),
                th.nn.ReLU(),
                th.nn.Linear(64, 2)
            ),
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
    def mmused_asd_anonymous(
            cls
    ):
        return cls(
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            mfccs=25,
            pooling_sizes=[5],
            normalize=True,
            remove_energy=True,
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.001
            },
            optimizer=th.optim.Adam,
            lstm_weights=[64, 32],
            dropout_rate=0.1,
            num_classes=2,
            seeds=[42, 2024, 666],
            batch_size=4,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
        )

    @classmethod
    def mmused_acc_mancini_2022(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 64),
                th.nn.ReLU(),
                th.nn.Linear(64, 2)
            ),
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

    @classmethod
    def mmused_acc_anonymous(
            cls
    ):
        return cls(
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
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
            mfccs=25,
            pooling_sizes=[5],
            normalize=True,
            remove_energy=True,
            num_classes=2,
            seeds=[42, 2024, 666],
            batch_size=4
        )

    @classmethod
    def mmused_fallacy_afc_anonymous(
            cls
    ):
        return cls(
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.001
            },
            optimizer=th.optim.Adam,
            lstm_weights=[64, 32],
            dropout_rate=0.1,
            mfccs=25,
            pooling_sizes=[5],
            normalize=True,
            remove_energy=True,
            num_classes=6,
            seeds=[42],
            batch_size=4,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
        )

    @classmethod
    def mmused_fallacy_afd_anonymous(
            cls
    ):
        return cls(
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
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
            mfccs=25,
            pooling_sizes=[5],
            normalize=True,
            remove_energy=True,
            num_classes=2,
            seeds=[42],
            batch_size=4,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.53858521, 6.97916667])),
        )


class BiLSTMTransformerConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous'}): 'ukdebates_asd_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous', 'hubert'}): 'ukdebates_asd_hubert_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous', 'wavlm'}): 'ukdebates_asd_wavlm_anonymous',

        ConfigKey(dataset='ukdebates', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'mancini-et-al-2022'}): 'ukdebates_asd_mancini_2022',
        ConfigKey(dataset='marg', input_mode=InputMode.AUDIO_ONLY, task_name='arc',
                  tags={'mancini-et-al-2022'}): 'marg_arc_mancini_2022',

        ConfigKey(dataset='marg', input_mode=InputMode.AUDIO_ONLY, task_name='arc',
                  tags={'anonymous'}): 'marg_arc_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.AUDIO_ONLY, task_name='arc',
                  tags={'anonymous', 'hubert'}): 'marg_arc_hubert_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.AUDIO_ONLY, task_name='arc',
                  tags={'anonymous', 'wavlm'}): 'marg_arc_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'mancini-et-al-2022'}): 'mmused_asd_mancini_2022',

        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous'}): 'mmused_asd_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous', 'hubert'}): 'mmused_asd_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous', 'wavlm'}): 'mmused_asd_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='acc',
                  tags={'mancini-et-al-2022'}): 'mmused_acc_mancini_2022',

        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='acc',
                  tags={'anonymous'}): 'mmused_acc_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='acc',
                  tags={'anonymous', 'hubert'}): 'mmused_acc_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='acc',
                  tags={'anonymous', 'wavlm'}): 'mmused_acc_wavlm_anonymous',

        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.AUDIO_ONLY, task_name='afc',
                  tags={'anonymous'}): 'mmused_fallacy_afc_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.AUDIO_ONLY, task_name='afc',
                  tags={'anonymous', 'hubert'}): 'mmused_fallacy_afc_hubert_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.AUDIO_ONLY, task_name='afc',
                  tags={'anonymous', 'wavlm'}): 'mmused_fallacy_afc_wavlm_anonymous'
    }

    def __init__(
            self,
            model_card,
            embedding_dim,
            sampling_rate,
            lstm_weights,
            head,
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
            head=lambda: th.nn.Sequential(
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
            seeds=[42, 2024, 666],
            batch_size=16,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
        )

    @classmethod
    def ukdebates_asd_hubert_anonymous(
            cls
    ):
        return cls(
            model_card='facebook/hubert-base-ls960',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
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
            seeds=[42, 2024, 666],
            batch_size=16,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
        )

    @classmethod
    def ukdebates_asd_wavlm_anonymous(
            cls
    ):
        return cls(
            model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
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
            seeds=[42, 2024, 666],
            batch_size=16,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
        )

    @classmethod
    def ukdebates_asd_mancini_2022(
            cls
    ):
        return cls(
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
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
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
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
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2),
            ),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.001
            },
            optimizer=th.optim.Adam,
            lstm_weights=[64],
            dropout_rate=0.3,
            aggregate=True,
            num_classes=2,
            batch_size=8,
            seeds=[15371, 15372, 15373]
        )

    @classmethod
    def marg_arc_anonymous(
            cls
    ):
        return cls(
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
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
            num_classes=3,
            seeds=[42, 2024, 666],
            batch_size=8,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
        )

    @classmethod
    def marg_arc_hubert_anonymous(
            cls
    ):
        return cls(
            model_card='facebook/hubert-base-ls960',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
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
            num_classes=3,
            seeds=[42, 2024, 666],
            batch_size=8,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
        )

    @classmethod
    def marg_arc_wavlm_anonymous(
            cls
    ):
        return cls(
            model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
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
            num_classes=3,
            seeds=[42, 2024, 666],
            batch_size=8,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
        )

    @classmethod
    def mmused_asd_mancini_2022(
            cls
    ):
        return cls(
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
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
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            seeds=[15371, 15372, 15373]
        )

    @classmethod
    def mmused_asd_anonymous(
            cls
    ):
        return cls(
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
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
            downsampling_factor=1 / 5,
            num_classes=2,
            seeds=[42, 2024, 666],
            batch_size=4,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
        )

    @classmethod
    def mmused_asd_hubert_anonymous(
            cls
    ):
        return cls(
            model_card='facebook/hubert-base-ls960',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
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
            downsampling_factor=1 / 5,
            num_classes=2,
            seeds=[42, 2024, 666],
            batch_size=4,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
        )

    @classmethod
    def mmused_asd_wavlm_anonymous(
            cls
    ):
        return cls(
            model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
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
            downsampling_factor=1 / 5,
            num_classes=2,
            seeds=[42, 2024, 666],
            batch_size=4,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
        )

    @classmethod
    def mmused_acc_mancini_2022(
            cls
    ):
        return cls(
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
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

    @classmethod
    def mmused_acc_anonymous(
            cls
    ):
        return cls(
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
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
            downsampling_factor=1 / 5,
            num_classes=2,
            seeds=[42, 2024, 666],
            batch_size=4
        )

    @classmethod
    def mmused_acc_hubert_anonymous(
            cls
    ):
        return cls(
            model_card='facebook/hubert-base-ls960',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
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
            downsampling_factor=1 / 5,
            num_classes=2,
            seeds=[42, 2024, 666],
            batch_size=4
        )

    @classmethod
    def mmused_acc_wavlm_anonymous(
            cls
    ):
        return cls(
            model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
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
            downsampling_factor=1 / 5,
            num_classes=2,
            seeds=[42, 2024, 666],
            batch_size=4
        )

    @classmethod
    def mmused_fallacy_afc_anonymous(
            cls
    ):
        return cls(
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.001
            },
            optimizer=th.optim.Adam,
            lstm_weights=[64, 32],
            dropout_rate=0.1,
            aggregate=False,
            downsampling_factor=1 / 5,
            num_classes=6,
            seeds=[42],
            batch_size=4,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
        )

    @classmethod
    def mmused_fallacy_afc_hubert_anonymous(
            cls
    ):
        return cls(
            model_card='facebook/hubert-base-ls960',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.001
            },
            optimizer=th.optim.Adam,
            lstm_weights=[64, 32],
            dropout_rate=0.1,
            aggregate=False,
            downsampling_factor=1 / 5,
            num_classes=6,
            seeds=[42],
            batch_size=4,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
        )

    @classmethod
    def mmused_fallacy_afc_wavlm_anonymous(
            cls
    ):
        return cls(
            model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            embedding_dim=768,
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.001
            },
            optimizer=th.optim.Adam,
            lstm_weights=[64, 32],
            dropout_rate=0.1,
            aggregate=False,
            downsampling_factor=1 / 5,
            num_classes=6,
            seeds=[42],
            batch_size=4,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
        )


class TransformerEncoderConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous'}): 'ukdebates_asd_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous', 'hubert'}): 'ukdebates_asd_hubert_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous', 'wavlm'}): 'ukdebates_asd_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous'}): 'mmused_asd_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous', 'hubert'}): 'mmused_asd_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='asd',
                  tags={'anonymous', 'wavlm'}): 'mmused_asd_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='acc',
                  tags={'anonymous'}): 'mmused_acc_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='acc',
                  tags={'anonymous', 'hubert'}): 'mmused_acc_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.AUDIO_ONLY, task_name='acc',
                  tags={'anonymous', 'wavlm'}): 'mmused_acc_wavlm_anonymous',

        ConfigKey(dataset='marg', input_mode=InputMode.AUDIO_ONLY, task_name='arc',
                  tags={'anonymous'}): 'marg_arc_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.AUDIO_ONLY, task_name='arc',
                  tags={'anonymous', 'hubert'}): 'marg_arc_hubert_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.AUDIO_ONLY, task_name='arc',
                  tags={'anonymous', 'wavlm'}): 'marg_arc_wavlm_anonymous',

        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.AUDIO_ONLY, task_name='afc',
                  tags={'anonymous'}): 'mmused_fallacy_afc_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.AUDIO_ONLY, task_name='afc',
                  tags={'anonymous', 'hubert'}): 'mmused_fallacy_afc_hubert_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.AUDIO_ONLY, task_name='afc',
                  tags={'anonymous', 'wavlm'}): 'mmused_fallacy_afc_wavlm_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.AUDIO_ONLY, task_name='afd',
                  tags={'anonymous', 'wavlm'}): 'mmused_fallacy_afd_wavlm_anonymous',
    }

    def __init__(
            self,
            model_card,
            embedding_dim,
            encoder,
            head,
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
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=2,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=None,
            sampling_rate=16000,
            batch_size=16,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.0,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            seeds=[42, 2024, 666],
        )

    @classmethod
    def ukdebates_asd_hubert_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            model_card='facebook/hubert-base-ls960',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=2,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=None,
            sampling_rate=16000,
            batch_size=16,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.0,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            seeds=[42, 2024, 666],
        )

    @classmethod
    def ukdebates_asd_wavlm_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=2,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=None,
            sampling_rate=16000,
            batch_size=16,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.0,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            seeds=[42, 2024, 666],
        )

    @classmethod
    def mmused_asd_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=2,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=1 / 5,
            sampling_rate=16000,
            batch_size=4,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.2,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            seeds=[42, 2024, 666],
        )

    @classmethod
    def mmused_asd_hubert_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            model_card='facebook/hubert-base-ls960',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=2,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=1 / 5,
            sampling_rate=16000,
            batch_size=4,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.2,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            seeds=[42, 2024, 666],
        )

    @classmethod
    def mmused_asd_wavlm_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=2,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=1 / 5,
            sampling_rate=16000,
            batch_size=4,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.2,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            seeds=[42, 2024, 666],
        )

    @classmethod
    def mmused_acc_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=2,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=1 / 5,
            sampling_rate=16000,
            batch_size=4,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.2,
            seeds=[42, 2024, 666],
        )

    @classmethod
    def mmused_acc_hubert_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            model_card='facebook/hubert-base-ls960',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=2,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=1 / 5,
            sampling_rate=16000,
            batch_size=4,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.2,
            seeds=[42, 2024, 666],
        )

    @classmethod
    def mmused_acc_wavlm_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=2,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=1 / 5,
            sampling_rate=16000,
            batch_size=4,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.2,
            seeds=[42, 2024, 666],
        )

    @classmethod
    def marg_arc_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=3,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=None,
            sampling_rate=16000,
            batch_size=4,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.2,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            seeds=[42, 2024, 666],
        )

    @classmethod
    def marg_arc_hubert_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            model_card='facebook/hubert-base-ls960',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=3,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=None,
            sampling_rate=16000,
            batch_size=4,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.2,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            seeds=[42, 2024, 666],
        )

    @classmethod
    def marg_arc_wavlm_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=3,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=None,
            sampling_rate=16000,
            batch_size=4,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.2,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            seeds=[42, 2024, 666],
        )

    @classmethod
    def mmused_fallacy_afc_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            model_card='facebook/wav2vec2-base-960h',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=6,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=1 / 5,
            sampling_rate=16000,
            batch_size=8,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.2,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            seeds=[42],
        )

    @classmethod
    def mmused_fallacy_afc_hubert_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            model_card='facebook/hubert-base-ls960',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=6,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=1 / 5,
            sampling_rate=16000,
            batch_size=8,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.2,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            seeds=[42],
        )

    @classmethod
    def mmused_fallacy_afc_wavlm_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=6,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=1 / 5,
            sampling_rate=16000,
            batch_size=8,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.2,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            seeds=[42],
        )

    @classmethod
    def mmused_fallacy_afd_wavlm_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            embedding_dim=768,
            encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=100, batch_first=True),
                num_layers=1
            ),
            num_classes=2,
            processor_args={},
            model_args={},
            aggregate=False,
            downsampling_factor=1 / 5,
            sampling_rate=16000,
            batch_size=8,
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            dropout_rate=0.2,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.53858521, 6.97916667])),
            seeds=[42],
        )
