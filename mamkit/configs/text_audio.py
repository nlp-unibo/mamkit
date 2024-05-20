import torch as th
from torchtext.data.utils import get_tokenizer

from mamkit.configs.base import BaseConfig, ConfigKey
from mamkit.data.datasets import InputMode
from mamkit.modules.transformer import CustomEncoder, PositionalEncoding


class BiLSTMMFCCsConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous'}): 'ukdebates_asd_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous'}): 'mmused_asd_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous'}): 'mmused_acc_anonymous',

        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous'}): 'marg_arc_anonymous',

        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous'}): 'mmused_fallacy_afc_anonymous'
    }

    def __init__(
            self,
            text_embedding_dim,
            text_lstm_weights,
            mfccs,
            audio_lstm_weights,
            head,
            num_classes,
            tokenizer,
            tokenization_args=None,
            embedding_model=None,
            sampling_rate=16000,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            pooling_sizes=None,
            normalize=True,
            remove_energy=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.mfccs = mfccs
        self.head = head
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.pooling_sizes = pooling_sizes
        self.normalize = normalize
        self.remove_energy = remove_energy
        self.audio_embedding_dim = mfccs + 19
        self.text_embedding_dim = text_embedding_dim
        self.embedding_model = embedding_model
        self.text_lstm_weights = text_lstm_weights
        self.audio_lstm_weights = audio_lstm_weights
        self.text_dropout_rate = text_dropout_rate
        self.audio_dropout_rate = audio_dropout_rate
        self.tokenizer = tokenizer
        self.tokenization_args = tokenization_args

    @classmethod
    def ukdebates_asd_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            audio_lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=16,
            num_classes=2,
            mfccs=25,
            pooling_sizes=None,
            normalize=True,
            remove_energy=True
        )

    @classmethod
    def marg_arc_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(256, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
            ),
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.001
            },
            optimizer=th.optim.Adam,
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            embedding_model='glove.6B.200d',
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            audio_lstm_weights=[64, 32],
            mfccs=25,
            normalize=True,
            remove_energy=True,
            num_classes=3,
            seeds=[42, 2024, 666],
            batch_size=8,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
        )

    @classmethod
    def mmused_asd_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            embedding_model='glove.6B.200d',
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            audio_lstm_weights=[64, 32],
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
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
            num_classes=2,
            seeds=[42, 2024, 666],
            batch_size=4,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
        )

    @classmethod
    def mmused_acc_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            embedding_model='glove.6B.200d',
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            audio_lstm_weights=[64, 32],
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.001
            },
            optimizer=th.optim.Adam,
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
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            embedding_model='glove.6B.200d',
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            audio_lstm_weights=[64, 32],
            sampling_rate=16000,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.001
            },
            optimizer=th.optim.Adam,
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


class BiLSTMTransformerConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous'}): 'ukdebates_asd_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'hubert'}): 'ukdebates_asd_hubert_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'wavlm'}): 'ukdebates_asd_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous'}): 'mmused_asd_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'hubert'}): 'mmused_asd_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'wavlm'}): 'mmused_asd_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous'}): 'mmused_acc_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous', 'hubert'}): 'mmused_acc_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous', 'wavlm'}): 'mmused_acc_wavlm_anonymous',

        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous'}): 'marg_arc_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous', 'hubert'}): 'marg_arc_hubert_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous', 'wavlm'}): 'marg_arc_wavlm_anonymous',

        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous'}): 'mmused_fallacy_afc_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous', 'hubert'}): 'mmused_fallacy_afc_hubert_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous', 'wavlm'}): 'mmused_fallacy_afc_wavlm_anonymous'
    }

    def __init__(
            self,
            text_embedding_dim,
            text_lstm_weights,
            audio_embedding_dim,
            audio_lstm_weights,
            head,
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
    def ukdebates_asd_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
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
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=16,
            num_classes=2,
        )

    @classmethod
    def ukdebates_asd_hubert_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=16,
            num_classes=2,
        )

    @classmethod
    def ukdebates_asd_wavlm_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=16,
            num_classes=2,
        )

    @classmethod
    def mmused_asd_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_asd_hubert_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_asd_wavlm_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_acc_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_acc_hubert_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_acc_wavlm_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def marg_arc_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(256, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
            ),
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
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
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
        )

    @classmethod
    def marg_arc_hubert_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(256, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
            ),
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
        )

    @classmethod
    def marg_arc_wavlm_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(256, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
            ),
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
        )

    @classmethod
    def mmused_fallacy_afc_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
        )

    @classmethod
    def mmused_fallacy_afc_hubert_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
        )

    @classmethod
    def mmused_fallacy_afc_wavlm_anonymous(
            cls
    ):
        return cls(
            text_embedding_dim=200,
            text_lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            audio_embedding_dim=768,
            audio_lstm_weights=[64, 32],
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            embedding_model='glove.6B.200d',
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
        )


class MMTransformerConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'bert', 'wav2vec'}): 'ukdebates_asd_bert_wav2vec_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'bert', 'clap'}): 'ukdebates_asd_bert_clap_anonymous',

        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'roberta', 'wav2vec'}): 'ukdebates_asd_roberta_wav2vec_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'roberta', 'hubert'}): 'ukdebates_asd_roberta_hubert_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'roberta', 'wavlm'}): 'ukdebates_asd_roberta_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'bert', 'wav2vec'}): 'mmused_asd_bert_wav2vec_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'bert', 'hubert'}): 'mmused_asd_bert_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'bert', 'wavlm'}): 'mmused_asd_bert_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'roberta', 'wav2vec'}): 'mmused_asd_roberta_wav2vec_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'roberta', 'hubert'}): 'mmused_asd_roberta_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'roberta', 'wavlm'}): 'mmused_asd_roberta_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous', 'bert', 'wav2vec'}): 'mmused_acc_bert_wav2vec_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous', 'bert', 'hubert'}): 'mmused_acc_bert_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous', 'bert', 'wavlm'}): 'mmused_acc_bert_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous', 'roberta', 'wav2vec'}): 'mmused_acc_roberta_wav2vec_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous', 'roberta', 'hubert'}): 'mmused_acc_roberta_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous', 'roberta', 'wavlm'}): 'mmused_acc_roberta_wavlm_anonymous',

        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous', 'bert', 'wav2vec'}): 'mmused_fallacy_afc_bert_wav2vec_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous', 'bert', 'hubert'}): 'mmused_fallacy_afc_bert_hubert_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous', 'bert', 'wavlm'}): 'mmused_fallacy_afc_bert_wavlm_anonymous',

        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous', 'roberta', 'wav2vec'}): 'mmused_fallacy_afc_roberta_wav2vec_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous', 'roberta', 'hubert'}): 'mmused_fallacy_afc_roberta_hubert_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous', 'roberta', 'wavlm'}): 'mmused_fallacy_afc_roberta_wavlm_anonymous',

        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous', 'bert', 'wav2vec'}): 'marg_arc_bert_wav2vec_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous', 'bert', 'hubert'}): 'marg_arc_bert_hubert_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous', 'bert', 'wavlm'}): 'marg_arc_bert_wavlm_anonymous',

        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous', 'roberta', 'wav2vec'}): 'marg_arc_roberta_wav2vec_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous', 'roberta', 'hubert'}): 'marg_arc_roberta_hubert_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous', 'roberta', 'wavlm'}): 'marg_arc_roberta_wavlm_anonymous'
    }

    def __init__(
            self,
            text_model_card,
            audio_model_card,
            text_embedding_dim,
            audio_embedding_dim,
            sampling_rate,
            lstm_weights,
            head,
            num_classes,
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            aggregate=False,
            downsampling_factor=None,
            processor_args=None,
            audio_model_args=None,
            is_transformer_trainable: bool = False,
            tokenizer_args=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.text_model_card = text_model_card
        self.audio_model_card = audio_model_card
        self.text_embedding_dim = text_embedding_dim
        self.audio_embedding_dim = audio_embedding_dim
        self.sampling_rate = sampling_rate
        self.lstm_weights = lstm_weights
        self.text_dropout_rate = text_dropout_rate
        self.audio_dropout_rate = audio_dropout_rate
        self.aggregate = aggregate
        self.downsampling_factor = downsampling_factor
        self.processor_args = processor_args
        self.audio_model_args = audio_model_args
        self.head = head
        self.num_classes = num_classes
        self.is_transformer_trainable = is_transformer_trainable
        self.tokenizer_args = tokenizer_args

    @classmethod
    def ukdebates_asd_bert_wav2vec_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=16,
            num_classes=2,
        )

    @classmethod
    def ukdebates_asd_bert_hubert_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=16,
            num_classes=2,
        )

    @classmethod
    def ukdebates_asd_bert_wavlm_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=16,
            num_classes=2,
        )

    @classmethod
    def ukdebates_asd_roberta_wav2vec_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=16,
            num_classes=2,
        )

    @classmethod
    def ukdebates_asd_roberta_hubert_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=16,
            num_classes=2,
        )

    @classmethod
    def ukdebates_asd_roberta_wavlm_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=16,
            num_classes=2,
        )

    @classmethod
    def mmused_asd_bert_wav2vec_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_asd_bert_hubert_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_asd_bert_wavlm_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_asd_roberta_wav2vec_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_asd_roberta_hubert_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_asd_roberta_wavlm_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_acc_bert_wav2vec_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_acc_bert_hubert_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_acc_bert_wavlm_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_acc_roberta_wav2vec_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_acc_roberta_hubert_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_acc_roberta_wavlm_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            batch_size=4,
            num_classes=2,
        )

    @classmethod
    def mmused_fallacy_afc_bert_wav2vec_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
        )

    @classmethod
    def mmused_fallacy_afc_bert_hubert_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
        )

    @classmethod
    def mmused_fallacy_afc_bert_wavlm_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
        )

    @classmethod
    def mmused_fallacy_afc_roberta_wav2vec_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
        )

    @classmethod
    def mmused_fallacy_afc_roberta_hubert_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
        )

    @classmethod
    def mmused_fallacy_afc_roberta_wavlm_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
        )

    @classmethod
    def marg_arc_bert_wav2vec_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832 * 2, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
        )

    @classmethod
    def marg_arc_bert_hubert_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832 * 2, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
        )

    @classmethod
    def marg_arc_bert_wavlm_anonymous(
            cls
    ):
        return cls(
            text_model_card='bert-base-uncased',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832 * 2, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
        )

    @classmethod
    def marg_arc_roberta_wav2vec_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832 * 2, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
        )

    @classmethod
    def marg_arc_roberta_hubert_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832 * 2, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
        )

    @classmethod
    def marg_arc_roberta_wavlm_anonymous(
            cls
    ):
        return cls(
            text_model_card='roberta-base',
            text_embedding_dim=768,
            head=lambda: th.nn.Sequential(
                th.nn.Linear(832 * 2, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
            ),
            text_dropout_rate=0.2,
            audio_dropout_rate=0.2,
            audio_embedding_dim=768,
            lstm_weights=[64, 32],
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-03,
                'weight_decay': 0.0005
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
        )


class CSAConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous'}): 'ukdebates_asd_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'hubert'}): 'ukdebates_asd_hubert_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'wavlm'}): 'ukdebates_asd_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous'}): 'mmused_asd_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'hubert'}): 'mmused_asd_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'wavlm'}): 'mmused_asd_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous'}): 'mmused_acc_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous', 'hubert'}): 'mmused_acc_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous', 'wavlm'}): 'mmused_acc_wavlm_anonymous',

        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous'}): 'mmused_fallacy_afc_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous', 'hubert'}): 'mmused_fallacy_afc_hubert_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous', 'wavlm'}): 'mmused_fallacy_afc_wavlm_anonymous',

        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous'}): 'marg_arc_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous', 'hubert'}): 'marg_arc_hubert_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous', 'wavlm'}): 'marg_arc_wavlm_anonymous',
    }

    def __init__(
            self,
            text_model_card,
            audio_model_card,
            transformer,
            head,
            positional_encoder,
            sampling_rate,
            num_classes,
            aggregate=False,
            downsampling_factor=None,
            processor_args=None,
            audio_model_args=None,
            tokenizer_args=None,
            text_model_args=None,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.transformer = transformer
        self.head = head
        self.positional_encoder = positional_encoder
        self.text_model_card = text_model_card
        self.audio_model_card = audio_model_card
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.aggregate = aggregate
        self.downsampling_factor = downsampling_factor
        self.processor_args = processor_args
        self.audio_model_args = audio_model_args
        self.tokenizer_args = tokenizer_args
        self.text_model_args = text_model_args
        self.head = head
        self.text_dropout_rate = text_dropout_rate
        self.audio_dropout_rate = audio_dropout_rate

    @classmethod
    def ukdebates_asd_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=16,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
        )

    @classmethod
    def ukdebates_asd_hubert_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=16,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
        )

    @classmethod
    def ukdebates_asd_wavlm_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=16,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
        )

    @classmethod
    def mmused_asd_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
        )

    @classmethod
    def mmused_asd_hubert_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
        )

    @classmethod
    def mmused_asd_wavlm_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
        )

    @classmethod
    def mmused_acc_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
        )

    @classmethod
    def mmused_acc_hubert_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
        )

    @classmethod
    def mmused_acc_wavlm_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
        )

    @classmethod
    def mmused_fallacy_afc_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
        )

    @classmethod
    def mmused_fallacy_afc_hubert_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
        )

    @classmethod
    def mmused_fallacy_afc_wavlm_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
        )

    @classmethod
    def marg_arc_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
        )

    @classmethod
    def marg_arc_hubert_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
        )

    @classmethod
    def marg_arc_wavlm_anonymous(
            cls
    ):
        return cls(
            transformer=lambda: CustomEncoder(d_model=768, ffn_hidden=2048, n_head=4, n_layers=1, drop_prob=0.1),
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            positional_encoder=lambda: PositionalEncoding(768, dual_modality=False),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
        )


class EnsembleConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous'}): 'ukdebates_asd_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'hubert'}): 'ukdebates_asd_hubert_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'wavlm'}): 'ukdebates_asd_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous'}): 'mmused_asd_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'hubert'}): 'mmused_asd_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'wavlm'}): 'mmused_asd_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous'}): 'mmused_acc_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous', 'hubert'}): 'mmused_acc_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous', 'wavlm'}): 'mmused_acc_wavlm_anonymous',

        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous'}): 'mmused_fallacy_afc_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous', 'hubert'}): 'mmused_fallacy_afc_hubert_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous', 'wavlm'}): 'mmused_fallacy_afc_wavlm_anonymous',

        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous'}): 'marg_arc_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous', 'hubert'}): 'marg_arc_hubert_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous', 'wavlm'}): 'marg_arc_wavlm_anonymous',
    }

    def __init__(
            self,
            text_model_card,
            audio_model_card,
            audio_encoder,
            text_head,
            audio_head,
            sampling_rate,
            audio_embedding_dim,
            positional_encoder,
            num_classes,
            audio_dropout_rate=0.0,
            text_dropout_rate=0.0,
            aggregate=False,
            downsampling_factor=None,
            processor_args=None,
            audio_model_args=None,
            tokenizer_args=None,
            text_model_args=None,
            lower_bound=0.3,
            upper_bound=0.7,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.audio_encoder = audio_encoder
        self.text_head = text_head
        self.audio_head = audio_head
        self.text_model_card = text_model_card
        self.audio_model_card = audio_model_card
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.aggregate = aggregate
        self.downsampling_factor = downsampling_factor
        self.processor_args = processor_args
        self.audio_model_args = audio_model_args
        self.tokenizer_args = tokenizer_args
        self.text_model_args = text_model_args
        self.audio_embedding_dim = audio_embedding_dim
        self.positional_encoder = positional_encoder
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.audio_dropout_rate = audio_dropout_rate
        self.text_dropout_rate = text_dropout_rate

    @classmethod
    def ukdebates_asd_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0
        )

    @classmethod
    def ukdebates_asd_hubert_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0
        )

    @classmethod
    def ukdebates_asd_wavlm_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0
        )

    @classmethod
    def mmused_asd_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1
        )

    @classmethod
    def mmused_asd_hubert_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1
        )

    @classmethod
    def mmused_asd_wavlm_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1
        )

    @classmethod
    def mmused_acc_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1
        )

    @classmethod
    def mmused_acc_hubert_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1
        )

    @classmethod
    def mmused_acc_wavlm_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1
        )

    @classmethod
    def mmused_fallacy_afc_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1
        )

    @classmethod
    def mmused_fallacy_afc_hubert_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1
        )

    @classmethod
    def mmused_fallacy_afc_wavlm_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1
        )

    @classmethod
    def marg_arc_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1
        )

    @classmethod
    def marg_arc_hubert_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1
        )

    @classmethod
    def marg_arc_wavlm_anonymous(
            cls
    ):
        return cls(
            audio_encoder=lambda: th.nn.TransformerEncoder(
                th.nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=2048, batch_first=True),
                num_layers=1
            ),
            text_head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            audio_head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            loss_function=lambda: th.nn.NLLLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
            lower_bound=0.3,
            upper_bound=0.7,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1
        )


class MulTAConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous'}): 'ukdebates_asd_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'hubert'}): 'ukdebates_asd_hubert_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'wavlm'}): 'ukdebates_asd_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous'}): 'mmused_asd_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'hubert'}): 'mmused_asd_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='asd',
                  tags={'anonymous', 'wavlm'}): 'mmused_asd_wavlm_anonymous',

        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous'}): 'mmused_acc_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous', 'hubert'}): 'mmused_acc_hubert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_AUDIO, task_name='acc',
                  tags={'anonymous', 'wavlm'}): 'mmused_acc_wavlm_anonymous',

        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous'}): 'mmused_fallacy_afc_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous', 'hubert'}): 'mmused_fallacy_afc_hubert_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_AUDIO, task_name='afc',
                  tags={'anonymous', 'wavlm'}): 'mmused_fallacy_afc_wavlm_anonymous',

        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous'}): 'marg_arc_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous', 'hubert'}): 'marg_arc_hubert_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_AUDIO, task_name='arc',
                  tags={'anonymous', 'wavlm'}): 'marg_arc_wavlm_anonymous',
    }

    def __init__(
            self,
            text_model_card,
            audio_model_card,
            head,
            audio_embedding_dim,
            text_embedding_dim,
            d_ffn,
            n_blocks,
            sampling_rate,
            positional_encoder,
            num_classes,
            audio_dropout_rate=0.0,
            text_dropout_rate=0.0,
            aggregate=False,
            downsampling_factor=None,
            processor_args=None,
            audio_model_args=None,
            tokenizer_args=None,
            text_model_args=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.head = head
        self.text_embedding_dim = text_embedding_dim
        self.d_ffn = d_ffn
        self.n_blocks = n_blocks
        self.audio_dropout_rate = audio_dropout_rate
        self.text_dropout_rate = text_dropout_rate
        self.text_model_card = text_model_card
        self.audio_model_card = audio_model_card
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.aggregate = aggregate
        self.downsampling_factor = downsampling_factor
        self.processor_args = processor_args
        self.audio_model_args = audio_model_args
        self.tokenizer_args = tokenizer_args
        self.text_model_args = text_model_args
        self.audio_embedding_dim = audio_embedding_dim
        self.positional_encoder = positional_encoder

    @classmethod
    def ukdebates_asd_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            d_ffn=2048,
            n_blocks=4,
            audio_dropout_rate=0.0,
            text_dropout_rate=0.0,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )

    @classmethod
    def ukdebates_asd_hubert_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            d_ffn=2048,
            n_blocks=4,
            audio_dropout_rate=0.0,
            text_dropout_rate=0.0,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )

    @classmethod
    def ukdebates_asd_wavlm_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            d_ffn=2048,
            n_blocks=4,
            audio_dropout_rate=0.0,
            text_dropout_rate=0.0,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )

    @classmethod
    def mmused_asd_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            d_ffn=2048,
            n_blocks=4,
            audio_dropout_rate=0.1,
            text_dropout_rate=0.1,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )

    @classmethod
    def mmused_asd_hubert_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            d_ffn=2048,
            n_blocks=4,
            audio_dropout_rate=0.1,
            text_dropout_rate=0.1,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )

    @classmethod
    def mmused_asd_wavlm_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            d_ffn=2048,
            n_blocks=4,
            audio_dropout_rate=0.1,
            text_dropout_rate=0.1,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )

    @classmethod
    def mmused_acc_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            d_ffn=2048,
            n_blocks=2,
            audio_dropout_rate=0.1,
            text_dropout_rate=0.1,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )

    @classmethod
    def mmused_acc_hubert_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            d_ffn=2048,
            n_blocks=2,
            audio_dropout_rate=0.1,
            text_dropout_rate=0.1,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )

    @classmethod
    def mmused_acc_wavlm_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            d_ffn=2048,
            n_blocks=2,
            audio_dropout_rate=0.1,
            text_dropout_rate=0.1,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            batch_size=4,
            num_classes=2,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )

    @classmethod
    def mmused_fallacy_afc_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            d_ffn=2048,
            n_blocks=4,
            audio_dropout_rate=0.1,
            text_dropout_rate=0.1,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=4,
            num_classes=6,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )

    @classmethod
    def mmused_fallacy_afc_hubert_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            d_ffn=2048,
            n_blocks=4,
            audio_dropout_rate=0.1,
            text_dropout_rate=0.1,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=4,
            num_classes=6,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )

    @classmethod
    def mmused_fallacy_afc_wavlm_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            d_ffn=2048,
            n_blocks=4,
            audio_dropout_rate=0.1,
            text_dropout_rate=0.1,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            batch_size=4,
            num_classes=6,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=1 / 5,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )

    @classmethod
    def marg_arc_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 4, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            d_ffn=2048,
            n_blocks=4,
            audio_dropout_rate=0.1,
            text_dropout_rate=0.1,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=4,
            num_classes=3,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/wav2vec2-base-960h',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )

    @classmethod
    def marg_arc_hubert_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 4, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            d_ffn=2048,
            n_blocks=4,
            audio_dropout_rate=0.1,
            text_dropout_rate=0.1,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=4,
            num_classes=3,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='facebook/hubert-base-ls960',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )

    @classmethod
    def marg_arc_wavlm_anonymous(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 4, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            d_ffn=2048,
            n_blocks=4,
            audio_dropout_rate=0.1,
            text_dropout_rate=0.1,
            positional_encoder=lambda: PositionalEncoding(d_model=768, dual_modality=False),
            audio_embedding_dim=768,
            text_embedding_dim=768,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=4,
            num_classes=3,
            audio_model_args={},
            processor_args={},
            tokenizer_args={},
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 1e-04,
                'weight_decay': 1e-03
            },
            audio_model_card='patrickvonplaten/wavlm-libri-clean-100h-base-plus',
            sampling_rate=16000,
            aggregate=False,
            downsampling_factor=None,
            text_model_args=None,
            text_model_card='bert-base-uncased',
        )
