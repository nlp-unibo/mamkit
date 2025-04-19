import torch as th
from torchtext.data.utils import get_tokenizer

from mamkit.configs.base import BaseConfig, ConfigKey
from mamkit.data.datasets import InputMode


class BiLSTMConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_ONLY, task_name='asd',
                  tags={'anonymous'}): 'ukdebates_asd_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_ONLY, task_name='asd',
                  tags={'mancini-et-al-2022'}): 'ukdebates_asd_mancini_2022',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_ONLY, task_name='asd',
                  tags={'mancini-et-al-2022'}): 'mmused_asd_mancini_2022',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_ONLY, task_name='asd',
                  tags={'anonymous'}): 'mmused_asd_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_ONLY, task_name='acc',
                  tags={'mancini-et-al-2022'}): 'mmused_acc_mancini_2022',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_ONLY, task_name='acc',
                  tags={'anonymous'}): 'mmused_acc_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_ONLY, task_name='arc',
                  tags={'mancini-et-al-2022'}): 'marg_arc_mancini_2022',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_ONLY, task_name='arc',
                  tags={'anonymous'}): 'marg_arc_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY, task_name='afc',
                  tags={'anonymous'}): 'mmused_fallacy_afc_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY, task_name='afd',
                  tags={'anonymous'}): 'mmused_fallacy_afd_anonymous'
    }

    def __init__(
            self,
            embedding_dim,
            lstm_weights,
            head,
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
        self.head = head
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.tokenization_args = tokenization_args

    @classmethod
    def ukdebates_asd_anonymous(
            cls
    ):
        return cls(
            embedding_dim=200,
            lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            dropout_rate=0.0,
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0001,
                'weight_decay': 0.0005
            },
            embedding_model='glove.6B.200d',
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            batch_size=16,
            num_classes=2
        )

    @classmethod
    def ukdebates_asd_mancini_2022(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
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
            seeds=[15371, 15372, 15373],
            batch_size=16,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684]))
        )

    @classmethod
    def marg_arc_mancini_2022(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(256, 64),
                th.nn.ReLU(),
                th.nn.Linear(64, 2)
            ),
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
            seeds=[15371, 15372, 15373],
            batch_size=8,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977]))
        )

    @classmethod
    def marg_arc_anonymous(
            cls
    ):
        return cls(
            embedding_dim=200,
            lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 3)
            ),
            dropout_rate=0.0,
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.0001
            },
            embedding_model='glove.6B.200d',
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3
        )

    @classmethod
    def mmused_asd_mancini_2022(
            cls
    ):
        return cls(
            head=lambda: th.nn.Sequential(
                th.nn.Linear(128, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            dropout_rate=0.0,
            embedding_dim=100,
            embedding_model='glove.6B.100d',
            lstm_weights=[64, 64],
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.0001
            },
            num_classes=2,
            seeds=[15371, 15372, 15373],
            batch_size=4,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223]))
        )

    @classmethod
    def mmused_asd_anonymous(
            cls
    ):
        return cls(
            embedding_dim=200,
            lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            dropout_rate=0.0,
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.0001
            },
            embedding_model='glove.6B.200d',
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            batch_size=4,
            num_classes=2
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
            dropout_rate=0.3,
            embedding_dim=100,
            embedding_model='glove.6B.100d',
            lstm_weights=[64, 32],
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.001,
                'weight_decay': 0.0005
            },
            num_classes=2,
            seeds=[15371, 15372, 15373],
            batch_size=4,
        )

    @classmethod
    def mmused_acc_anonymous(
            cls
    ):
        return cls(
            embedding_dim=200,
            lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            dropout_rate=0.0,
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.0001
            },
            embedding_model='glove.6B.200d',
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            batch_size=4,
            num_classes=2
        )

    @classmethod
    def mmused_fallacy_afc_anonymous(
            cls
    ):
        return cls(
            embedding_dim=200,
            lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 6)
            ),
            dropout_rate=0.0,
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.0001
            },
            embedding_model='glove.6B.200d',
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303 , 4.09689922, 5.18137255])),
            batch_size=8,
            num_classes=6
        )

    @classmethod
    def mmused_fallacy_afd_anonymous(
            cls
    ):
        return cls(
            embedding_dim=200,
            lstm_weights=[128, 32],
            head=lambda: th.nn.Sequential(
                th.nn.Linear(64, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            dropout_rate=0.0,
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={
                'lr': 0.0002,
                'weight_decay': 0.0001
            },
            embedding_model='glove.6B.200d',
            tokenizer=get_tokenizer(tokenizer='basic_english'),
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.53858521, 6.97916667])),
            batch_size=8,
            num_classes=2
        )


class TransformerConfig(BaseConfig):
    configs = {
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_ONLY, task_name='asd',
                  tags={'anonymous', 'bert'}): 'ukdebates_asd_bert_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_ONLY, task_name='asd',
                  tags={'anonymous', 'roberta'}): 'ukdebates_asd_roberta_anonymous',
        ConfigKey(dataset='ukdebates', input_mode=InputMode.TEXT_ONLY, task_name='asd',
                  tags={'mancini-et-al-2022'}): 'ukdebates_asd_mancini_2022',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_ONLY, task_name='asd',
                  tags={'anonymous', 'bert'}): 'mmused_asd_bert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_ONLY, task_name='asd',
                  tags={'anonymous', 'roberta'}): 'mmused_asd_roberta_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY, task_name='afc',
                  tags={'mancini-et-al-2024', 'bert'}): 'mmused_fallacy_afc_bert_mancini_2024',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY, task_name='afc',
                  tags={'mancini-et-al-2024', 'roberta'}): 'mmused_fallacy_afc_roberta_mancini_2024',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY, task_name='afc',
                  tags={'anonymous', 'bert'}): 'mmused_fallacy_afc_bert_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY, task_name='afc',
                  tags={'anonymous', 'roberta'}): 'mmused_fallacy_afc_roberta_anonymous',
        ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY, task_name='afd',
                  tags={'anonymous', 'roberta'}): 'mmused_fallacy_afd_roberta_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_ONLY, task_name='acc',
                  tags={'anonymous', 'bert'}): 'mmused_acc_bert_anonymous',
        ConfigKey(dataset='mmused', input_mode=InputMode.TEXT_ONLY, task_name='acc',
                  tags={'anonymous', 'roberta'}): 'mmused_acc_roberta_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_ONLY, task_name='arc',
                  tags={'anonymous', 'bert'}): 'marg_arc_bert_anonymous',
        ConfigKey(dataset='marg', input_mode=InputMode.TEXT_ONLY, task_name='arc',
                  tags={'anonymous', 'roberta'}): 'marg_arc_roberta_anonymous'

    }

    def __init__(
            self,
            model_card,
            head,
            num_classes,
            dropout_rate=0.0,
            is_transformer_trainable: bool = False,
            tokenizer_args=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.model_card = model_card
        self.head = head
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.is_transformer_trainable = is_transformer_trainable
        self.tokenizer_args = tokenizer_args

    @classmethod
    def ukdebates_asd_bert_anonymous(
            cls
    ):
        return cls(
            model_card='bert-base-uncased',
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            dropout_rate=0.0,
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            batch_size=16,
            num_classes=2,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            is_transformer_trainable=False
        )

    @classmethod
    def ukdebates_asd_roberta_anonymous(
            cls
    ):
        return cls(
            model_card='roberta-base',
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            dropout_rate=0.0,
            seeds=[42, 2024, 666, 11, 1492],
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            batch_size=16,
            num_classes=2,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
            is_transformer_trainable=False
        )

    @classmethod
    def ukdebates_asd_mancini_2022(
            cls
    ):
        return cls(
            model_card='bert-base-uncased',
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 128),
                th.nn.ReLU(),
                th.nn.Linear(128, 2)
            ),
            num_classes=2,
            dropout_rate=0.0,
            is_transformer_trainable=True,
            tokenizer_args={},
            batch_size=16,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.82478632, 1.26973684])),
        )

    @classmethod
    def mmused_asd_bert_anonymous(
            cls
    ):
        return cls(
            model_card='bert-base-uncased',
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            dropout_rate=0.2,
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            batch_size=4,
            num_classes=2,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            is_transformer_trainable=False
        )

    @classmethod
    def mmused_asd_roberta_anonymous(
            cls
    ):
        return cls(
            model_card='roberta-base',
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            dropout_rate=0.2,
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            batch_size=4,
            num_classes=2,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([2.15385234, 0.65116223])),
            is_transformer_trainable=False
        )

    @classmethod
    def mmused_acc_bert_anonymous(
            cls
    ):
        return cls(
            model_card='bert-base-uncased',
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            dropout_rate=0.2,
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            batch_size=4,
            num_classes=2,
            is_transformer_trainable=False
        )

    @classmethod
    def mmused_acc_roberta_anonymous(
            cls
    ):
        return cls(
            model_card='roberta-base',
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            dropout_rate=0.2,
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            batch_size=4,
            num_classes=2,
            is_transformer_trainable=False
        )

    @classmethod
    def mmused_fallacy_afc_bert_mancini_2024(
            cls
    ):
        return cls(
            model_card='bert-base-uncased',
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 100),
                th.nn.ReLU(),
                th.nn.Linear(100, 50),
                th.nn.ReLU(),
                th.nn.Linear(50, 6)
            ),
            num_classes=6,
            dropout_rate=0.1,
            is_transformer_trainable=True,
            tokenizer_args={},
            batch_size=8,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303 , 4.09689922, 5.18137255])),
        )

    @classmethod
    def mmused_fallacy_afc_roberta_mancini_2024(
            cls
    ):
        return cls(
            model_card='roberta-base',
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 100),
                th.nn.ReLU(),
                th.nn.Linear(100, 50),
                th.nn.ReLU(),
                th.nn.Linear(50, 6)
            ),
            num_classes=6,
            dropout_rate=0.1,
            is_transformer_trainable=True,
            tokenizer_args={},
            batch_size=8,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303 , 4.09689922, 5.18137255])),
        )

    @classmethod
    def mmused_fallacy_afc_bert_anonymous(
            cls
    ):
        return cls(
            model_card='bert-base-uncased',
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            dropout_rate=0.2,
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            batch_size=8,
            num_classes=6,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
            is_transformer_trainable=False
        )

    @classmethod
    def mmused_fallacy_afc_roberta_anonymous(
            cls
    ):
        return cls(
            model_card='roberta-base',
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 6)
            ),
            dropout_rate=0.2,
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            batch_size=8,
            num_classes=6,
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(
                weight=th.Tensor([0.2586882, 1.05489022, 2.28787879, 3.2030303, 4.09689922, 5.18137255])),
        )

    @classmethod
    def mmused_fallacy_afd_roberta_anonymous(
            cls
    ):
        return cls(
            model_card='roberta-base',
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 2)
            ),
            dropout_rate=0.2,
            seeds=[42],
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            batch_size=8,
            num_classes=2,
            is_transformer_trainable=False,
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.53858521, 6.97916667])),
        )

    @classmethod
    def marg_arc_bert_anonymous(
            cls
    ):
        return cls(
            model_card='bert-base-uncased',
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            dropout_rate=0.2,
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
            is_transformer_trainable=False
        )

    @classmethod
    def marg_arc_roberta_anonymous(
            cls
    ):
        return cls(
            model_card='roberta-base',
            head=lambda: th.nn.Sequential(
                th.nn.Linear(768 * 2, 256),
                th.nn.ReLU(),
                th.nn.Linear(256, 3)
            ),
            dropout_rate=0.2,
            seeds=[42, 2024, 666],
            optimizer=th.optim.Adam,
            optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
            loss_function=lambda: th.nn.CrossEntropyLoss(weight=th.Tensor([0.35685072, 6.16919192, 28.08045977])),
            batch_size=8,
            num_classes=3,
            is_transformer_trainable=False
        )
