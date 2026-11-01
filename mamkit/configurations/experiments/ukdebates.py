"""UKDebates registrations of the published experiments.

Generated from the legacy configurations in ``mamkit.configs`` and the demos;
``tests/test_registrations.py`` checks them against ``tests/golden/configs.json``.
"""

from cinnamon.registry import register_method

from mamkit.configurations.datasets import UKDebatesConfig
from mamkit.configurations.features import (
    EncodedAudioConfig,
    EncodedTextConfig,
    MFCCAudioConfig,
    TokenizedTextConfig,
    VocabularyTextConfig,
)
from mamkit.configurations.keys import (
    UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO,
    UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL,
    UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_TASK,
    UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_AUDIO,
    UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_MODEL,
    UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_TASK,
    UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO,
    UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_MODEL,
    UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_TASK,
    UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_AUDIO,
    UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_MODEL,
    UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_TASK,
    UKDEBATES_ASD_AUDIO_ONLY_LOADER,
    UKDEBATES_ASD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO,
    UKDEBATES_ASD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL,
    UKDEBATES_ASD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_TASK,
    UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO,
    UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT,
    UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL,
    UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TASK,
    UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT,
    UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO,
    UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_CONCAT,
    UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_MODEL,
    UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TASK,
    UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TEXT,
    UKDEBATES_ASD_TEXT_AUDIO_CSA_ANONYMOUS_CSA,
    UKDEBATES_ASD_TEXT_AUDIO_CSA_ANONYMOUS_MODEL,
    UKDEBATES_ASD_TEXT_AUDIO_CSA_ANONYMOUS_TASK,
    UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO,
    UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO_MODEL,
    UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_MODEL,
    UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TASK,
    UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT,
    UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT_MODEL,
    UKDEBATES_ASD_TEXT_AUDIO_LOADER,
    UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_AUDIO,
    UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_CONCAT,
    UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_MODEL,
    UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TASK,
    UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TEXT,
    UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO,
    UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT,
    UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
    UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TASK,
    UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
    UKDEBATES_ASD_TEXT_AUDIO_MULTA_ANONYMOUS_MODEL,
    UKDEBATES_ASD_TEXT_AUDIO_MULTA_ANONYMOUS_MULTA,
    UKDEBATES_ASD_TEXT_AUDIO_MULTA_ANONYMOUS_TASK,
    UKDEBATES_ASD_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL,
    UKDEBATES_ASD_TEXT_ONLY_BILSTM_ANONYMOUS_TASK,
    UKDEBATES_ASD_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT,
    UKDEBATES_ASD_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_MODEL,
    UKDEBATES_ASD_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_TASK,
    UKDEBATES_ASD_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_TEXT,
    UKDEBATES_ASD_TEXT_ONLY_LOADER,
    UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_MODEL,
    UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TASK,
    UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TEXT,
    UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
    UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TASK,
    UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
    UKDEBATES_ENCODED_AUDIO_1,
    UKDEBATES_ENCODED_AUDIO_2,
    UKDEBATES_ENCODED_AUDIO_3,
    UKDEBATES_ENCODED_TEXT_1,
    UKDEBATES_F1,
    UKDEBATES_MANCINI_2022_SPLIT,
    UKDEBATES_MFCC_1,
    UKDEBATES_MFCC_2,
    UKDEBATES_TOKENIZED_1,
    UKDEBATES_TOKENIZED_2,
    UKDEBATES_VOCABULARY_1,
)
from mamkit.configurations.models import (
    AudioBiLSTMConfig,
    AudioTransformerConfig,
    ClassifierConfig,
    ConcatConfig,
    CSAConfig,
    EnsembleConfig,
    MulTAConfig,
    TextBiLSTMConfig,
    TextPoolConfig,
    TextTransformerConfig,
)
from mamkit.configurations.tasks import TaskConfig


class UKDebatesRegistrations:
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_LOADER,
        component="mamkit.components.loaders.UKDebates",
    )
    def ukdebates_asd_audio_only_loader(cls):
        config = UKDebatesConfig(
            task_name="asd",
            input_mode="audio-only",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_MFCC_1, component="mamkit.components.features.MFCCAudio"
    )
    def ukdebates_mfcc_1(cls):
        config = MFCCAudioConfig(
            mfccs=25,
            sampling_rate=16000,
            pooling_sizes=None,
            remove_energy=True,
            normalize=True,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def ukdebates_asd_audio_only_bilstm_mfcc_anonymous_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=44,
            lstm_weights=[64, 32],
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_audio_only_bilstm_mfcc_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO,
            head=[128, 2],
            pair=False,
        )
        return config

    # max_epochs from the demo ukdebates_audio_only_lstm_mfcc.py.
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_audio_only_bilstm_mfcc_anonymous_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_AUDIO_ONLY_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            audio=UKDEBATES_MFCC_1,
            model=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[42, 2024, 666],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0002, "weight_decay": 0.001},
            batch_size=16,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/ukdebates/asd/audio-only/bilstm-mfcc/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_MFCC_2, component="mamkit.components.features.MFCCAudio"
    )
    def ukdebates_mfcc_2(cls):
        config = MFCCAudioConfig(
            mfccs=25,
            sampling_rate=16000,
            pooling_sizes=[5, 5, 5],
            remove_energy=True,
            normalize=True,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def ukdebates_asd_audio_only_bilstm_mfcc_mancini_et_al_2022_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=44,
            lstm_weights=[64, 32],
            dropout_rate=0.2,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_audio_only_bilstm_mfcc_mancini_et_al_2022_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_AUDIO,
            head=[256, 2],
            pair=False,
        )
        return config

    # max_epochs from the sibling demos (same configuration, corpus, and mode).
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_audio_only_bilstm_mfcc_mancini_et_al_2022_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_AUDIO_ONLY_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            audio=UKDEBATES_MFCC_2,
            model=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[15371, 15372, 15373],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.001, "weight_decay": 0.005},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/ukdebates/asd/audio-only/bilstm-mfcc/mancini-et-al-2022",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ENCODED_AUDIO_1,
        component="mamkit.components.features.EncodedAudio",
    )
    def ukdebates_encoded_audio_1(cls):
        config = EncodedAudioConfig(
            model_card="facebook/wav2vec2-base-960h",
            sampling_rate=16000,
            downsampling_factor=None,
            aggregate=False,
            processor_args=None,
            model_args=None,
        )
        config.meta.model_card.variants.extend(
            [
                "facebook/hubert-base-ls960",
                "patrickvonplaten/wavlm-libri-clean-100h-base-plus",
            ]
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def ukdebates_asd_audio_only_bilstm_transformer_anonymous_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=768,
            lstm_weights=[64, 32],
            dropout_rate=0.1,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_audio_only_bilstm_transformer_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO,
            head=[128, 2],
            pair=False,
        )
        return config

    # max_epochs from the demo ukdebates_audio_only_lstm.py.
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_audio_only_bilstm_transformer_anonymous_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_AUDIO_ONLY_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            audio=UKDEBATES_ENCODED_AUDIO_1,
            model=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[42, 2024, 666],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0002, "weight_decay": 0.001},
            batch_size=16,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/ukdebates/asd/audio-only/bilstm-transformer/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ENCODED_AUDIO_2,
        component="mamkit.components.features.EncodedAudio",
    )
    def ukdebates_encoded_audio_2(cls):
        config = EncodedAudioConfig(
            model_card="facebook/wav2vec2-base-960h",
            sampling_rate=16000,
            downsampling_factor=None,
            aggregate=True,
            processor_args=None,
            model_args=None,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def ukdebates_asd_audio_only_bilstm_transformer_mancini_et_al_2022_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=768,
            lstm_weights=[64, 32],
            dropout_rate=0.5,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_audio_only_bilstm_transformer_mancini_et_al_2022_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_AUDIO,
            head=[32, 32, 2],
            pair=False,
        )
        return config

    # max_epochs from the sibling demos (same configuration, corpus, and mode).
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_audio_only_bilstm_transformer_mancini_et_al_2022_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_AUDIO_ONLY_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            audio=UKDEBATES_ENCODED_AUDIO_2,
            model=UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[15371, 15372, 15373],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0002, "weight_decay": 0.001},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/ukdebates/asd/audio-only/bilstm-transformer/mancini-et-al-2022",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioTransformer",
    )
    def ukdebates_asd_audio_only_transformer_encoder_anonymous_audio(cls):
        config = AudioTransformerConfig(
            embedding_dim=768,
            n_heads=8,
            ffn_dim=100,
            n_layers=1,
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_audio_only_transformer_encoder_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO,
            head=[256, 2],
            pair=False,
        )
        return config

    # max_epochs from the demo ukdebates_audio_only_transformer.py.
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_audio_only_transformer_encoder_anonymous_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_AUDIO_ONLY_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            audio=UKDEBATES_ENCODED_AUDIO_1,
            model=UKDEBATES_ASD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[42, 2024, 666],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.001, "weight_decay": 1e-05},
            batch_size=16,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/ukdebates/asd/audio-only/transformer-encoder/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_ONLY_LOADER,
        component="mamkit.components.loaders.UKDebates",
    )
    def ukdebates_asd_text_only_loader(cls):
        config = UKDebatesConfig(
            task_name="asd",
            input_mode="text-only",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_VOCABULARY_1,
        component="mamkit.components.features.VocabularyText",
    )
    def ukdebates_vocabulary_1(cls):
        config = VocabularyTextConfig(
            embedding_model="glove.6B.200d",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT,
        component="mamkit.components.encoders.TextBiLSTM",
    )
    def ukdebates_asd_text_only_bilstm_anonymous_text(cls):
        config = TextBiLSTMConfig(
            embedding_dim=200,
            lstm_weights=[128, 32],
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_text_only_bilstm_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT,
            head=[128, 2],
            pair=False,
        )
        return config

    # max_epochs from the demo ukdebates_text_only_lstm.py.
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_ONLY_BILSTM_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_text_only_bilstm_anonymous_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_TEXT_ONLY_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            text=UKDEBATES_VOCABULARY_1,
            model=UKDEBATES_ASD_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[42, 2024, 666],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0001, "weight_decay": 0.0005},
            batch_size=16,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 100,
            },
            save_path="results/ukdebates/asd/text-only/bilstm/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_TEXT,
        component="mamkit.components.encoders.TextBiLSTM",
    )
    def ukdebates_asd_text_only_bilstm_mancini_et_al_2022_text(cls):
        config = TextBiLSTMConfig(
            embedding_dim=200,
            lstm_weights=[128, 32],
            dropout_rate=0.5,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_text_only_bilstm_mancini_et_al_2022_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_TEXT,
            head=[128, 2],
            pair=False,
        )
        return config

    # max_epochs from the sibling demos (same configuration, corpus, and mode).
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_text_only_bilstm_mancini_et_al_2022_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_TEXT_ONLY_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            text=UKDEBATES_VOCABULARY_1,
            model=UKDEBATES_ASD_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[15371, 15372, 15373],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0001, "weight_decay": 0.0005},
            batch_size=16,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 100,
            },
            save_path="results/ukdebates/asd/text-only/bilstm/mancini-et-al-2022",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_TOKENIZED_1, component="mamkit.components.features.TokenizedText"
    )
    def ukdebates_tokenized_1(cls):
        config = TokenizedTextConfig(
            model_card="bert-base-uncased",
            tokenizer_args=None,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TEXT,
        component="mamkit.components.encoders.TextTransformer",
    )
    def ukdebates_asd_text_only_transformer_anonymous_bert_text(cls):
        config = TextTransformerConfig(
            model_card="bert-base-uncased",
            dropout_rate=0.0,
            is_transformer_trainable=False,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_text_only_transformer_anonymous_bert_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TEXT,
            head=[256, 2],
            pair=False,
        )
        return config

    # max_epochs from the demo ukdebates_text_only_bert.py.
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_text_only_transformer_anonymous_bert_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_TEXT_ONLY_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            text=UKDEBATES_TOKENIZED_1,
            model=UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[42, 2024, 666],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.001, "weight_decay": 1e-05},
            batch_size=16,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/ukdebates/asd/text-only/transformer/anonymous/bert",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_TOKENIZED_2, component="mamkit.components.features.TokenizedText"
    )
    def ukdebates_tokenized_2(cls):
        config = TokenizedTextConfig(
            model_card="roberta-base",
            tokenizer_args=None,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
        component="mamkit.components.encoders.TextTransformer",
    )
    def ukdebates_asd_text_only_transformer_anonymous_roberta_text(cls):
        config = TextTransformerConfig(
            model_card="roberta-base",
            dropout_rate=0.0,
            is_transformer_trainable=False,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_text_only_transformer_anonymous_roberta_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
            head=[256, 2],
            pair=False,
        )
        return config

    # max_epochs from the demo ukdebates_text_only_roberta.py.
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_text_only_transformer_anonymous_roberta_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_TEXT_ONLY_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            text=UKDEBATES_TOKENIZED_2,
            model=UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[42, 2024, 666, 11, 1492],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.001, "weight_decay": 1e-05},
            batch_size=16,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/ukdebates/asd/text-only/transformer/anonymous/roberta",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_LOADER,
        component="mamkit.components.loaders.UKDebates",
    )
    def ukdebates_asd_text_audio_loader(cls):
        config = UKDebatesConfig(
            task_name="asd",
            input_mode="text-audio",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT,
        component="mamkit.components.encoders.TextBiLSTM",
    )
    def ukdebates_asd_text_audio_bilstm_mfcc_anonymous_text(cls):
        config = TextBiLSTMConfig(
            embedding_dim=200,
            lstm_weights=[128, 32],
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def ukdebates_asd_text_audio_bilstm_mfcc_anonymous_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=44,
            lstm_weights=[64, 32],
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT,
        component="mamkit.components.encoders.Concat",
    )
    def ukdebates_asd_text_audio_bilstm_mfcc_anonymous_concat(cls):
        config = ConcatConfig(
            encoders=[
                UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT,
                UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO,
            ],
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_text_audio_bilstm_mfcc_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT,
            head=[128, 2],
            pair=False,
        )
        return config

    # max_epochs from the demo ukdebates_text_audio_lstm_mfcc.py.
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_text_audio_bilstm_mfcc_anonymous_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_TEXT_AUDIO_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            text=UKDEBATES_VOCABULARY_1,
            audio=UKDEBATES_MFCC_1,
            model=UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[42, 2024, 666],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0001, "weight_decay": 0.0005},
            batch_size=16,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/ukdebates/asd/text-audio/bilstm-mfcc/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TEXT,
        component="mamkit.components.encoders.TextBiLSTM",
    )
    def ukdebates_asd_text_audio_bilstm_transformer_anonymous_text(cls):
        config = TextBiLSTMConfig(
            embedding_dim=200,
            lstm_weights=[128, 32],
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def ukdebates_asd_text_audio_bilstm_transformer_anonymous_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=768,
            lstm_weights=[64, 32],
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_CONCAT,
        component="mamkit.components.encoders.Concat",
    )
    def ukdebates_asd_text_audio_bilstm_transformer_anonymous_concat(cls):
        config = ConcatConfig(
            encoders=[
                UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TEXT,
                UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO,
            ],
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_text_audio_bilstm_transformer_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_CONCAT,
            head=[128, 2],
            pair=False,
        )
        return config

    # max_epochs from the demo ukdebates_text_audio_lstm.py.
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_text_audio_bilstm_transformer_anonymous_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_TEXT_AUDIO_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            text=UKDEBATES_VOCABULARY_1,
            audio=UKDEBATES_ENCODED_AUDIO_1,
            model=UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[42, 2024, 666],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0001, "weight_decay": 0.0005},
            batch_size=16,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 100,
            },
            save_path="results/ukdebates/asd/text-audio/bilstm-transformer/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ENCODED_TEXT_1, component="mamkit.components.features.EncodedText"
    )
    def ukdebates_encoded_text_1(cls):
        config = EncodedTextConfig(
            model_card="bert-base-uncased",
            tokenizer_args=None,
            model_args=None,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_CSA_ANONYMOUS_CSA,
        component="mamkit.components.encoders.CSA",
    )
    def ukdebates_asd_text_audio_csa_anonymous_csa(cls):
        config = CSAConfig(
            embedding_dim=768,
            ffn_dim=2048,
            n_heads=4,
            n_layers=1,
            encoder_dropout_rate=0.1,
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_CSA_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_text_audio_csa_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_TEXT_AUDIO_CSA_ANONYMOUS_CSA,
            head=[256, 2],
            pair=False,
        )
        return config

    # max_epochs from the demo ukdebates_text_audio_csa.py.
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_CSA_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_text_audio_csa_anonymous_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_TEXT_AUDIO_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            text=UKDEBATES_ENCODED_TEXT_1,
            audio=UKDEBATES_ENCODED_AUDIO_1,
            model=UKDEBATES_ASD_TEXT_AUDIO_CSA_ANONYMOUS_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[42, 2024, 666],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0001, "weight_decay": 0.001},
            batch_size=16,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/ukdebates/asd/text-audio/csa/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT,
        component="mamkit.components.encoders.TextPool",
    )
    def ukdebates_asd_text_audio_ensemble_anonymous_text(cls):
        config = TextPoolConfig(
            embedding_dim=768,
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioTransformer",
    )
    def ukdebates_asd_text_audio_ensemble_anonymous_audio(cls):
        config = AudioTransformerConfig(
            embedding_dim=768,
            n_heads=4,
            ffn_dim=2048,
            n_layers=1,
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_text_audio_ensemble_anonymous_text_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT,
            head=[256, 2],
            pair=False,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_text_audio_ensemble_anonymous_audio_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO,
            head=[256, 2],
            pair=False,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_MODEL,
        component="mamkit.components.models.Ensemble",
    )
    def ukdebates_asd_text_audio_ensemble_anonymous_model(cls):
        config = EnsembleConfig(
            text=UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT_MODEL,
            audio=UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO_MODEL,
            lower_bound=0.3,
            upper_bound=0.7,
        )
        return config

    # max_epochs from the demo ukdebates_text_audio_ensemble.py.
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_text_audio_ensemble_anonymous_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_TEXT_AUDIO_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            text=UKDEBATES_ENCODED_TEXT_1,
            audio=UKDEBATES_ENCODED_AUDIO_1,
            model=UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[42, 2024, 666],
            loss="torch.nn.NLLLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0001, "weight_decay": 0.001},
            batch_size=4,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/ukdebates/asd/text-audio/ensemble/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ENCODED_AUDIO_3,
        component="mamkit.components.features.EncodedAudio",
    )
    def ukdebates_encoded_audio_3(cls):
        config = EncodedAudioConfig(
            model_card="facebook/wav2vec2-base-960h",
            sampling_rate=16000,
            downsampling_factor=None,
            aggregate=False,
            processor_args=None,
            model_args=None,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TEXT,
        component="mamkit.components.encoders.TextTransformer",
    )
    def ukdebates_asd_text_audio_mmtransformer_anonymous_bert_text(cls):
        config = TextTransformerConfig(
            model_card="bert-base-uncased",
            dropout_rate=0.0,
            is_transformer_trainable=False,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def ukdebates_asd_text_audio_mmtransformer_anonymous_bert_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=768,
            lstm_weights=[64, 32],
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_CONCAT,
        component="mamkit.components.encoders.Concat",
    )
    def ukdebates_asd_text_audio_mmtransformer_anonymous_bert_concat(cls):
        config = ConcatConfig(
            encoders=[
                UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TEXT,
                UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_AUDIO,
            ],
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_text_audio_mmtransformer_anonymous_bert_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_CONCAT,
            head=[128, 2],
            pair=False,
        )
        return config

    # max_epochs from the demo ukdebates_text_audio_bert.py.
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_text_audio_mmtransformer_anonymous_bert_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_TEXT_AUDIO_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            text=UKDEBATES_TOKENIZED_1,
            audio=UKDEBATES_ENCODED_AUDIO_3,
            model=UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[42, 2024, 666],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.001, "weight_decay": 0.0005},
            batch_size=16,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/ukdebates/asd/text-audio/mmtransformer/anonymous/bert",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
        component="mamkit.components.encoders.TextTransformer",
    )
    def ukdebates_asd_text_audio_mmtransformer_anonymous_roberta_text(cls):
        config = TextTransformerConfig(
            model_card="roberta-base",
            dropout_rate=0.0,
            is_transformer_trainable=False,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def ukdebates_asd_text_audio_mmtransformer_anonymous_roberta_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=768,
            lstm_weights=[64, 32],
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT,
        component="mamkit.components.encoders.Concat",
    )
    def ukdebates_asd_text_audio_mmtransformer_anonymous_roberta_concat(cls):
        config = ConcatConfig(
            encoders=[
                UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
                UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO,
            ],
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_text_audio_mmtransformer_anonymous_roberta_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT,
            head=[128, 2],
            pair=False,
        )
        return config

    # max_epochs from the demo ukdebates_text_audio_roberta_hubert.py.
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_text_audio_mmtransformer_anonymous_roberta_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_TEXT_AUDIO_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            text=UKDEBATES_TOKENIZED_2,
            audio=UKDEBATES_ENCODED_AUDIO_1,
            model=UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[42, 2024, 666],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.001, "weight_decay": 0.0005},
            batch_size=16,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/ukdebates/asd/text-audio/mmtransformer/anonymous/roberta",
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_MULTA_ANONYMOUS_MULTA,
        component="mamkit.components.encoders.MulTA",
    )
    def ukdebates_asd_text_audio_multa_anonymous_multa(cls):
        config = MulTAConfig(
            embedding_dim=768,
            ffn_dim=2048,
            n_blocks=4,
            text_dropout_rate=0.0,
            audio_dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_MULTA_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def ukdebates_asd_text_audio_multa_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=UKDEBATES_ASD_TEXT_AUDIO_MULTA_ANONYMOUS_MULTA,
            head=[256, 2],
            pair=False,
        )
        return config

    # max_epochs from the demo ukdebates_text_audio_multa.py.
    @classmethod
    @register_method(
        key=UKDEBATES_ASD_TEXT_AUDIO_MULTA_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def ukdebates_asd_text_audio_multa_anonymous_task(cls):
        config = TaskConfig(
            loader=UKDEBATES_ASD_TEXT_AUDIO_LOADER,
            split=UKDEBATES_MANCINI_2022_SPLIT,
            text=UKDEBATES_ENCODED_TEXT_1,
            audio=UKDEBATES_ENCODED_AUDIO_1,
            model=UKDEBATES_ASD_TEXT_AUDIO_MULTA_ANONYMOUS_MODEL,
            metrics={"f1": UKDEBATES_F1},
            seeds=[42, 2024, 666],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.8247863054275513, 1.269736886024475],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0001, "weight_decay": 0.001},
            batch_size=4,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/ukdebates/asd/text-audio/multa/anonymous",
        )
        return config
