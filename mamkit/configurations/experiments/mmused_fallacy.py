"""MMUSEDFallacy registrations of the published experiments.

Generated from the legacy configurations in ``mamkit.configs`` and the demos;
``tests/test_registrations.py`` checks them against ``tests/golden/configs.json``.
"""

from cinnamon.registry import register_method

from mamkit.configurations.datasets import MMUSEDFallacyConfig
from mamkit.configurations.features import (
    EncodedAudioConfig,
    EncodedTextConfig,
    MFCCAudioConfig,
    TokenizedTextConfig,
    VocabularyTextConfig,
)
from mamkit.configurations.keys import (
    MACRO_F1_2_CLASSES,
    MACRO_F1_6_CLASSES,
    MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO,
    MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL,
    MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_TASK,
    MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO,
    MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_MODEL,
    MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_TASK,
    MMUSED_FALLACY_AFC_AUDIO_ONLY_LOADER,
    MMUSED_FALLACY_AFC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO,
    MMUSED_FALLACY_AFC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL,
    MMUSED_FALLACY_AFC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_TASK,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TASK,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_CONCAT,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_MODEL,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TASK,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TEXT,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_CSA_ANONYMOUS_CSA,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_CSA_ANONYMOUS_MODEL,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_CSA_ANONYMOUS_TASK,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO_MODEL,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_MODEL,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TASK,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT_MODEL,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_LOADER,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_AUDIO,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_CONCAT,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_MODEL,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TASK,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TEXT,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TASK,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_MULTA_ANONYMOUS_MODEL,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_MULTA_ANONYMOUS_MULTA,
    MMUSED_FALLACY_AFC_TEXT_AUDIO_MULTA_ANONYMOUS_TASK,
    MMUSED_FALLACY_AFC_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL,
    MMUSED_FALLACY_AFC_TEXT_ONLY_BILSTM_ANONYMOUS_TASK,
    MMUSED_FALLACY_AFC_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT,
    MMUSED_FALLACY_AFC_TEXT_ONLY_LOADER,
    MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_MODEL,
    MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TASK,
    MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TEXT,
    MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
    MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TASK,
    MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
    MMUSED_FALLACY_AFD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO,
    MMUSED_FALLACY_AFD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL,
    MMUSED_FALLACY_AFD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_TASK,
    MMUSED_FALLACY_AFD_AUDIO_ONLY_LOADER,
    MMUSED_FALLACY_AFD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO,
    MMUSED_FALLACY_AFD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL,
    MMUSED_FALLACY_AFD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_TASK,
    MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO,
    MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT,
    MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL,
    MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TASK,
    MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT,
    MMUSED_FALLACY_AFD_TEXT_AUDIO_LOADER,
    MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO,
    MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT,
    MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
    MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TASK,
    MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
    MMUSED_FALLACY_AFD_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL,
    MMUSED_FALLACY_AFD_TEXT_ONLY_BILSTM_ANONYMOUS_TASK,
    MMUSED_FALLACY_AFD_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT,
    MMUSED_FALLACY_AFD_TEXT_ONLY_LOADER,
    MMUSED_FALLACY_AFD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
    MMUSED_FALLACY_AFD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TASK,
    MMUSED_FALLACY_AFD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
    MMUSED_FALLACY_ENCODED_AUDIO_1,
    MMUSED_FALLACY_ENCODED_AUDIO_2,
    MMUSED_FALLACY_ENCODED_TEXT_1,
    MMUSED_FALLACY_MANCINI_2024_SPLIT,
    MMUSED_FALLACY_MFCC_1,
    MMUSED_FALLACY_TOKENIZED_1,
    MMUSED_FALLACY_TOKENIZED_2,
    MMUSED_FALLACY_VOCABULARY_1,
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


class MMUSEDFallacyRegistrations:
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_AUDIO_ONLY_LOADER,
        component="mamkit.components.loaders.MMUSEDFallacy",
    )
    def mmused_fallacy_afc_audio_only_loader(cls):
        config = MMUSEDFallacyConfig(
            task_name="afc",
            input_mode="audio-only",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_MFCC_1, component="mamkit.components.features.MFCCAudio"
    )
    def mmused_fallacy_mfcc_1(cls):
        config = MFCCAudioConfig(
            mfccs=25,
            sampling_rate=16000,
            pooling_sizes=[5],
            remove_energy=True,
            normalize=True,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def mmused_fallacy_afc_audio_only_bilstm_mfcc_anonymous_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=44,
            lstm_weights=[64, 32],
            dropout_rate=0.1,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afc_audio_only_bilstm_mfcc_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO,
            head=[128, 6],
            pair=False,
        )
        return config

    # max_epochs from the demo mmused_fallacy_afc_audio_only_lstm_mfcc.py.
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afc_audio_only_bilstm_mfcc_anonymous_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFC_AUDIO_ONLY_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            audio=MMUSED_FALLACY_MFCC_1,
            model=MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL,
            metrics={"f1": MACRO_F1_6_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[
                0.25868821144104004,
                1.0548902750015259,
                2.2878787517547607,
                3.2030303478240967,
                4.096899032592773,
                5.18137264251709,
            ],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0002, "weight_decay": 0.001},
            batch_size=4,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afc/audio-only/bilstm-mfcc/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_AUDIO_ONLY_LOADER,
        component="mamkit.components.loaders.MMUSEDFallacy",
    )
    def mmused_fallacy_afd_audio_only_loader(cls):
        config = MMUSEDFallacyConfig(
            task_name="afd",
            input_mode="audio-only",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def mmused_fallacy_afd_audio_only_bilstm_mfcc_anonymous_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=44,
            lstm_weights=[64, 32],
            dropout_rate=0.1,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afd_audio_only_bilstm_mfcc_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO,
            head=[128, 2],
            pair=False,
        )
        return config

    # max_epochs from the sibling demos (same configuration, corpus, and mode).
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afd_audio_only_bilstm_mfcc_anonymous_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFD_AUDIO_ONLY_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            audio=MMUSED_FALLACY_MFCC_1,
            model=MMUSED_FALLACY_AFD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL,
            metrics={"f1": MACRO_F1_2_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.5385851860046387, 6.979166507720947],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0002, "weight_decay": 0.001},
            batch_size=4,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afd/audio-only/bilstm-mfcc/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_ENCODED_AUDIO_1,
        component="mamkit.components.features.EncodedAudio",
    )
    def mmused_fallacy_encoded_audio_1(cls):
        config = EncodedAudioConfig(
            model_card="facebook/wav2vec2-base-960h",
            sampling_rate=16000,
            downsampling_factor=0.2,
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
        key=MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def mmused_fallacy_afc_audio_only_bilstm_transformer_anonymous_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=768,
            lstm_weights=[64, 32],
            dropout_rate=0.1,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afc_audio_only_bilstm_transformer_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO,
            head=[128, 6],
            pair=False,
        )
        return config

    # max_epochs from the demo mmused_fallacy_afc_audio_only_lstm.py.
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afc_audio_only_bilstm_transformer_anonymous_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFC_AUDIO_ONLY_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            audio=MMUSED_FALLACY_ENCODED_AUDIO_1,
            model=MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_MODEL,
            metrics={"f1": MACRO_F1_6_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[
                0.25868821144104004,
                1.0548902750015259,
                2.2878787517547607,
                3.2030303478240967,
                4.096899032592773,
                5.18137264251709,
            ],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0002, "weight_decay": 0.001},
            batch_size=4,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afc/audio-only/bilstm-transformer/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioTransformer",
    )
    def mmused_fallacy_afc_audio_only_transformer_encoder_anonymous_audio(cls):
        config = AudioTransformerConfig(
            embedding_dim=768,
            n_heads=8,
            ffn_dim=100,
            n_layers=1,
            dropout_rate=0.2,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afc_audio_only_transformer_encoder_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO,
            head=[256, 6],
            pair=False,
        )
        return config

    # max_epochs from the demo mmused_fallacy_afc_audio_only_transformer.py.
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afc_audio_only_transformer_encoder_anonymous_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFC_AUDIO_ONLY_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            audio=MMUSED_FALLACY_ENCODED_AUDIO_1,
            model=MMUSED_FALLACY_AFC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL,
            metrics={"f1": MACRO_F1_6_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[
                0.25868821144104004,
                1.0548902750015259,
                2.2878787517547607,
                3.2030303478240967,
                4.096899032592773,
                5.18137264251709,
            ],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.001, "weight_decay": 1e-05},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afc/audio-only/transformer-encoder/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_ENCODED_AUDIO_2,
        component="mamkit.components.features.EncodedAudio",
    )
    def mmused_fallacy_encoded_audio_2(cls):
        config = EncodedAudioConfig(
            model_card="patrickvonplaten/wavlm-libri-clean-100h-base-plus",
            sampling_rate=16000,
            downsampling_factor=0.2,
            aggregate=False,
            processor_args=None,
            model_args=None,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioTransformer",
    )
    def mmused_fallacy_afd_audio_only_transformer_encoder_anonymous_audio(cls):
        config = AudioTransformerConfig(
            embedding_dim=768,
            n_heads=8,
            ffn_dim=100,
            n_layers=1,
            dropout_rate=0.2,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afd_audio_only_transformer_encoder_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO,
            head=[256, 2],
            pair=False,
        )
        return config

    # max_epochs from the sibling demos (same configuration, corpus, and mode).
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afd_audio_only_transformer_encoder_anonymous_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFD_AUDIO_ONLY_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            audio=MMUSED_FALLACY_ENCODED_AUDIO_2,
            model=MMUSED_FALLACY_AFD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL,
            metrics={"f1": MACRO_F1_2_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.5385851860046387, 6.979166507720947],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.001, "weight_decay": 1e-05},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afd/audio-only/transformer-encoder/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_ONLY_LOADER,
        component="mamkit.components.loaders.MMUSEDFallacy",
    )
    def mmused_fallacy_afc_text_only_loader(cls):
        config = MMUSEDFallacyConfig(
            task_name="afc",
            input_mode="text-only",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_VOCABULARY_1,
        component="mamkit.components.features.VocabularyText",
    )
    def mmused_fallacy_vocabulary_1(cls):
        config = VocabularyTextConfig(
            embedding_model="glove.6B.200d",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT,
        component="mamkit.components.encoders.TextBiLSTM",
    )
    def mmused_fallacy_afc_text_only_bilstm_anonymous_text(cls):
        config = TextBiLSTMConfig(
            embedding_dim=200,
            lstm_weights=[128, 32],
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afc_text_only_bilstm_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFC_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT,
            head=[128, 6],
            pair=False,
        )
        return config

    # max_epochs from the demo mmused_fallacy_afc_text_only_lstm.py.
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_ONLY_BILSTM_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afc_text_only_bilstm_anonymous_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFC_TEXT_ONLY_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            text=MMUSED_FALLACY_VOCABULARY_1,
            model=MMUSED_FALLACY_AFC_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL,
            metrics={"f1": MACRO_F1_6_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[
                0.25868821144104004,
                1.0548902750015259,
                2.2878787517547607,
                3.2030303478240967,
                4.096899032592773,
                5.18137264251709,
            ],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0002, "weight_decay": 0.0001},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afc/text-only/bilstm/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_ONLY_LOADER,
        component="mamkit.components.loaders.MMUSEDFallacy",
    )
    def mmused_fallacy_afd_text_only_loader(cls):
        config = MMUSEDFallacyConfig(
            task_name="afd",
            input_mode="text-only",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT,
        component="mamkit.components.encoders.TextBiLSTM",
    )
    def mmused_fallacy_afd_text_only_bilstm_anonymous_text(cls):
        config = TextBiLSTMConfig(
            embedding_dim=200,
            lstm_weights=[128, 32],
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afd_text_only_bilstm_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFD_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT,
            head=[128, 2],
            pair=False,
        )
        return config

    # max_epochs from the sibling demos (same configuration, corpus, and mode).
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_ONLY_BILSTM_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afd_text_only_bilstm_anonymous_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFD_TEXT_ONLY_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            text=MMUSED_FALLACY_VOCABULARY_1,
            model=MMUSED_FALLACY_AFD_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL,
            metrics={"f1": MACRO_F1_2_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.5385851860046387, 6.979166507720947],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0002, "weight_decay": 0.0001},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afd/text-only/bilstm/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_TOKENIZED_1,
        component="mamkit.components.features.TokenizedText",
    )
    def mmused_fallacy_tokenized_1(cls):
        config = TokenizedTextConfig(
            model_card="bert-base-uncased",
            tokenizer_args=None,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TEXT,
        component="mamkit.components.encoders.TextTransformer",
    )
    def mmused_fallacy_afc_text_only_transformer_anonymous_bert_text(cls):
        config = TextTransformerConfig(
            model_card="bert-base-uncased",
            dropout_rate=0.2,
            is_transformer_trainable=False,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afc_text_only_transformer_anonymous_bert_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TEXT,
            head=[256, 6],
            pair=False,
        )
        return config

    # max_epochs from the demo mmused_fallacy_afc_text_only_bert.py.
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afc_text_only_transformer_anonymous_bert_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFC_TEXT_ONLY_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            text=MMUSED_FALLACY_TOKENIZED_1,
            model=MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_MODEL,
            metrics={"f1": MACRO_F1_6_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[
                0.25868821144104004,
                1.0548902750015259,
                2.2878787517547607,
                3.2030303478240967,
                4.096899032592773,
                5.18137264251709,
            ],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.001, "weight_decay": 1e-05},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/mmused-fallacy/afc/text-only/transformer/anonymous/bert",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_TOKENIZED_2,
        component="mamkit.components.features.TokenizedText",
    )
    def mmused_fallacy_tokenized_2(cls):
        config = TokenizedTextConfig(
            model_card="roberta-base",
            tokenizer_args=None,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
        component="mamkit.components.encoders.TextTransformer",
    )
    def mmused_fallacy_afc_text_only_transformer_anonymous_roberta_text(cls):
        config = TextTransformerConfig(
            model_card="roberta-base",
            dropout_rate=0.2,
            is_transformer_trainable=False,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afc_text_only_transformer_anonymous_roberta_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
            head=[256, 6],
            pair=False,
        )
        return config

    # max_epochs from the demo mmused_fallacy_afc_text_only_roberta.py.
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afc_text_only_transformer_anonymous_roberta_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFC_TEXT_ONLY_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            text=MMUSED_FALLACY_TOKENIZED_2,
            model=MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
            metrics={"f1": MACRO_F1_6_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[
                0.25868821144104004,
                1.0548902750015259,
                2.2878787517547607,
                3.2030303478240967,
                4.096899032592773,
                5.18137264251709,
            ],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.001, "weight_decay": 1e-05},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/mmused-fallacy/afc/text-only/transformer/anonymous/roberta",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
        component="mamkit.components.encoders.TextTransformer",
    )
    def mmused_fallacy_afd_text_only_transformer_anonymous_roberta_text(cls):
        config = TextTransformerConfig(
            model_card="roberta-base",
            dropout_rate=0.2,
            is_transformer_trainable=False,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afd_text_only_transformer_anonymous_roberta_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
            head=[256, 2],
            pair=False,
        )
        return config

    # max_epochs from the sibling demos (same configuration, corpus, and mode).
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afd_text_only_transformer_anonymous_roberta_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFD_TEXT_ONLY_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            text=MMUSED_FALLACY_TOKENIZED_2,
            model=MMUSED_FALLACY_AFD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
            metrics={"f1": MACRO_F1_2_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.5385851860046387, 6.979166507720947],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.001, "weight_decay": 1e-05},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 50,
            },
            save_path="results/mmused-fallacy/afd/text-only/transformer/anonymous/roberta",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_LOADER,
        component="mamkit.components.loaders.MMUSEDFallacy",
    )
    def mmused_fallacy_afc_text_audio_loader(cls):
        config = MMUSEDFallacyConfig(
            task_name="afc",
            input_mode="text-audio",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT,
        component="mamkit.components.encoders.TextBiLSTM",
    )
    def mmused_fallacy_afc_text_audio_bilstm_mfcc_anonymous_text(cls):
        config = TextBiLSTMConfig(
            embedding_dim=200,
            lstm_weights=[128, 32],
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def mmused_fallacy_afc_text_audio_bilstm_mfcc_anonymous_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=44,
            lstm_weights=[64, 32],
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT,
        component="mamkit.components.encoders.Concat",
    )
    def mmused_fallacy_afc_text_audio_bilstm_mfcc_anonymous_concat(cls):
        config = ConcatConfig(
            encoders=[
                MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT,
                MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO,
            ],
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afc_text_audio_bilstm_mfcc_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT,
            head=[128, 6],
            pair=False,
        )
        return config

    # max_epochs from the demo mmused_fallacy_afc_text_audio_lstm_mfcc.py.
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afc_text_audio_bilstm_mfcc_anonymous_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFC_TEXT_AUDIO_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            text=MMUSED_FALLACY_VOCABULARY_1,
            audio=MMUSED_FALLACY_MFCC_1,
            model=MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL,
            metrics={"f1": MACRO_F1_6_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[
                0.25868821144104004,
                1.0548902750015259,
                2.2878787517547607,
                3.2030303478240967,
                4.096899032592773,
                5.18137264251709,
            ],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0002, "weight_decay": 0.001},
            batch_size=4,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afc/text-audio/bilstm-mfcc/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_AUDIO_LOADER,
        component="mamkit.components.loaders.MMUSEDFallacy",
    )
    def mmused_fallacy_afd_text_audio_loader(cls):
        config = MMUSEDFallacyConfig(
            task_name="afd",
            input_mode="text-audio",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT,
        component="mamkit.components.encoders.TextBiLSTM",
    )
    def mmused_fallacy_afd_text_audio_bilstm_mfcc_anonymous_text(cls):
        config = TextBiLSTMConfig(
            embedding_dim=200,
            lstm_weights=[128, 32],
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def mmused_fallacy_afd_text_audio_bilstm_mfcc_anonymous_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=44,
            lstm_weights=[64, 32],
            dropout_rate=0.0,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT,
        component="mamkit.components.encoders.Concat",
    )
    def mmused_fallacy_afd_text_audio_bilstm_mfcc_anonymous_concat(cls):
        config = ConcatConfig(
            encoders=[
                MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT,
                MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO,
            ],
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afd_text_audio_bilstm_mfcc_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT,
            head=[128, 2],
            pair=False,
        )
        return config

    # max_epochs from the sibling demos (same configuration, corpus, and mode).
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afd_text_audio_bilstm_mfcc_anonymous_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFD_TEXT_AUDIO_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            text=MMUSED_FALLACY_VOCABULARY_1,
            audio=MMUSED_FALLACY_MFCC_1,
            model=MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL,
            metrics={"f1": MACRO_F1_2_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.5385851860046387, 6.979166507720947],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0002, "weight_decay": 0.001},
            batch_size=4,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afd/text-audio/bilstm-mfcc/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TEXT,
        component="mamkit.components.encoders.TextBiLSTM",
    )
    def mmused_fallacy_afc_text_audio_bilstm_transformer_anonymous_text(cls):
        config = TextBiLSTMConfig(
            embedding_dim=200,
            lstm_weights=[128, 32],
            dropout_rate=0.1,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def mmused_fallacy_afc_text_audio_bilstm_transformer_anonymous_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=768,
            lstm_weights=[64, 32],
            dropout_rate=0.1,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_CONCAT,
        component="mamkit.components.encoders.Concat",
    )
    def mmused_fallacy_afc_text_audio_bilstm_transformer_anonymous_concat(cls):
        config = ConcatConfig(
            encoders=[
                MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TEXT,
                MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO,
            ],
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afc_text_audio_bilstm_transformer_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_CONCAT,
            head=[128, 6],
            pair=False,
        )
        return config

    # max_epochs from the demo mmused_fallacy_afc_text_audio_lstm.py.
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afc_text_audio_bilstm_transformer_anonymous_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFC_TEXT_AUDIO_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            text=MMUSED_FALLACY_VOCABULARY_1,
            audio=MMUSED_FALLACY_ENCODED_AUDIO_1,
            model=MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_MODEL,
            metrics={"f1": MACRO_F1_6_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[
                0.25868821144104004,
                1.0548902750015259,
                2.2878787517547607,
                3.2030303478240967,
                4.096899032592773,
                5.18137264251709,
            ],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0001, "weight_decay": 0.0005},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afc/text-audio/bilstm-transformer/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_ENCODED_TEXT_1,
        component="mamkit.components.features.EncodedText",
    )
    def mmused_fallacy_encoded_text_1(cls):
        config = EncodedTextConfig(
            model_card="bert-base-uncased",
            tokenizer_args=None,
            model_args=None,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_CSA_ANONYMOUS_CSA,
        component="mamkit.components.encoders.CSA",
    )
    def mmused_fallacy_afc_text_audio_csa_anonymous_csa(cls):
        config = CSAConfig(
            embedding_dim=768,
            ffn_dim=2048,
            n_heads=4,
            n_layers=1,
            encoder_dropout_rate=0.1,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_CSA_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afc_text_audio_csa_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFC_TEXT_AUDIO_CSA_ANONYMOUS_CSA,
            head=[256, 6],
            pair=False,
        )
        return config

    # max_epochs from the demo mmused_fallacy_afc_text_audio_csa.py.
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_CSA_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afc_text_audio_csa_anonymous_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFC_TEXT_AUDIO_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            text=MMUSED_FALLACY_ENCODED_TEXT_1,
            audio=MMUSED_FALLACY_ENCODED_AUDIO_1,
            model=MMUSED_FALLACY_AFC_TEXT_AUDIO_CSA_ANONYMOUS_MODEL,
            metrics={"f1": MACRO_F1_6_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[
                0.25868821144104004,
                1.0548902750015259,
                2.2878787517547607,
                3.2030303478240967,
                4.096899032592773,
                5.18137264251709,
            ],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0001, "weight_decay": 0.001},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afc/text-audio/csa/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT,
        component="mamkit.components.encoders.TextPool",
    )
    def mmused_fallacy_afc_text_audio_ensemble_anonymous_text(cls):
        config = TextPoolConfig(
            embedding_dim=768,
            dropout_rate=0.1,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO,
        component="mamkit.components.encoders.AudioTransformer",
    )
    def mmused_fallacy_afc_text_audio_ensemble_anonymous_audio(cls):
        config = AudioTransformerConfig(
            embedding_dim=768,
            n_heads=4,
            ffn_dim=2048,
            n_layers=1,
            dropout_rate=0.1,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afc_text_audio_ensemble_anonymous_text_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT,
            head=[256, 6],
            pair=False,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afc_text_audio_ensemble_anonymous_audio_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO,
            head=[256, 6],
            pair=False,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_MODEL,
        component="mamkit.components.models.Ensemble",
    )
    def mmused_fallacy_afc_text_audio_ensemble_anonymous_model(cls):
        config = EnsembleConfig(
            text=MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT_MODEL,
            audio=MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO_MODEL,
            lower_bound=0.3,
            upper_bound=0.7,
        )
        return config

    # max_epochs from the demo mmused_fallacy_afc_text_audio_ensemble.py.
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afc_text_audio_ensemble_anonymous_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFC_TEXT_AUDIO_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            text=MMUSED_FALLACY_ENCODED_TEXT_1,
            audio=MMUSED_FALLACY_ENCODED_AUDIO_1,
            model=MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_MODEL,
            metrics={"f1": MACRO_F1_6_CLASSES},
            seeds=[42],
            loss="torch.nn.NLLLoss",
            class_weights=[
                0.25868821144104004,
                1.0548902750015259,
                2.2878787517547607,
                3.2030303478240967,
                4.096899032592773,
                5.18137264251709,
            ],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0001, "weight_decay": 0.001},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afc/text-audio/ensemble/anonymous",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TEXT,
        component="mamkit.components.encoders.TextTransformer",
    )
    def mmused_fallacy_afc_text_audio_mmtransformer_anonymous_bert_text(cls):
        config = TextTransformerConfig(
            model_card="bert-base-uncased",
            dropout_rate=0.2,
            is_transformer_trainable=False,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def mmused_fallacy_afc_text_audio_mmtransformer_anonymous_bert_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=768,
            lstm_weights=[64, 32],
            dropout_rate=0.2,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_CONCAT,
        component="mamkit.components.encoders.Concat",
    )
    def mmused_fallacy_afc_text_audio_mmtransformer_anonymous_bert_concat(cls):
        config = ConcatConfig(
            encoders=[
                MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TEXT,
                MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_AUDIO,
            ],
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afc_text_audio_mmtransformer_anonymous_bert_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_CONCAT,
            head=[128, 6],
            pair=False,
        )
        return config

    # max_epochs from the demo mmused_fallacy_afc_text_audio_bert_hubert.py.
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afc_text_audio_mmtransformer_anonymous_bert_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFC_TEXT_AUDIO_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            text=MMUSED_FALLACY_TOKENIZED_1,
            audio=MMUSED_FALLACY_ENCODED_AUDIO_1,
            model=MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_MODEL,
            metrics={"f1": MACRO_F1_6_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[
                0.25868821144104004,
                1.0548902750015259,
                2.2878787517547607,
                3.2030303478240967,
                4.096899032592773,
                5.18137264251709,
            ],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.001, "weight_decay": 0.0005},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afc/text-audio/mmtransformer/anonymous/bert",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
        component="mamkit.components.encoders.TextTransformer",
    )
    def mmused_fallacy_afc_text_audio_mmtransformer_anonymous_roberta_text(cls):
        config = TextTransformerConfig(
            model_card="roberta-base",
            dropout_rate=0.2,
            is_transformer_trainable=False,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def mmused_fallacy_afc_text_audio_mmtransformer_anonymous_roberta_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=768,
            lstm_weights=[64, 32],
            dropout_rate=0.2,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT,
        component="mamkit.components.encoders.Concat",
    )
    def mmused_fallacy_afc_text_audio_mmtransformer_anonymous_roberta_concat(cls):
        config = ConcatConfig(
            encoders=[
                MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
                MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO,
            ],
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afc_text_audio_mmtransformer_anonymous_roberta_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT,
            head=[128, 6],
            pair=False,
        )
        return config

    # max_epochs from the demo mmused_fallacy_afc_text_audio_roberta_hubert.py.
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afc_text_audio_mmtransformer_anonymous_roberta_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFC_TEXT_AUDIO_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            text=MMUSED_FALLACY_TOKENIZED_2,
            audio=MMUSED_FALLACY_ENCODED_AUDIO_1,
            model=MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
            metrics={"f1": MACRO_F1_6_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[
                0.25868821144104004,
                1.0548902750015259,
                2.2878787517547607,
                3.2030303478240967,
                4.096899032592773,
                5.18137264251709,
            ],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.001, "weight_decay": 0.0005},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afc/text-audio/mmtransformer/anonymous/roberta",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
        component="mamkit.components.encoders.TextTransformer",
    )
    def mmused_fallacy_afd_text_audio_mmtransformer_anonymous_roberta_text(cls):
        config = TextTransformerConfig(
            model_card="roberta-base",
            dropout_rate=0.2,
            is_transformer_trainable=False,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO,
        component="mamkit.components.encoders.AudioBiLSTM",
    )
    def mmused_fallacy_afd_text_audio_mmtransformer_anonymous_roberta_audio(cls):
        config = AudioBiLSTMConfig(
            embedding_dim=768,
            lstm_weights=[64, 32],
            dropout_rate=0.2,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT,
        component="mamkit.components.encoders.Concat",
    )
    def mmused_fallacy_afd_text_audio_mmtransformer_anonymous_roberta_concat(cls):
        config = ConcatConfig(
            encoders=[
                MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT,
                MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO,
            ],
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afd_text_audio_mmtransformer_anonymous_roberta_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT,
            head=[128, 2],
            pair=False,
        )
        return config

    # max_epochs from the sibling demos (same configuration, corpus, and mode).
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afd_text_audio_mmtransformer_anonymous_roberta_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFD_TEXT_AUDIO_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            text=MMUSED_FALLACY_TOKENIZED_2,
            audio=MMUSED_FALLACY_ENCODED_AUDIO_2,
            model=MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL,
            metrics={"f1": MACRO_F1_2_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[0.5385851860046387, 6.979166507720947],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.001, "weight_decay": 0.0005},
            batch_size=8,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afd/text-audio/mmtransformer/anonymous/roberta",
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_MULTA_ANONYMOUS_MULTA,
        component="mamkit.components.encoders.MulTA",
    )
    def mmused_fallacy_afc_text_audio_multa_anonymous_multa(cls):
        config = MulTAConfig(
            embedding_dim=768,
            ffn_dim=2048,
            n_blocks=4,
            text_dropout_rate=0.1,
            audio_dropout_rate=0.1,
        )
        return config

    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_MULTA_ANONYMOUS_MODEL,
        component="mamkit.components.models.Classifier",
    )
    def mmused_fallacy_afc_text_audio_multa_anonymous_model(cls):
        config = ClassifierConfig(
            encoder=MMUSED_FALLACY_AFC_TEXT_AUDIO_MULTA_ANONYMOUS_MULTA,
            head=[256, 6],
            pair=False,
        )
        return config

    # max_epochs from the demo mmused_fallacy_afc_text_audio_multa.py.
    @classmethod
    @register_method(
        key=MMUSED_FALLACY_AFC_TEXT_AUDIO_MULTA_ANONYMOUS_TASK,
        component="mamkit.components.tasks.Task",
        run_method="run",
    )
    def mmused_fallacy_afc_text_audio_multa_anonymous_task(cls):
        config = TaskConfig(
            loader=MMUSED_FALLACY_AFC_TEXT_AUDIO_LOADER,
            split=MMUSED_FALLACY_MANCINI_2024_SPLIT,
            text=MMUSED_FALLACY_ENCODED_TEXT_1,
            audio=MMUSED_FALLACY_ENCODED_AUDIO_1,
            model=MMUSED_FALLACY_AFC_TEXT_AUDIO_MULTA_ANONYMOUS_MODEL,
            metrics={"f1": MACRO_F1_6_CLASSES},
            seeds=[42],
            loss="torch.nn.CrossEntropyLoss",
            class_weights=[
                0.25868821144104004,
                1.0548902750015259,
                2.2878787517547607,
                3.2030303478240967,
                4.096899032592773,
                5.18137264251709,
            ],
            optimizer="torch.optim.Adam",
            optimizer_args={"lr": 0.0001, "weight_decay": 0.001},
            batch_size=4,
            trainer_args={
                "accelerator": "auto",
                "devices": 1,
                "accumulate_grad_batches": 3,
                "max_epochs": 20,
            },
            save_path="results/mmused-fallacy/afc/text-audio/multa/anonymous",
        )
        return config
