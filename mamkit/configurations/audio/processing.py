from typing import Type, List, Dict, Optional

from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method, RegistrationKey

from mamkit.components.audio.processing import (
    MFCCExtractor,
    PairMFCCExtractor,
    AudioTransformer,
    PairAudioTransformer
)
from mamkit.components.processing import UnimodalProcessor, PairUnimodalProcessor

__all__ = [
    'UnimodalProcessorConfig',
    'MFCCExtractorConfig',
    'AudioTransformerConfig'
]


class UnimodalProcessorConfig(Configuration):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='feature_processor',
                   type_hint=RegistrationKey,
                   description='Feature processor.')
        config.add(name='label_processor',
                   type_hint=RegistrationKey,
                   description='Label processor.')

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'mode:audio-only'},
                     namespace='mamkit',
                     component_class=UnimodalProcessor)
    def audio_only(
            cls
    ):
        config = cls.default()

        config.get('feature_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'mfcc', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'mfcc', 'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'mfcc', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'mfcc', 'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'mfcc', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'mfcc', 'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused-fallacy', 'task:afc', 'mfcc', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused-fallacy', 'task:afc', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit')
        ]
        config.label_processor = None

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'mode:audio-only'},
                     namespace='mamkit',
                     component_class=PairUnimodalProcessor)
    def pair_audio_only(
            cls
    ):
        config = cls.default()

        config.get('feature_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'mfcc', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'mfcc', 'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit')
        ]
        config.label_processor = None

        return config


class MFCCExtractorConfig(Configuration):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='mfccs',
                   value=25,
                   type_hint=int,
                   is_required=True,
                   description='Number of MFCC coefficients to extract')
        config.add(name='pooling_sizes',
                   type_hint=Optional[List[int]],
                   description='List of cascade average pooling size to perform.')
        config.add(name='remove_energy',
                   value=True,
                   type_hint=bool,
                   is_required=True,
                   description='Whether to remove energy MFCC or not')
        config.add(name='normalize',
                   value=True,
                   type_hint=bool,
                   is_required=True,
                   description='Whether to normalize audio or not')

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:ukdebates', 'task:asd', 'mfcc', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MFCCExtractor)
    def ukdebates_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.pooling_sizes = None

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:ukdebates', 'task:asd', 'mfcc', 'source:mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=MFCCExtractor)
    def ukdebates_asd_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.pooling_sizes = [5, 5, 5]

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused', 'task:asd', 'mfcc', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MFCCExtractor)
    def mmused_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.pooling_sizes = [5]

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused', 'task:asd', 'mfcc', 'source:mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=MFCCExtractor)
    def mmused_asd_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.pooling_sizes = [5]

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused', 'task:acc', 'mfcc', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MFCCExtractor)
    def mmused_acc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.pooling_sizes = [5]

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused', 'task:acc', 'mfcc', 'source:mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=MFCCExtractor)
    def mmused_acc_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.pooling_sizes = [10]

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused-fallacy', 'task:afc', 'mfcc', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MFCCExtractor)
    def mmused_fallacy_afc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.pooling_sizes = [5]

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:marg', 'task:arc', 'mfcc', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=PairMFCCExtractor)
    def marg_arc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.pooling_sizes = None

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:marg', 'task:arc', 'mfcc', 'source:mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=PairMFCCExtractor)
    def marg_arc_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.pooling_sizes = None

        return config


class AudioTransformerConfig(Configuration):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='model_card',
                   type_hint=str,
                   is_required=True,
                   description='Huggingface model card.',
                   variants=[
                       'facebook/wav2vec2-base-960h',
                       'facebook/hubert-base-ls960',
                       'patrickvonplaten/wavlm-libri-clean-100h-base-plus'
                   ])
        config.add(name='sampling_rate',
                   value=16000,
                   type_hint=int,
                   description='Audio sampling rate')
        config.add(name='downsampling_factor',
                   type_hint=float,
                   description='Downsampling factor to shorten audio')
        config.add(name='aggregate',
                   type_hint=bool,
                   description='Whether to perform mean pooling or not')
        config.add(name='processor_args',
                   type_hint=Dict,
                   description='Additional audio processor arguments')
        config.add(name='model_args',
                   type_hint=Dict,
                   description='Additional model arguments')

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'audio-transformer', 'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=AudioTransformer)
    def ukdebates_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = None
        config.aggregate = False

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'audio-transformer', 'mancini_2022_argmining'},
                     namespace='mamkit',
                     component_class=AudioTransformer)
    def ukdebates_asd_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = None
        config.aggregate = True

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'audio-transformer', 'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=AudioTransformer)
    def mmused_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = 1 / 5
        config.aggregate = False

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'audio-transformer', 'mancini_2022_argmining'},
                     namespace='mamkit',
                     component_class=AudioTransformer)
    def mmused_asd_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = None
        config.aggregate = True

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'audio-transformer', 'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=AudioTransformer)
    def mmused_acc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = 1 / 5
        config.aggregate = False

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'audio-transformer', 'mancini_2022_argmining'},
                     namespace='mamkit',
                     component_class=AudioTransformer)
    def mmused_acc_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = None
        config.aggregate = True

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused-fallacy', 'task:afc', 'audio-transformer', 'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=AudioTransformer)
    def mmused_fallacy_afc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = 1 / 5
        config.aggregate = False

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'audio-transformer', 'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=PairAudioTransformer)
    def marg_arc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = None
        config.aggregate = False

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'audio-transformer', 'mancini_2022_argmining'},
                     namespace='mamkit',
                     component_class=PairAudioTransformer)
    def marg_arc_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = None
        config.aggregate = True

        return config
