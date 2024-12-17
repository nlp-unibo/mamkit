from typing import Type

from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method, RegistrationKey

from mamkit.components.processing import MultimodalProcessor, PairMultimodalProcessor


class MultimodalProcessorConfig(Configuration):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='text_processor',
                   type_hint=RegistrationKey,
                   description='Text processor.',
                   is_required=False)
        config.add(name='audio_processor',
                   type_hint=RegistrationKey,
                   description='Audio processor.',
                   is_required=False)
        config.add(name='label_processor',
                   type_hint=RegistrationKey,
                   description='Label processor.',
                   is_required=False)

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'mode:text-audio', 'lstm', 'transformer'},
                     namespace='mamkit',
                     component_class=MultimodalProcessor)
    def lstm_transformer(
            cls
    ):
        config = cls.default()

        config.get('audio_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused-fallacy', 'task:afc', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit')
        ]
        config.get('text_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'vocab-builder', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'vocab-builder',
                                  'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'vocab-builder',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'vocab-builder',
                                  'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'vocab-builder',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'vocab-builder',
                                  'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused-fallacy', 'task:afc', 'vocab-builder',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
        ]
        config.label_processor = None

        config.add_condition(name='matching-compounds',
                             condition=lambda c: c.audio_processor.compound_tags == c.text_processor.compound_tags)

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'mode:text-audio', 'lstm', 'transformer', 'pair'},
                     namespace='mamkit',
                     component_class=PairMultimodalProcessor)
    def pair_lstm_transformer(
            cls
    ):
        config = cls.default()

        config.get('audio_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit')
        ]
        config.get('text_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'vocab-builder',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'vocab-builder',
                                  'source:mancini-2022-argmining'},
                            namespace='mamkit'),
        ]
        config.label_processor = None

        config.add_condition(name='matching-compounds',
                             condition=lambda c: c.audio_processor.compound_tags == c.text_processor.compound_tags)

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'mode:text-audio', 'lstm', 'mfcc'},
                     namespace='mamkit',
                     component_class=MultimodalProcessor)
    def lstm_mfcc(
            cls
    ):
        config = cls.default()

        config.get('audio_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'mfcc', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'mfcc', 'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'mfcc', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'mfcc', 'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'mfcc', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'mfcc', 'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused-fallacy', 'task:afc', 'mfcc', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
        ]
        config.get('text_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'vocab-builder', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'vocab-builder',
                                  'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'vocab-builder',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'vocab-builder',
                                  'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'vocab-builder',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'vocab-builder',
                                  'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused-fallacy', 'task:afc', 'vocab-builder',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
        ]
        config.label_processor = None

        config.add_condition(name='matching-compounds',
                             condition=lambda c: c.audio_processor.compound_tags == c.text_processor.compound_tags)

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'mode:text-audio', 'lstm', 'mfcc', 'pair'},
                     namespace='mamkit',
                     component_class=PairMultimodalProcessor)
    def pair_lstm_mfcc(
            cls
    ):
        config = cls.default()

        config.get('audio_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'mfcc', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'mfcc', 'source:mancini-2022-argmining'},
                            namespace='mamkit'),
        ]
        config.get('text_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'vocab-builder',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'vocab-builder',
                                  'source:mancini-2022-argmining'},
                            namespace='mamkit'),
        ]
        config.label_processor = None

        config.add_condition(name='matching-compounds',
                             condition=lambda c: c.audio_processor.compound_tags == c.text_processor.compound_tags,
                             description='Make sure that above variants are matched based on compound tags like'
                                         'data:, task:, source:')

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'mode:text-audio', 'transformer'},
                     namespace='mamkit',
                     component_class=MultimodalProcessor)
    def transformer(
            cls
    ):
        config = cls.default()

        config.get('audio_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused-fallacy', 'task:afc', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit')
        ]
        config.text_processor = None
        config.label_processor = None

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'mode:text-audio', 'transformer', 'pair'},
                     namespace='mamkit',
                     component_class=PairMultimodalProcessor)
    def pair_transformer(
            cls
    ):
        config = cls.default()

        config.get('audio_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit')
        ]
        config.text_processor = None
        config.label_processor = None

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'mode:text-audio', 'multimodal', 'transformer'},
                     namespace='mamkit',
                     component_class=MultimodalProcessor)
    def multimodal_transformer(
            cls
    ):
        config = cls.default()

        config.get('audio_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused-fallacy', 'task:afc', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit')
        ]
        config.text_processor = RegistrationKey(name='processor',
                                                tags={'text-transformer'},
                                                namespace='mamkit')
        config.label_processor = None

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'mode:text-audio', 'multimodal', 'transformer', 'pair'},
                     namespace='mamkit',
                     component_class=PairMultimodalProcessor)
    def pair_multimodal_transformer(
            cls
    ):
        config = cls.default()

        config.get('audio_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'audio-transformer', 'mancini_2024_mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'audio-transformer', 'mancini_2022_argmining'},
                            namespace='mamkit'),
        ]
        config.text_processor = RegistrationKey(name='processor',
                                                tags={'text-transformer', 'pair'},
                                                namespace='mamkit')
        config.label_processor = None

        return config
