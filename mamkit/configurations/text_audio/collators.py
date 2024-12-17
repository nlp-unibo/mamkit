from typing import Type

from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method, RegistrationKey

from mamkit.components.collators import MultimodalCollator, PairMultimodalCollator


class MultimodalCollatorConfig(Configuration):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='text_collator',
                   type_hint=RegistrationKey,
                   description='Text collator.')
        config.add(name='audio_collator',
                   type_hint=RegistrationKey,
                   description='Audio collator.')
        config.add(name='label_collator',
                   value=RegistrationKey(name='collator',
                                         tags={'label'},
                                         namespace='mamkit'),
                   type_hint=RegistrationKey,
                   description='Label collator.')

        return config

    @classmethod
    @register_method(name='collator',
                     tags={'mode:text-audio', 'lstm'},
                     namespace='mamkit',
                     component_class=MultimodalCollator)
    def lstm(
            cls
    ):
        config = cls.default()

        config.text_collator = RegistrationKey(name='collator',
                                               tags={'text'},
                                               namespace='mamkit')
        config.audio_collator = RegistrationKey(name='collator',
                                                tags={'audio'},
                                                namespace='mamkit')
        config.label_collator = RegistrationKey(name='collator',
                                                tags={'label'},
                                                namespace='mamkit')

        return config

    @classmethod
    @register_method(name='collator',
                     tags={'mode:text-audio', 'lstm', 'pair'},
                     namespace='mamkit',
                     component_class=PairMultimodalCollator)
    def pair_lstm(
            cls
    ):
        config = cls.default()

        config.text_collator = RegistrationKey(name='collator',
                                               tags={'text', 'pair'},
                                               namespace='mamkit')
        config.audio_collator = RegistrationKey(name='collator',
                                                tags={'audio', 'pair'},
                                                namespace='mamkit')
        config.label_collator = RegistrationKey(name='collator',
                                                tags={'label'},
                                                namespace='mamkit')

        return config

    @classmethod
    @register_method(name='collator',
                     tags={'mode:text-audio', 'transformer'},
                     namespace='mamkit',
                     component_class=MultimodalCollator)
    def transformer(
            cls
    ):
        config = cls.default()

        config.text_collator = RegistrationKey(name='collator',
                                               tags={'text-transformer'},
                                               namespace='mamkit')
        config.audio_collator = RegistrationKey(name='collator',
                                                tags={'audio'},
                                                namespace='mamkit')
        config.label_collator = RegistrationKey(name='collator',
                                                tags={'label'},
                                                namespace='mamkit')

        return config

    @classmethod
    @register_method(name='collator',
                     tags={'mode:text-audio', 'transformer', 'pair'},
                     namespace='mamkit',
                     component_class=PairMultimodalCollator)
    def pair_transformer(
            cls
    ):
        config = cls.default()

        config.text_collator = RegistrationKey(name='collator',
                                               tags={'text-transformer', 'pair'},
                                               namespace='mamkit')
        config.audio_collator = RegistrationKey(name='collator',
                                                tags={'audio', 'pair'},
                                                namespace='mamkit')
        config.label_collator = RegistrationKey(name='collator',
                                                tags={'label'},
                                                namespace='mamkit')

        return config

    @classmethod
    @register_method(name='collator',
                     tags={'mode:text-audio', 'transformer', 'multimodal'},
                     namespace='mamkit',
                     component_class=MultimodalCollator)
    def multimodal_transformer(
            cls
    ):
        config = cls.default()

        config.text_collator = RegistrationKey(name='collator',
                                               tags={'text-transformer-output'},
                                               namespace='mamkit')
        config.audio_collator = RegistrationKey(name='collator',
                                                tags={'audio'},
                                                namespace='mamkit')
        config.label_collator = RegistrationKey(name='collator',
                                                tags={'label'},
                                                namespace='mamkit')

        return config

    @classmethod
    @register_method(name='collator',
                     tags={'mode:text-audio', 'transformer', 'multimodal', 'pair'},
                     namespace='mamkit',
                     component_class=PairMultimodalCollator)
    def pair_multimodal_transformer(
            cls
    ):
        config = cls.default()

        config.text_collator = RegistrationKey(name='collator',
                                               tags={'text-transformer-output', 'pair'},
                                               namespace='mamkit')
        config.audio_collator = RegistrationKey(name='collator',
                                                tags={'audio', 'pair'},
                                                namespace='mamkit')
        config.label_collator = RegistrationKey(name='collator',
                                                tags={'label'},
                                                namespace='mamkit')

        return config
