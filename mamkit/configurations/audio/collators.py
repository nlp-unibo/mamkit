from typing import Type, Dict

from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method

from mamkit.components.audio.collators import (
    AudioCollator,
    AudioTransformerCollator,
    PairAudioCollator
)


class AudioCollatorConfig(Configuration):

    @classmethod
    @register_method(name='collator',
                     tags={'audio'},
                     namespace='mamkit',
                     component_class=AudioCollator)
    @register_method(name='collator',
                     tags={'audio', 'pair'},
                     namespace='mamkit',
                     component_class=PairAudioCollator)
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()
        return config


class AudioTransformerCollatorConfig(Configuration):

    @classmethod
    @register_method(name='collator',
                     tags={'audio-transformer'},
                     namespace='mamkit',
                     component_class=AudioTransformerCollator)
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
