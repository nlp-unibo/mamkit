from typing import Type, Dict

from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method

from mamkit.components.text.collators import (
    TextCollator,
    PairTextCollator,
    TextTransformerCollator,
    PairTextTransformerCollator,
    TextTransformerOutputCollator,
    PairTextTransformerOutputCollator
)


class TextCollatorConfig(Configuration):

    @classmethod
    @register_method(name='collator',
                     tags={'text'},
                     namespace='mamkit',
                     component_class=TextCollator)
    @register_method(name='collator',
                     tags={'text', 'pair'},
                     namespace='mamkit',
                     component_class=PairTextCollator)
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()
        return config


class TextTransformerCollatorConfig(Configuration):

    @classmethod
    @register_method(name='collator',
                     tags={'text-transformer'},
                     namespace='mamkit',
                     component_class=TextTransformerCollator)
    @register_method(name='collator',
                     tags={'text-transformer', 'pair'},
                     namespace='mamkit',
                     component_class=PairTextTransformerCollator)
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='model_card',
                   type_hint=str,
                   is_required=True,
                   description='Huggingface model card.',
                   variants=['bert-base-uncased', 'roberta-base'])
        config.add(name='tokenizer_args',
                   type_hint=Dict,
                   description='Additional tokenization arguments')

        return config


class TextTransformerOutputCollatorConfig(Configuration):

    @classmethod
    @register_method(name='collator',
                     tags={'text-transformer-output'},
                     namespace='mamkit',
                     component_class=TextTransformerOutputCollator)
    @register_method(name='collator',
                     tags={'text-transformer-output', 'pair'},
                     namespace='mamkit',
                     component_class=PairTextTransformerOutputCollator)
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()
        return config
