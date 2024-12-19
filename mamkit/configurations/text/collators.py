from typing import Type, Dict, Optional

from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method, RegistrationKey
import torchtext

torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer

from mamkit.components.text.collators import (
    TextCollator,
    PairTextCollator,
    TextTransformerCollator,
    PairTextTransformerCollator,
    TextTransformerOutputCollator,
    PairTextTransformerOutputCollator
)
from mamkit.components.collators import UnimodalCollator, PairUnimodalCollator


class UnimodalCollatorConfig(Configuration):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='feature_collator',
                   type_hint=RegistrationKey,
                   description='Feature collator.')
        config.add(name='label_collator',
                   value=RegistrationKey(name='collator',
                                         tags={'label'},
                                         namespace='mamkit'),
                   type_hint=RegistrationKey,
                   description='Label collator.',
                   is_required=False)

        return config

    @classmethod
    @register_method(name='collator',
                     tags={'mode:text-only', 'text'},
                     namespace='mamkit',
                     component_class=UnimodalCollator)
    def text(
            cls
    ):
        config = cls.default()
        config.feature_collator = RegistrationKey(name='collator',
                                                  tags={'text'},
                                                  namespace='mamkit')
        return config

    @classmethod
    @register_method(name='collator',
                     tags={'mode:text-only', 'text', 'pair'},
                     namespace='mamkit',
                     component_class=UnimodalCollator)
    def text_pair(
            cls
    ):
        config = cls.default()
        config.feature_collator = RegistrationKey(name='collator',
                                                  tags={'text', 'pair'},
                                                  namespace='mamkit')
        return config

    @classmethod
    @register_method(name='collator',
                     tags={'mode:text-only', 'text-transformer'},
                     namespace='mamkit',
                     component_class=UnimodalCollator)
    def text_transformer(
            cls
    ):
        config = cls.default()
        config.feature_collator = RegistrationKey(name='collator',
                                                  tags={'text-transformer'},
                                                  namespace='mamkit')
        return config

    @classmethod
    @register_method(name='collator',
                     tags={'mode:text-only', 'text-transformer', 'pair'},
                     namespace='mamkit',
                     component_class=UnimodalCollator)
    def pair_text_transformer(
            cls
    ):
        config = cls.default()
        config.feature_collator = RegistrationKey(name='collator',
                                                  tags={'text-transformer', 'pair'},
                                                  namespace='mamkit')
        return config

    @classmethod
    @register_method(name='collator',
                     tags={'mode:text-only', 'text-transformer-output'},
                     namespace='mamkit',
                     component_class=UnimodalCollator)
    def text_transformer_output(
            cls
    ):
        config = cls.default()
        config.feature_collator = RegistrationKey(name='collator',
                                                  tags={'text-transformer-output'},
                                                  namespace='mamkit')
        return config

    @classmethod
    @register_method(name='collator',
                     tags={'mode:text-only', 'text-transformer-output', 'pair'},
                     namespace='mamkit',
                     component_class=UnimodalCollator)
    def pair_text_transformer_output(
            cls
    ):
        config = cls.default()
        config.feature_collator = RegistrationKey(name='collator',
                                                  tags={'text-transformer-output', 'pair'},
                                                  namespace='mamkit')
        return config


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

        config.add(name='tokenizer',
                   value=get_tokenizer(tokenizer='basic_english'),
                   is_required=True,
                   description='Tokenizer model for text tokenization')

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
                   description='Huggingface model card.',
                   variants=['bert-base-uncased', 'roberta-base'])
        config.add(name='tokenizer_args',
                   type_hint=Optional[Dict],
                   description='Additional tokenization arguments',
                   is_required=False)

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
