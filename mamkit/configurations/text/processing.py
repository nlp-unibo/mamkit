from typing import Type, Dict, Any

from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method, RegistrationKey
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer

from mamkit.components.processing import UnimodalProcessor, PairUnimodalProcessor
from mamkit.components.text.processing import (
    VocabBuilder,
    PairVocabBuilder,
    TextTransformer,
    PairTextTransformer,
)

__all__ = [
    'UnimodalProcessorConfig',
    'VocabBuilderConfig',
    'TextTransformerConfig'
]


class UnimodalProcessorConfig(Configuration):

    @classmethod
    @register_method(name='processor',
                     tags={'mode:text-only', 'default'},
                     namespace='mamkit')
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='feature_processor',
                   type_hint=RegistrationKey,
                   description='Feature processor.',
                   is_required=False)
        config.add(name='label_processor',
                   type_hint=RegistrationKey,
                   description='Label processor.',
                   is_required=False)

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'mode:text-only'},
                     namespace='mamkit',
                     component_class=UnimodalProcessor)
    def text_only(
            cls
    ):
        config = cls.default()

        config.get('feature_processor').variants = [
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
            RegistrationKey(name='processor',
                            tags={'text-transformer'},
                            namespace='mamkit')
        ]
        config.label_processor = None

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'mode:text-only'},
                     namespace='mamkit',
                     component_class=PairUnimodalProcessor)
    def pair_text_only(
            cls
    ):
        config = cls.default()

        config.get('feature_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'vocab-builder',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'vocab-builder',
                                  'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'text-transformer', 'pair'},
                            namespace='mamkit')
        ]
        config.label_processor = None

        return config


class VocabBuilderConfig(Configuration):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='tokenizer',
                   value=get_tokenizer(tokenizer='basic_english'),
                   is_required=True,
                   description='Tokenizer model for text tokenization')
        config.add(name='embedding_dim',
                   type_hint=int,
                   is_required=True,
                   description='Embedding size')
        config.add(name='embedding_model',
                   type_hint=str,
                   description='Gensim pre-trained embedding model name')
        config.add(name='tokenization_args',
                   type_hint=Dict[str, Any],
                   description='Tokenization arguments')

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:ukdebates', 'task:asd', 'vocab-builder', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=VocabBuilder)
    def ukdebates_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.embedding_dim = 200
        config.embedding_model = 'glove.6B.200d'

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:ukdebates', 'task:asd', 'vocab-builder', 'source:mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=VocabBuilder)
    def ukdebates_asd_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.embedding_dim = 200
        config.embedding_model = 'glove.6B.200d'

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused', 'task:asd', 'vocab-builder', 'source:mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=VocabBuilder)
    def mmused_asd_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.embedding_dim = 100
        config.embedding_model = 'glove.6B.100d'

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused', 'task:asd', 'vocab-builder', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=VocabBuilder)
    def mmused_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.embedding_dim = 200
        config.embedding_model = 'glove.6B.200d'

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused', 'task:acc', 'vocab-builder', 'source:mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=VocabBuilder)
    def mmused_acc_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.embedding_dim = 100
        config.embedding_model = 'glove.6B.100d'

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused', 'task:acc', 'vocab-builder', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=VocabBuilder)
    def mmused_acc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.embedding_dim = 200
        config.embedding_model = 'glove.6B.200d'

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:marg', 'task:arc', 'vocab-builder', 'source:mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=PairVocabBuilder)
    def marg_arc_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.embedding_dim = 100
        config.embedding_model = 'glove.6B.100d'

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:marg', 'task:arg', 'vocab-builder', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=PairVocabBuilder)
    def marg_arc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.embedding_dim = 200
        config.embedding_model = 'glove.6B.200d'

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused-fallacy', 'task:afc', 'vocab-builder', 'source:mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=VocabBuilder)
    def mmused_fallacy_afc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.embedding_dim = 200
        config.embedding_model = 'glove.6B.200d'

        return config


class TextTransformerConfig(Configuration):

    @classmethod
    @register_method(name='processor',
                     tags={'text-transformer'},
                     namespace='mamkit',
                     component_class=TextTransformer)
    @register_method(name='processor',
                     tags={'pair', 'text-transformer'},
                     namespace='mamkit',
                     component_class=PairTextTransformer)
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
        config.add(name='model_args',
                   type_hint=Dict,
                   description='Additional model arguments')

        return config
