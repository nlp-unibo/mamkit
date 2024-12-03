from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method
from typing import Type, List, Dict, Any, Optional
from torchtext.data.utils import get_tokenizer

from mamkit.components.data.processing import (
    VocabBuilder,
    PairVocabBuilder,
    MFCCExtractor,
    PairMFCCExtractor,
    TextTransformer,
    PairTextTransformer,
    AudioTransformer,
    AudioTransformerExtractor,
    PairAudioTransformer,
    PairAudioTransformerExtractor
)


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
                     tags={'data:ukdebates', 'task:asd', 'bilstm', 'mancini-2024-mamkit'},
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
                     tags={'data:ukdebates', 'task:asd', 'bilstm', 'mancini-2022-argmining'},
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
                     tags={'data:mmused', 'task:asd', 'bilstm', 'mancini-2022-argmining'},
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
                     tags={'data:mmused', 'task:asd', 'bilstm', 'mancini-2024-mamkit'},
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
                     tags={'data:mmused', 'task:acc', 'bilstm', 'mancini-2022-argmining'},
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
                     tags={'data:mmused', 'task:acc', 'bilstm', 'mancini-2024-mamkit'},
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
                     tags={'data:marg', 'task:arc', 'bilstm', 'mancini-2022-argmining'},
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
                     tags={'data:marg', 'task:arg', 'bilstm', 'mancini-2024-mamkit'},
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
                     tags={'data:mmused-fallacy', 'task:afc', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=VocabBuilder)
    def mmused_fallacy_afc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.embedding_dim = 200
        config.embedding_model = 'glove.6B.200d'

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
                     tags={'data:ukdebates', 'task:asd', 'bilstm', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=MFCCExtractor)
    def ukdebates_asd_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.pooling_size = [5, 5, 5]

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:ukdebates', 'task:asd', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MFCCExtractor)
    def ukdebates_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.pooling_size = None

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:marg', 'task:arc', 'bilstm', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=PairMFCCExtractor)
    def marg_arc_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.pooling_size = None

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:marg', 'task:arc', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=PairMFCCExtractor)
    def marg_arc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.pooling_size = None

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused', 'task:asd', 'bilstm', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=MFCCExtractor)
    def mmused_asd_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.pooling_size = [5]

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused', 'task:asd', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MFCCExtractor)
    def mmused_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.pooling_size = [5]

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused', 'task:acc', 'bilstm', 'mancini-2022-argmining'},
                     namespace='mamkit',
                     component_class=MFCCExtractor)
    def mmused_acc_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.pooling_size = [10]

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused', 'task:acc', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MFCCExtractor)
    def mmused_acc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.pooling_size = [5]

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'data:mmused-fallacy', 'task:afc', 'bilstm', 'mancini-2024-mamkit'},
                     namespace='mamkit',
                     component_class=MFCCExtractor)
    def mmused_fallacy_afc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.pooling_size = [5]

        return config
