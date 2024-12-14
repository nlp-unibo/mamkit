from typing import Type, List, Dict, Any, Optional

from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method, RegistrationKey
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
                     tags={'mode:text-only', 'transformer'},
                     namespace='mamkit')
    def transformer_text_only(
            cls
    ):
        config = cls.default()

        config.feature_processor = None
        config.label_processor = None

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'mode:text-only', 'bilstm'},
                     namespace='mamkit')
    def bilstm_text_only(
            cls
    ):
        config = cls.default()

        config.get('feature_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'bilstm', 'mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'bilstm', 'mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'bilstm', 'mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'bilstm', 'mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'bilstm', 'mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'bilstm', 'mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'bilstm', 'mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arg', 'bilstm', 'mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused-fallacy', 'task:afc', 'bilstm', 'mancini-2024-mamkit'},
                            namespace='mamkit')
        ]
        config.label_processor = None

        return config

    @classmethod
    @register_method(name='processor',
                     tags={'mode:audio-only', 'bilstm'},
                     namespace='mamkit')
    def bilstm_audio_only(
            cls
    ):
        config = cls.default()

        config.get('feature_processor').variants = [
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'bilstm', 'mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:ukdebates', 'task:asd', 'bilstm', 'mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'bilstm', 'mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:marg', 'task:arc', 'bilstm', 'mancini-2024-mamkit'},
                            namespace='mamkit')
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'bilstm', 'mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:asd', 'bilstm', 'mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'bilstm', 'mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused', 'task:acc', 'bilstm', 'mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='processor',
                            tags={'data:mmused-fallacy', 'task:afc', 'bilstm', 'mancini-2024-mamkit'},
                            namespace='mamkit')
        ]

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


class TextTransformerConfig(Configuration):

    @classmethod
    @register_method(name='model',
                     tags={'mode:text-only', 'transformer'},
                     namespace='mamkit',
                     component_class=TextTransformer)
    @register_method(name='model',
                     tags={'pair', 'mode:text-only', 'transformer'},
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
                     tags={'data:ukdebates', 'task:asd', 'mode:audio-only', 'transformer', 'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=AudioTransformer)
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mode:audio-only', 'transformer-extractor',
                           'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=AudioTransformerExtractor)
    def ukdebates_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = None
        config.aggregate = False

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mode:audio-only', 'transformer', 'mancini_2022_argmining'},
                     namespace='mamkit',
                     component_class=AudioTransformer)
    @register_method(name='model',
                     tags={'data:ukdebates', 'task:asd', 'mode:audio-only', 'transformer-extractor',
                           'mancini_2022_argmining'},
                     namespace='mamkit',
                     component_class=AudioTransformerExtractor)
    def ukdebates_asd_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = None
        config.aggregate = True

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mode:audio-only', 'transformer', 'mancini_2022_argmining'},
                     namespace='mamkit',
                     component_class=PairAudioTransformer)
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mode:audio-only', 'transformer-extractor',
                           'mancini_2022_argmining'},
                     namespace='mamkit',
                     component_class=PairAudioTransformerExtractor)
    def marg_arc_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = None
        config.aggregate = True

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mode:audio-only', 'transformer', 'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=PairAudioTransformer)
    @register_method(name='model',
                     tags={'data:marg', 'task:arc', 'mode:audio-only', 'transformer-extractor', 'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=PairAudioTransformerExtractor)
    def marg_arc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = None
        config.aggregate = False

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mode:audio-only', 'transformer', 'mancini_2022_argmining'},
                     namespace='mamkit',
                     component_class=AudioTransformer)
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mode:audio-only', 'transformer-extractor',
                           'mancini_2022_argmining'},
                     namespace='mamkit',
                     component_class=AudioTransformerExtractor)
    def mmused_asd_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = None
        config.aggregate = True

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mode:audio-only', 'transformer', 'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=AudioTransformer)
    @register_method(name='model',
                     tags={'data:mmused', 'task:asd', 'mode:audio-only', 'transformer-extractor',
                           'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=AudioTransformerExtractor)
    def mmused_asd_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = 1 / 5
        config.aggregate = False

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mode:audio-only', 'transformer', 'mancini_2022_argmining'},
                     namespace='mamkit',
                     component_class=AudioTransformer)
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mode:audio-only', 'transformer-extractor',
                           'mancini_2022_argmining'},
                     namespace='mamkit',
                     component_class=AudioTransformerExtractor)
    def mmused_acc_mancini_2022_argmining(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = None
        config.aggregate = True

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mode:audio-only', 'transformer', 'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=AudioTransformer)
    @register_method(name='model',
                     tags={'data:mmused', 'task:acc', 'mode:audio-only', 'transformer-extractor',
                           'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=AudioTransformerExtractor)
    def mmused_acc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = 1 / 5
        config.aggregate = False

        return config

    @classmethod
    @register_method(name='model',
                     tags={'data:mmused-fallacy', 'task:afc', 'mode:audio-only', 'transformer', 'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=AudioTransformer)
    @register_method(name='model',
                     tags={'data:mmused-fallacy', 'task:afc', 'mode:audio-only', 'transformer-extractor',
                           'mancini_2024_mamkit'},
                     namespace='mamkit',
                     component_class=AudioTransformerExtractor)
    def mmused_fallacy_afc_mancini_2024_mamkit(
            cls
    ):
        config = cls.default()

        config.downsampling_factor = 1 / 5
        config.aggregate = False

        return config
