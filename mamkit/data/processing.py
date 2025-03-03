import logging
import pickle
from pathlib import Path
from typing import Optional, Iterable, Dict, List, Union, Any

import torch as th
from torchtext.vocab import pretrained_aliases, build_vocab_from_iterator
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoFeatureExtractor

from mamkit.data.datasets import UnimodalDataset, MultimodalDataset, MAMDataset, PairUnimodalDataset, \
    PairMultimodalDataset
from mamkit.utility.processing import encode_audio_and_context_mfcc, encode_audio_and_context_nn, \
    encode_text_and_context_nn


class MAMProcessor:
    """
    Base MAM processor interface
    """

    def fit(
            self,
            train_data: MAMDataset
    ):
        """
        Fits on training MAM data
        """
        pass

    def clear(
            self
    ):
        pass

    def reset(
            self
    ):
        """
        Clears the processor internal state.
        """
        pass

    def __call__(
            self,
            data: MAMDataset
    ):
        return data


class ProcessorComponent:
    """
    Base processor interface
    """

    def fit(
            self,
            inputs: Any,
            context: Any = None
    ):
        """
        Fits on input and context features.

        Args:
            inputs: any input feature
            context: any context features associated with inputs
        """
        pass

    def clear(
            self
    ):
        pass

    def reset(
            self
    ):
        """
        Resets the processor internal state.
        """
        pass

    def __call__(
            self,
            inputs: Any,
            context: Any = None
    ):
        return inputs, context


class PairProcessorComponent:
    """
    Pair base processor interface
    """

    def fit(
            self,
            a_inputs: Any,
            b_inputs: Any,
            a_context: Any = None,
            b_context: Any = None
    ):
        """
        Fits on input and context features.

        Args:
            a_inputs: any input feature corresponding to input A
            b_inputs: any input feature corresponding to input B
            a_context: any context features associated with input A
            b_context: any context features associated with input B
        """
        pass

    def clear(
            self
    ):
        pass

    def reset(
            self
    ):
        pass

    def __call__(
            self,
            a_inputs: Any,
            b_inputs: Any,
            a_context: Any = None,
            b_context: Any = None
    ):
        return a_inputs, b_inputs, a_context, b_context


class UnimodalProcessor(MAMProcessor):
    """
    Unimodal processor interface supporting a features processor and a label processor.
    """

    def __init__(
            self,
            features_processor: ProcessorComponent = None,
            label_processor: ProcessorComponent = None,
    ):
        """
        Args:
            features_processor: any feature processor
            label_processor: any label processor
        """

        self.features_processor = features_processor
        self.label_processor = label_processor

    def fit(
            self,
            train_data: UnimodalDataset
    ):
        if self.features_processor is not None:
            self.features_processor.fit(inputs=train_data.inputs, context=train_data.context)

        if self.label_processor is not None:
            self.label_processor.fit(train_data.labels)

    def __call__(
            self,
            data: UnimodalDataset
    ):
        if self.features_processor is not None:
            data.inputs, data.context = self.features_processor(inputs=data.inputs, context=data.context)

        if self.label_processor is not None:
            data.labels = self.label_processor(data.labels)

        return data

    def clear(
            self
    ):
        if self.features_processor is not None:
            self.features_processor.clear()

        if self.label_processor is not None:
            self.label_processor.clear()


class PairUnimodalProcessor(MAMProcessor):
    """
    Pair unimodal processor interface supporting a features processor and a label processor.
    """

    def __init__(
            self,
            features_processor: PairProcessorComponent = None,
            label_processor: ProcessorComponent = None,
    ):
        """
        Args:
            features_processor: any feature processor
            label_processor: any label processor
        """

        self.features_processor = features_processor
        self.label_processor = label_processor

    def fit(
            self,
            train_data: PairUnimodalDataset
    ):
        if self.features_processor is not None:
            self.features_processor.fit(a_inputs=train_data.a_inputs,
                                        b_inputs=train_data.b_inputs,
                                        a_context=train_data.a_context,
                                        b_context=train_data.b_context)

        if self.label_processor is not None:
            self.label_processor.fit(train_data.labels)

    def __call__(
            self,
            data: PairUnimodalDataset
    ):
        if self.features_processor is not None:
            data.a_inputs, data.b_inputs, \
                data.a_context, data.b_context = self.features_processor(a_inputs=data.a_inputs,
                                                                         b_inputs=data.b_inputs,
                                                                         a_context=data.a_context,
                                                                         b_context=data.b_context)

        if self.label_processor is not None:
            data.labels = self.label_processor(data.labels)

        return data

    def clear(
            self
    ):
        if self.features_processor is not None:
            self.features_processor.clear()

        if self.label_processor is not None:
            self.label_processor.clear()


class MultimodalProcessor(MAMProcessor):
    """
    Multimodal processor interface supporting a text processor, an audio processor, and a label processor.
    """

    def __init__(
            self,
            text_processor: ProcessorComponent = None,
            audio_processor: ProcessorComponent = None,
            label_processor: ProcessorComponent = None
    ):
        """
        Args:
            text_processor: any text feature processor
            audio_processor: any audio feature processor
            label_processor: any label processor
        """

        self.text_processor = text_processor
        self.audio_processor = audio_processor
        self.label_processor = label_processor

    def fit(
            self,
            train_data: MultimodalDataset
    ):
        if self.text_processor is not None:
            self.text_processor.fit(inputs=train_data.texts, context=train_data.text_context)

        if self.audio_processor is not None:
            self.audio_processor.fit(inputs=train_data.audio, context=train_data.audio_context)

        if self.label_processor is not None:
            self.label_processor.fit(train_data.labels)

    def __call__(
            self,
            data: MultimodalDataset
    ):
        if self.text_processor is not None:
            data.texts, data.text_context = self.text_processor(inputs=data.texts, context=data.text_context)

        if self.audio_processor is not None:
            data.audio, data.audio_context = self.audio_processor(inputs=data.audio, context=data.audio_context)

        if self.label_processor is not None:
            data.labels = self.label_processor(data.labels)

        return data

    def clear(
            self
    ):
        if self.text_processor is not None:
            self.text_processor.clear()

        if self.audio_processor is not None:
            self.audio_processor.clear()

        if self.label_processor is not None:
            self.label_processor.clear()


class PairMultimodalProcessor(MAMProcessor):
    """
    Pair multimodal processor interface supporting a text processor, an audio processor, and a label processor.
    """

    def __init__(
            self,
            text_processor: PairProcessorComponent = None,
            audio_processor: PairProcessorComponent = None,
            label_processor: ProcessorComponent = None
    ):
        """
        Args:
            text_processor: any text feature processor
            audio_processor: any audio feature processor
            label_processor: any label processor
        """

        self.text_processor = text_processor
        self.audio_processor = audio_processor
        self.label_processor = label_processor

    def fit(
            self,
            train_data: PairMultimodalDataset
    ):
        if self.text_processor is not None:
            self.text_processor.fit(a_inputs=train_data.a_texts, b_inputs=train_data.b_texts,
                                    a_context=train_data.a_text_context, b_context=train_data.b_text_context)

        if self.audio_processor is not None:
            self.audio_processor.fit(a_inputs=train_data.a_audio, b_inputs=train_data.b_audio,
                                     a_context=train_data.a_audio_context, b_context=train_data.b_audio_context)

        if self.label_processor is not None:
            self.label_processor.fit(train_data.labels)

    def __call__(
            self,
            data: PairMultimodalDataset
    ):
        if self.text_processor is not None:
            data.a_texts, data.b_texts, \
                data.a_text_context, data.b_text_context = self.text_processor(a_inputs=data.a_texts,
                                                                               b_inputs=data.b_texts,
                                                                               a_context=data.a_text_context,
                                                                               b_context=data.b_text_context)

        if self.audio_processor is not None:
            data.a_audio, data.b_audio, \
                data.a_audio_context, data.b_audio_context = self.audio_processor(a_inputs=data.a_audio,
                                                                                  b_inputs=data.b_audio,
                                                                                  a_context=data.a_audio_context,
                                                                                  b_context=data.b_audio_context)

        if self.label_processor is not None:
            data.labels = self.label_processor(data.labels)

        return data

    def clear(
            self
    ):
        if self.text_processor is not None:
            self.text_processor.clear()

        if self.audio_processor is not None:
            self.audio_processor.clear()

        if self.label_processor is not None:
            self.label_processor.clear()


class VocabBuilder(ProcessorComponent):
    """
    Torch text processor that uses a tokenizer to build a vocabulary from text.
    """

    def __init__(
            self,
            tokenizer,
            embedding_dim: int,
            embedding_model: Optional[str] = None,
            tokenization_args=None
    ):
        """
        Args:
            tokenizer: any torchtext tokenizer
            embedding_dim: embedding dimension for pre-trained vocabs
            embedding_model: pre-trained embedding model, if any
            tokenization_args: any additional tokenizer specific arguments, if any
        """

        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.embedding_model = pretrained_aliases[embedding_model]() if embedding_model is not None else None
        self.vocab = None
        self.tokenization_args = tokenization_args if tokenization_args is not None else {}
        self.embedding_matrix = None

    def fit(
            self,
            inputs: List[str],
            context: List[str] = None
    ):
        """
        Fits tokenizer to build a vocabulary based on input texts and contexts, if any

        Args:
            inputs: input texts to tokenize and encode
            context: context texts to tokenize and encode
        """

        logging.info('Building vocabulary...')
        vocab_input = inputs if context is None else set(inputs + context)
        self.vocab = build_vocab_from_iterator(
            iterator=iter([self.tokenizer(text) for text in vocab_input]),
            specials=['<pad>', '<unk>'],
            special_first=True,
            **self.tokenization_args
        )
        self.vocab.set_default_index(self.vocab['<unk>'])

        if self.embedding_model is not None:
            self.embedding_matrix = self.embedding_model.get_vecs_by_tokens(self.vocab.get_itos())

    def __call__(
            self,
            inputs: List[str],
            context: List[str] = None
    ):
        return inputs, context

    def clear(
            self
    ):
        self.embedding_model = None

    def reset(
            self
    ):
        self.embedding_matrix = None


class PairVocabBuilder(PairProcessorComponent):
    """
    Pair torch text processor that uses a tokenizer to build a vocabulary from text.
    """

    def __init__(
            self,
            tokenizer,
            embedding_dim: int,
            embedding_model: Optional[str] = None,
            tokenization_args=None
    ):
        """
        Args:
            tokenizer: any torchtext tokenizer
            embedding_dim: embedding dimension for pre-trained vocabs
            embedding_model: pre-trained embedding model, if any
            tokenization_args: any additional tokenizer specific arguments, if any
        """

        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.embedding_model = pretrained_aliases[embedding_model]() if embedding_model is not None else None
        self.vocab = None
        self.tokenization_args = tokenization_args if tokenization_args is not None else {}
        self.embedding_matrix = None

    def fit(
            self,
            a_inputs: List[str],
            b_inputs: List[str],
            a_context: List[str] = None,
            b_context: List[str] = None
    ):
        """
        Fits tokenizer to build a vocabulary based on input texts and contexts, if any

        Args:
            a_inputs: input texts to tokenize and encode corresponding to input A
            b_inputs: input texts to tokenize and encode corresponding to input B
            a_context: context texts to tokenize and encode associated with input A
            b_context: context texts to tokenize and encode associated with input B
        """

        logging.info('Building vocabulary...')

        vocab_input = a_inputs + b_inputs
        if a_context is not None:
            vocab_input += a_context
        if b_context is not None:
            vocab_input += b_context

        vocab_input = set(vocab_input)
        self.vocab = build_vocab_from_iterator(
            iterator=iter([self.tokenizer(text) for text in vocab_input]),
            specials=['<pad>', '<unk>'],
            special_first=True,
            **self.tokenization_args
        )
        self.vocab.set_default_index(self.vocab['<unk>'])

        if self.embedding_model is not None:
            self.embedding_matrix = self.embedding_model.get_vecs_by_tokens(self.vocab.get_itos())

    def __call__(
            self,
            a_inputs: List[str],
            b_inputs: List[str],
            a_context: List[str] = None,
            b_context: List[str] = None
    ):
        return a_inputs, b_inputs, a_context, b_context

    def clear(
            self
    ):
        self.embedding_model = None

    def reset(
            self
    ):
        self.embedding_matrix = None


class MFCCExtractor(ProcessorComponent):
    """
    Torch MFCC audio processor
    """

    def __init__(
            self,
            mfccs: int,
            sampling_rate=16000,
            pooling_sizes: Optional[Iterable[int]] = None,
            remove_energy: bool = True,
            normalize: bool = True,
            serialization_path: Path = None
    ):
        """
        Args:
            mfccs: number of MFCCs to extract
            sampling_rate: audio sampling rate
            pooling_sizes: where to apply (nested) mean pooling to reduce the number of audio frames
            remove_energy: whether to remove energy MFCC
            normalize: whether to normalize MFCCs or not
            serialization_path: where to store computed MFCCs to speed up future processing
        """

        self.mfccs = mfccs
        self.sampling_rate = sampling_rate
        self.pooling_sizes = pooling_sizes
        self.remove_energy = remove_energy
        self.normalize = normalize
        self.serialization_path = serialization_path if serialization_path is not None else Path('mfccs.pkl')
        self.preloaded_mfccs = {}

        if self.serialization_path.exists():
            with self.serialization_path.open('rb') as f:
                self.preloaded_mfccs: Dict = pickle.load(f)

    def __call__(
            self,
            inputs: List[Union[Path, List[Path]]],
            context: List[List[Path]] = None
    ):
        preloaded_length = len(self.preloaded_mfccs)

        input_context = context if context is not None else [None] * len(inputs)

        input_features, context_features = [], []
        for audio_input, audio_context in tqdm(zip(inputs, input_context),
                                               desc='Extracting MFCCs',
                                               total=len(inputs)):
            audio_features, context_audio_features = encode_audio_and_context_mfcc(audio_input=audio_input,
                                                                                   audio_context=audio_context,
                                                                                   preloaded_mfccs=self.preloaded_mfccs,
                                                                                   mfccs=self.mfccs,
                                                                                   normalize=self.normalize,
                                                                                   pooling_sizes=self.pooling_sizes,
                                                                                   remove_energy=self.remove_energy,
                                                                                   sampling_rate=self.sampling_rate)

            input_features.append(audio_features)
            if context_audio_features is not None:
                context_features.append(context_audio_features)

        if len(self.preloaded_mfccs) != preloaded_length:
            with self.serialization_path.open('wb') as f:
                pickle.dump(self.preloaded_mfccs, f)

        return input_features, context_features if len(context_features) else None


class PairMFCCExtractor(PairProcessorComponent):
    """
    Torch MFCC audio processor
    """

    def __init__(
            self,
            mfccs: int,
            sampling_rate=16000,
            pooling_sizes: Optional[Iterable[int]] = None,
            remove_energy: bool = True,
            normalize: bool = True,
            serialization_path: Path = None
    ):
        """
        Args:
            mfccs: number of MFCCs to extract
            sampling_rate: audio sampling rate
            pooling_sizes: where to apply (nested) mean pooling to reduce the number of audio frames
            remove_energy: whether to remove energy MFCC
            normalize: whether to normalize MFCCs or not
            serialization_path: where to store computed MFCCs to speed up future processing
        """

        self.mfccs = mfccs
        self.sampling_rate = sampling_rate
        self.pooling_sizes = pooling_sizes
        self.remove_energy = remove_energy
        self.normalize = normalize
        self.serialization_path = serialization_path if serialization_path is not None else Path('mfccs.pkl')
        self.preloaded_mfccs = {}

        if self.serialization_path.exists():
            with self.serialization_path.open('rb') as f:
                self.preloaded_mfccs: Dict = pickle.load(f)

    def __call__(
            self,
            a_inputs: List[Union[Path, List[Path]]],
            b_inputs: List[Union[Path, List[Path]]],
            a_context: List[List[Path]] = None,
            b_context: List[List[Path]] = None
    ):
        preloaded_length = len(self.preloaded_mfccs)

        a_input_context = a_context if a_context is not None else [None] * len(a_inputs)
        b_input_context = b_context if b_context is not None else [None] * len(b_inputs)

        a_features, b_features, a_context_features, b_context_features = [], [], [], []
        for a_audio_input, b_audio_input, \
                a_audio_context, b_audio_context in tqdm(zip(a_inputs, b_inputs, a_input_context, b_input_context),
                                                         desc='Extracting MFCCs'):
            a_audio_features, \
                a_context_audio_features = encode_audio_and_context_mfcc(audio_input=a_audio_input,
                                                                         audio_context=a_audio_context,
                                                                         preloaded_mfccs=self.preloaded_mfccs,
                                                                         mfccs=self.mfccs,
                                                                         normalize=self.normalize,
                                                                         pooling_sizes=self.pooling_sizes,
                                                                         remove_energy=self.remove_energy,
                                                                         sampling_rate=self.sampling_rate)
            b_audio_features, \
                b_context_audio_features = encode_audio_and_context_mfcc(audio_input=b_audio_input,
                                                                         audio_context=b_audio_context,
                                                                         preloaded_mfccs=self.preloaded_mfccs,
                                                                         mfccs=self.mfccs,
                                                                         normalize=self.normalize,
                                                                         pooling_sizes=self.pooling_sizes,
                                                                         remove_energy=self.remove_energy,
                                                                         sampling_rate=self.sampling_rate)

            a_features.append(a_audio_features)
            b_features.append(b_audio_features)

            if a_context_audio_features is not None:
                a_context_features.append(a_context_audio_features)

            if b_context_audio_features is not None:
                b_context_features.append(b_context_audio_features)

        if len(self.preloaded_mfccs) != preloaded_length:
            with self.serialization_path.open('wb') as f:
                pickle.dump(self.preloaded_mfccs, f)

        a_context_features = a_context_features if len(a_context_features) else None
        b_context_features = b_context_features if len(b_context_features) else None

        return a_features, b_features, a_context_features, b_context_features


class TextTransformer(ProcessorComponent):
    """
    Text transformer processor
    """

    def __init__(
            self,
            model_card: str,
            tokenizer_args: Optional[Dict] = None,
            model_args: Optional[Dict] = None,
    ):
        """
        Args:
            model_card: any huggingface model card
            tokenizer_args: any tokenizer-specific additional arguments
            model_args: any model-specific additional arguments for encoding inputs
        """

        self.model_card = model_card
        self.tokenizer_args = tokenizer_args if tokenizer_args is not None else {}
        self.model_args = model_args if model_args is not None else {}
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None

    def _init_models(
            self
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        self.model = AutoModel.from_pretrained(self.model_card).to(self.device)

    def __call__(
            self,
            inputs: List[str],
            context: List[str] = None
    ):
        if self.model is None:
            self._init_models()

        context_input = context if context is not None else [None] * len(inputs)

        text_features, context_features = [], []
        with th.inference_mode():
            for text, context in tqdm(zip(inputs, context_input), desc='Encoding text...', total=len(inputs)):
                text_emb, context_emb = encode_text_and_context_nn(text=text,
                                                                   context=context,
                                                                   model=self.model,
                                                                   tokenizer=self.tokenizer,
                                                                   model_args=self.model_args,
                                                                   tokenizer_args=self.tokenizer_args,
                                                                   device=self.device)

                text_features.append(text_emb)
                if context_emb is not None:
                    context_features.append(context_emb)

        return text_features, context_features if len(context_features) else None

    def clear(
            self
    ):
        self.tokenizer = None
        self.model = None
        th.cuda.empty_cache()


class PairTextTransformer(PairProcessorComponent):
    """
    Pair text transformer processor
    """

    def __init__(
            self,
            model_card,
            tokenizer_args=None,
            model_args=None,
    ):
        """
        Args:
            model_card: any huggingface model card
            tokenizer_args: any tokenizer-specific additional arguments
            model_args: any model-specific additional arguments for encoding inputs
        """

        self.model_card = model_card
        self.tokenizer_args = tokenizer_args if tokenizer_args is not None else {}
        self.model_args = model_args if model_args is not None else {}
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None

    def _init_models(
            self
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        self.model = AutoModel.from_pretrained(self.model_card).to(self.device)

    def __call__(
            self,
            a_inputs: List[str],
            b_inputs: List[str],
            a_context: List[str] = None,
            b_context: List[str] = None
    ):
        if self.model is None:
            self._init_models()

        a_input_context = a_context if a_context is not None else [None] * len(a_inputs)
        b_input_context = b_context if b_context is not None else [None] * len(b_inputs)

        a_text_features, b_text_features, a_context_features, b_context_features = [], [], [], []
        with th.inference_mode():
            for a_text, b_text, \
                    a_context, b_context in tqdm(zip(a_inputs, b_inputs,
                                                     a_input_context, b_input_context),
                                                 desc='Encoding text...',
                                                 total=len(a_inputs)):
                a_text_emb, a_context_emb = encode_text_and_context_nn(text=a_text,
                                                                       context=a_context,
                                                                       model=self.model,
                                                                       tokenizer=self.tokenizer,
                                                                       model_args=self.model_args,
                                                                       tokenizer_args=self.tokenizer_args,
                                                                       device=self.device)
                a_text_features.append(a_text_emb)
                if a_context_emb is not None:
                    a_context_features.append(a_context_emb)

                b_text_emb, b_context_emb = encode_text_and_context_nn(text=b_text,
                                                                       context=b_context,
                                                                       model=self.model,
                                                                       tokenizer=self.tokenizer,
                                                                       model_args=self.model_args,
                                                                       tokenizer_args=self.tokenizer_args,
                                                                       device=self.device)
                b_text_features.append(b_text_emb)
                if b_context_emb is not None:
                    b_context_features.append(b_context_emb)

        a_context_features = a_context_features if len(a_context_features) else None
        b_context_features = b_context_features if len(b_context_features) else None

        return a_text_features, b_text_features, a_context_features, b_context_features

    def clear(
            self
    ):
        self.tokenizer = None
        self.model = None
        th.cuda.empty_cache()


class AudioTransformer(ProcessorComponent):
    """
    Audio transformer processor
    """

    def __init__(
            self,
            model_card,
            sampling_rate,
            downsampling_factor=None,
            aggregate: bool = False,
            processor_args=None,
            model_args=None
    ):
        """
        Args:
            model_card: any huggingface model card
            sampling_rate: audio sampling rate
            downsampling_factor: whether to perform audio downsampling to reduce the number of frames
            aggregate: whether to compute mean audio feature vector
            processor_args: any audio processor-specific additional arguments
            model_args: any model-specific additional arguments for encoding inputs
        """

        self.model_card = model_card
        self.sampling_rate = sampling_rate
        self.downsampling_factor = downsampling_factor
        self.aggregate = aggregate
        self.processor_args = processor_args if processor_args is not None else {}
        self.model_args = model_args if model_args is not None else {}
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.processor = None
        self.model = None

    def _init_models(
            self
    ):
        self.processor = AutoProcessor.from_pretrained(self.model_card)
        self.model = AutoModel.from_pretrained(self.model_card).to(self.device)

    def __call__(
            self,
            inputs: List[Union[Path, List[Path]]],
            context: List[List[Path]] = None
    ):
        if self.model is None:
            self._init_models()

        input_context = context if context is not None else [None] * len(inputs)

        input_features, context_features = [], []
        for audio_input, audio_context in tqdm(zip(inputs, input_context),
                                               desc='Extracting Audio Features...',
                                               total=len(inputs)):
            audio_features, \
                context_audio_features = encode_audio_and_context_nn(audio_input=audio_input,
                                                                     audio_context=audio_context,
                                                                     model=self.model,
                                                                     processor=self.processor,
                                                                     model_args=self.model_args,
                                                                     processor_args=self.processor_args,
                                                                     sampling_rate=self.sampling_rate,
                                                                     aggregate=self.aggregate,
                                                                     downsampling_factor=self.downsampling_factor,
                                                                     device=self.device)
            input_features.append(audio_features)
            if context_audio_features is not None:
                context_features.append(context_audio_features)

        return input_features, context_features if len(context_features) else None

    def clear(
            self
    ):
        self.model = None
        self.processor = None
        th.cuda.empty_cache()


class PairAudioTransformer(PairProcessorComponent):
    """
    Pair audio transformer processor
    """

    def __init__(
            self,
            model_card,
            sampling_rate,
            downsampling_factor=None,
            aggregate: bool = False,
            processor_args=None,
            model_args=None
    ):
        """
        Args:
            model_card: any huggingface model card
            sampling_rate: audio sampling rate
            downsampling_factor: whether to perform audio downsampling to reduce the number of frames
            aggregate: whether to compute mean audio feature vector
            processor_args: any audio processor-specific additional arguments
            model_args: any model-specific additional arguments for encoding inputs
        """

        self.model_card = model_card
        self.sampling_rate = sampling_rate
        self.downsampling_factor = downsampling_factor
        self.aggregate = aggregate
        self.processor_args = processor_args if processor_args is not None else {}
        self.model_args = model_args if model_args is not None else {}
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.processor = None
        self.model = None

    def _init_models(
            self
    ):
        self.processor = AutoProcessor.from_pretrained(self.model_card)
        self.model = AutoModel.from_pretrained(self.model_card).to(self.device)

    def __call__(
            self,
            a_inputs: List[Union[Path, List[Path]]],
            b_inputs: List[Union[Path, List[Path]]],
            a_context: List[List[Path]] = None,
            b_context: List[List[Path]] = None
    ):
        if self.model is None:
            self._init_models()

        a_input_context = a_context if a_context is not None else [None] * len(a_inputs)
        b_input_context = b_context if b_context is not None else [None] * len(b_inputs)

        a_features, b_features, a_context_features, b_context_features = [], [], [], []
        for a_audio_input, b_audio_input, \
                a_audio_context, b_audio_context in tqdm(zip(a_inputs, b_inputs, a_input_context, b_input_context),
                                                         desc='Extracting Audio features...',
                                                         total=len(a_inputs)):
            a_audio_features, \
                a_context_audio_features = encode_audio_and_context_nn(audio_input=a_audio_input,
                                                                       model=self.model,
                                                                       processor=self.processor,
                                                                       model_args=self.model_args,
                                                                       processor_args=self.processor_args,
                                                                       sampling_rate=self.sampling_rate,
                                                                       aggregate=self.aggregate,
                                                                       downsampling_factor=self.downsampling_factor,
                                                                       device=self.device)
            b_audio_features, \
                b_context_audio_features = encode_audio_and_context_nn(audio_input=a_audio_input,
                                                                       model=self.model,
                                                                       processor=self.processor,
                                                                       model_args=self.model_args,
                                                                       processor_args=self.processor_args,
                                                                       sampling_rate=self.sampling_rate,
                                                                       aggregate=self.aggregate,
                                                                       downsampling_factor=self.downsampling_factor,
                                                                       device=self.device)

            a_features.append(a_audio_features)
            b_features.append(b_audio_features)

            if a_context_audio_features is not None:
                a_context_features.append(a_context_audio_features)

            if b_context_audio_features is not None:
                b_context_features.append(b_context_audio_features)

        a_context_features = a_context_features if len(a_context_features) else None
        b_context_features = b_context_features if len(b_context_features) else None

        return a_features, b_features, a_context_features, b_context_features

    def clear(
            self
    ):
        self.model = None
        self.processor = None
        th.cuda.empty_cache()


class AudioTransformerExtractor(AudioTransformer):
    """
    Audio transformer extractor
    """

    def _init_models(
            self
    ):
        self.processor = AutoFeatureExtractor.from_pretrained(self.model_card)
        self.model = AutoModel.from_pretrained(self.model_card).to(self.device)


class PairAudioTransformerExtractor(PairAudioTransformer):
    """
    Pair audio transformer extractor
    """

    def _init_models(
            self
    ):
        self.processor = AutoFeatureExtractor.from_pretrained(self.model_card)
        self.model = AutoModel.from_pretrained(self.model_card).to(self.device)
