import logging
from typing import Optional
import torch as th
from torchtext.vocab import pretrained_aliases, build_vocab_from_iterator
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoFeatureExtractor

from mamkit.components.processing import ProcessorComponent

__all__ = [
    'VocabBuilder',
    'PairVocabBuilder',
    'TextTransformer',
    'PairTextTransformer',
]


class VocabBuilder(ProcessorComponent):

    def __init__(
            self,
            tokenizer,
            embedding_dim: int,
            embedding_model: Optional[str] = None,
            tokenization_args=None
    ):
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.embedding_model = pretrained_aliases[embedding_model]() if embedding_model is not None else None
        self.vocab = None
        self.tokenization_args = tokenization_args if tokenization_args is not None else {}
        self.embedding_matrix = None

    def fit(
            self,
            texts
    ):
        logging.info('Building vocabulary...')
        self.vocab = build_vocab_from_iterator(
            iterator=iter([self.tokenizer(text) for text in texts]),
            specials=['<pad>', '<unk>'],
            special_first=True,
            **self.tokenization_args
        )
        self.vocab.set_default_index(self.vocab['<unk>'])

        if self.embedding_model is not None:
            self.embedding_matrix = self.embedding_model.get_vecs_by_tokens(self.vocab.get_itos())

    def __call__(
            self,
            texts
    ):
        return texts

    def clear(
            self
    ):
        self.embedding_model = None

    def reset(
            self
    ):
        self.embedding_matrix = None


class PairVocabBuilder(ProcessorComponent):

    def __init__(
            self,
            tokenizer,
            embedding_dim: int,
            embedding_model: Optional[str] = None,
            tokenization_args=None
    ):
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.embedding_model = pretrained_aliases[embedding_model]() if embedding_model is not None else None
        self.vocab = None
        self.tokenization_args = tokenization_args if tokenization_args is not None else {}
        self.embedding_matrix = None

    def fit(
            self,
            a_texts,
            b_texts
    ):
        logging.info('Building vocabulary...')
        self.vocab = build_vocab_from_iterator(
            iterator=iter([self.tokenizer(text) for text in list(a_texts) + list(b_texts)]),
            specials=['<pad>', '<unk>'],
            special_first=True,
            **self.tokenization_args
        )
        self.vocab.set_default_index(self.vocab['<unk>'])

        if self.embedding_model is not None:
            self.embedding_matrix = self.embedding_model.get_vecs_by_tokens(self.vocab.get_itos())

    def __call__(
            self,
            a_texts,
            b_texts
    ):
        return a_texts, b_texts

    def clear(
            self
    ):
        self.embedding_model = None

    def reset(
            self
    ):
        self.embedding_matrix = None


class TextTransformer(ProcessorComponent):

    def __init__(
            self,
            model_card,
            tokenizer_args=None,
            model_args=None,
    ):
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
            texts
    ):
        if self.model is None:
            self._init_models()

        text_features = []
        with th.inference_mode():
            for text in tqdm(texts, desc='Encoding text...'):
                tokenized = self.tokenizer([text],
                                           padding=True,
                                           return_tensors='pt',
                                           **self.tokenizer_args).to(self.device)
                model_output = self.model(**tokenized, **self.model_args)
                text_emb = model_output.last_hidden_state * tokenized.attention_mask[:, :, None]
                text_emb = text_emb.detach().cpu().numpy()[0]
                text_features.append(text_emb)
        return text_features

    def clear(
            self
    ):
        self.tokenizer = None
        self.model = None
        th.cuda.empty_cache()


class PairTextTransformer(ProcessorComponent):

    def __init__(
            self,
            model_card,
            tokenizer_args=None,
            model_args=None,
    ):
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
            a_texts,
            b_texts
    ):
        if self.model is None:
            self._init_models()

        a_text_features, b_text_features = [], []
        with th.inference_mode():
            for a_text, b_text in tqdm(zip(a_texts, b_texts), desc='Encoding text...'):
                tokenized = self.tokenizer([a_text, b_text],
                                           padding=True,
                                           return_tensors='pt',
                                           **self.tokenizer_args).to(self.device)
                model_output = self.model(**tokenized, **self.model_args)
                text_emb = model_output.last_hidden_state * tokenized.attention_mask[:, :, None]
                text_emb = text_emb.detach().cpu().numpy()
                a_text_features.append(text_emb[0])
                b_text_features.append(text_emb[1])

        return a_text_features, b_text_features

    def clear(
            self
    ):
        self.tokenizer = None
        self.model = None
        th.cuda.empty_cache()

