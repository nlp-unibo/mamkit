import torch as th
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from mamkit.components.collators import DataCollator

__all__ = [
    'TextCollator',
    'PairTextCollator',
    'TextTransformerCollator',
    'PairTextTransformerCollator',
    'TextTransformerOutputCollator',
    'PairTextTransformerOutputCollator',
]


class TextCollator(DataCollator):

    def __init__(
            self,
            tokenizer,
            vocab
    ):
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __call__(
            self,
            batch
    ):
        batch = [th.tensor(self.vocab(self.tokenizer(text))) for text in batch]
        batch = pad_sequence(batch, padding_value=0, batch_first=True)
        return batch


class PairTextCollator(TextCollator):

    def __call__(
            self,
            batch
    ):
        a_texts, b_texts = batch
        a_texts = [th.tensor(self.vocab(self.tokenizer(text))) for text in a_texts]
        a_texts = pad_sequence(a_texts, padding_value=0, batch_first=True)

        b_texts = [th.tensor(self.vocab(self.tokenizer(text))) for text in b_texts]
        b_texts = pad_sequence(b_texts, padding_value=0, batch_first=True)

        return a_texts, b_texts


class TextTransformerCollator(DataCollator):
    def __init__(
            self,
            model_card,
            tokenizer_args=None,
    ):
        self.model_card = model_card
        self.tokenizer_args = tokenizer_args if tokenizer_args is not None else {}

        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_card)

    def __call__(
            self,
            batch
    ):
        tokenized = self.tokenizer(batch,
                                   padding=True,
                                   return_tensors='pt',
                                   **self.tokenizer_args).to(self.device)
        return tokenized['input_ids'], tokenized['attention_mask']


class PairTextTransformerCollator(TextTransformerCollator):

    def __call__(
            self,
            batch
    ):
        a_texts, b_texts = batch
        a_tokenized = self.tokenizer(a_texts,
                                     padding=True,
                                     return_tensors='pt',
                                     **self.tokenizer_args).to(self.device)
        b_tokenized = self.tokenizer(b_texts,
                                     padding=True,
                                     return_tensors='pt',
                                     **self.tokenizer_args).to(self.device)
        return (a_tokenized['input_ids'], a_tokenized['attention_mask']), \
            (b_tokenized['input_ids'], b_tokenized['attention_mask'])


class TextTransformerOutputCollator(DataCollator):

    def __call__(
            self,
            batch
    ):
        batch = pad_sequence([th.tensor(text, dtype=th.float32) for text in batch], padding_value=0.0, batch_first=True)
        attention_mask = batch[:, :, 0] != 0.0
        return batch, attention_mask.to(th.float32)


class PairTextTransformerOutputCollator(TextTransformerOutputCollator):

    def __call__(
            self,
            batch
    ):
        a_texts, b_texts = batch

        a_texts = pad_sequence([th.tensor(text, dtype=th.float32) for text in a_texts], padding_value=0.0,
                               batch_first=True)
        a_attention_mask = a_texts[:, :, 0] != 0.0

        b_texts = pad_sequence([th.tensor(text, dtype=th.float32) for text in b_texts], padding_value=0.0,
                               batch_first=True)
        b_attention_mask = b_texts[:, :, 0] != 0.0

        return (a_texts, a_attention_mask.to(th.float32)), (b_texts, b_attention_mask.to(th.float32))


