import torch as th
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import vocab
from transformers import AutoTokenizer, AutoModel


class UnimodalCollator:
    def __init__(
            self,
            features_collator,
            label_collator
    ):
        self.features_collator = features_collator
        self.label_collator = label_collator

    def __call__(
            self,
            batch
    ):
        features_raw, labels = zip(*batch)
        if self.features_collator is None:
            features_collated = features_raw
        else:
            features_collated = self.features_collator(features_raw)

        if self.label_collator is None:
            labels_collated = labels
        else:
            labels_collated = self.label_collator(labels)

        return features_collated, labels_collated


class MultimodalCollator:
    def __init__(
            self,
            text_collator,
            audio_collator,
            label_collator
    ):
        self.text_collator = text_collator
        self.audio_collator = audio_collator
        self.label_collator = label_collator

    def __call__(self, batch):
        text_raw, audio_raw, labels = zip(*batch)
        if self.text_collator is None:
            text_collated = text_raw
        else:
            text_collated = self.text_collator(text_raw)

        if self.audio_collator is None:
            audio_collated = audio_raw
        else:
            audio_collated = self.audio_collator(audio_raw)

        if self.label_collator is None:
            labels_collated = labels
        else:
            labels_collated = self.label_collator(labels)

        return (*text_collated, *audio_collated), labels_collated


class TextCollator:

    def __init__(
            self,
            tokenizer,
            vocab
    ):
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __call__(
            self,
            texts
    ):
        texts = [th.tensor(self.vocab(self.tokenizer(text))) for text in texts]
        return pad_sequence(texts, padding_value=0, batch_first=True)


class TextTransformerCollator:
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_card)
        self.model = AutoModel.from_pretrained(model_card).to(self.device)

    def __call__(
            self,
            text
    ):
        tokenized = self.tokenizer(text,
                                   padding=True,
                                   truncation=True,
                                   return_tensors='pt',
                                   **self.tokenizer_args).to(self.device)
        text_features = self.model(**tokenized, **self.model_args).last_hidden_state
        text_attentions = tokenized.attention_mask
        return text_features, text_attentions
