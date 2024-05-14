import torch as th
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoProcessor


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
        texts = pad_sequence(texts, padding_value=0, batch_first=True)
        return texts


class TextTransformerCollator:
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
            text
    ):
        tokenized = self.tokenizer(text,
                                   padding=True,
                                   return_tensors='pt',
                                   **self.tokenizer_args).to(self.device)
        return tokenized['input_ids'], tokenized['attention_mask']


class AudioTransformerCollator:

    def __init__(
            self,
            model_card,
            processor_args=None,
            model_args=None,
    ):
        self.model_card = model_card
        self.processor_args = processor_args if processor_args is not None else {}
        self.model_args = model_args if model_args is not None else {}

        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(model_card)

    def __call__(
            self,
            audio
    ):
        processed = self.processor(audio,
                                   padding=True,
                                   return_tensors='pt',
                                   return_attention_mask=True,
                                   **self.processor_args).to(self.device)
        return processed['input_values'], processed['attention_mask']




class AudioCollator:

    def __call__(
            self,
            features
    ):
        features = [th.tensor(feature_set, dtype=th.float32) for feature_set in features]
        features = pad_sequence(features, batch_first=True, padding_value=float('-inf'))
        features[(features == float('-inf'))] = 0

        if len(features.shape) == 3:
            attention_mask = features[:, :, 0] != float('-inf')
        else:
            attention_mask = th.ones((features.shape[0]), dtype=th.int32)
            features = features[:, None, :]

        return features, attention_mask
