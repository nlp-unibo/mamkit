import torch
from transformers import AutoTokenizer, AutoModel


# TODO: add text and audio processing steps


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


class TextTransformerCollator:
    def __init__(
            self,
            model_card,
    ):
        self.model_card = model_card
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_card)
        self.model = AutoModel.from_pretrained(model_card).to(self.device)

    def __call__(
            self,
            text
    ):
        tokenized = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        text_features = self.model(**tokenized).last_hidden_state
        text_attentions = tokenized.attention_mask
        return text_features, text_attentions


# TODO: fix this
class AudioTransformerCollator:

    def __init__(
            self,
            model_card
    ):
        self.model_card = model_card
        self.model = AutoModel.from_pretrained(model_card).to(self.device)

    def __call__(
            self,
            audio
    ):
        pass
