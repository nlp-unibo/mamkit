from cinnamon.component import Component
import torch as th

__all__ = [
    'DataCollator',
    'UnimodalCollator',
    'PairUnimodalCollator',
    'MultimodalCollator',
    'PairMultimodalCollator',
    'LabelCollator'
]


class DataCollator(Component):

    def __call__(
            self,
            batch
    ):
        return batch


class UnimodalCollator(DataCollator):
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


class PairUnimodalCollator(UnimodalCollator):

    def __call__(
            self,
            batch
    ):
        a_features, b_features, labels = zip(*batch)
        if self.features_collator is None:
            features_collated = (a_features, b_features)
        else:
            features_collated = self.features_collator((a_features, b_features))

        if self.label_collator is None:
            labels_collated = labels
        else:
            labels_collated = self.label_collator(labels)

        return features_collated, labels_collated


class MultimodalCollator(DataCollator):
    def __init__(
            self,
            text_collator,
            audio_collator,
            label_collator
    ):
        self.text_collator = text_collator
        self.audio_collator = audio_collator
        self.label_collator = label_collator

    def __call__(
            self,
            batch
    ):
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

        return (text_collated, audio_collated), labels_collated


class PairMultimodalCollator(MultimodalCollator):

    def __call__(
            self,
            batch
    ):
        a_text, b_text, a_audio, b_audio, labels = zip(*batch)
        if self.text_collator is None:
            text_collated = (a_text, b_text)
        else:
            text_collated = self.text_collator((a_text, b_text))

        if self.audio_collator is None:
            audio_collated = (a_audio, b_audio)
        else:
            audio_collated = self.audio_collator((a_audio, b_audio))

        if self.label_collator is None:
            labels_collated = labels
        else:
            labels_collated = self.label_collator(labels)

        return (text_collated, audio_collated), labels_collated


class LabelCollator(DataCollator):

    def __call__(
            self,
            batch
    ):
        return th.tensor(batch)