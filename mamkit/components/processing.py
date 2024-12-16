from typing import Dict, Any

from cinnamon.component import Component

from mamkit.components.datasets import UnimodalDataset, MultimodalDataset, MAMDataset, PairUnimodalDataset, \
    PairMultimodalDataset

__all__ = [
    'Processor',
    'ProcessorComponent',
    'UnimodalProcessor',
    'MultimodalProcessor',
    'PairUnimodalProcessor',
    'PairMultimodalProcessor',
]


class Processor(Component):

    def fit(
            self,
            train_data: MAMDataset
    ):
        pass

    def clear(
            self
    ):
        pass

    def __call__(
            self,
            data: MAMDataset
    ):
        return data

    def get_collator_args(
            self
    ) -> Dict[str, Any]:
        pass


class ProcessorComponent(Component):

    def fit(
            self,
            *args,
            **kwargs
    ):
        pass

    def clear(
            self
    ):
        pass

    def __call__(
            self,
            *args,
            **kwargs
    ):
        return args, kwargs

    def get_collator_args(
            self
    ) -> Dict[str, Any]:
        return {}


class UnimodalProcessor(Processor):

    def __init__(
            self,
            features_processor: ProcessorComponent = None,
            label_processor: ProcessorComponent = None,
    ):
        self.features_processor = features_processor
        self.label_processor = label_processor

    def fit(
            self,
            train_data: UnimodalDataset
    ):
        if self.features_processor is not None:
            self.features_processor.fit(train_data.inputs)

        if self.label_processor is not None:
            self.label_processor.fit(train_data.labels)

    def __call__(
            self,
            data: UnimodalDataset
    ):
        if self.features_processor is not None:
            data.inputs = self.features_processor(data.inputs)

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

    def get_collator_args(
            self
    ) -> Dict[str, Any]:
        collator_args = {}
        if self.features_processor is not None:
            collator_args = {**self.features_processor.get_collator_args(), **collator_args}
        if self.label_processor is not None:
            collator_args = {**self.label_processor.get_collator_args(), **collator_args}
        return collator_args


class MultimodalProcessor(Processor):
    def __init__(
            self,
            text_processor: ProcessorComponent = None,
            audio_processor: ProcessorComponent = None,
            label_processor: ProcessorComponent = None
    ):
        self.text_processor = text_processor
        self.audio_processor = audio_processor
        self.label_processor = label_processor

    def fit(
            self,
            train_data: MultimodalDataset
    ):
        if self.text_processor is not None:
            self.text_processor.fit(train_data.texts)

        if self.audio_processor is not None:
            self.audio_processor.fit(train_data.audio)

        if self.label_processor is not None:
            self.label_processor.fit(train_data.labels)

    def __call__(
            self,
            data: MultimodalDataset
    ):
        if self.text_processor is not None:
            data.texts = self.text_processor(data.texts)

        if self.audio_processor is not None:
            data.audio = self.audio_processor(data.audio)

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

    def get_collator_args(
            self
    ) -> Dict[str, Any]:
        collator_args = {}
        if self.text_processor is not None:
            collator_args = {**self.text_processor.get_collator_args(), **collator_args}
        if self.audio_processor is not None:
            collator_args = {**self.audio_processor.get_collator_args(), **collator_args}
        if self.label_processor is not None:
            collator_args = {**self.label_processor.get_collator_args(), **collator_args}
        return collator_args


class PairUnimodalProcessor(Processor):
    def __init__(
            self,
            features_processor: ProcessorComponent = None,
            label_processor: ProcessorComponent = None,
    ):
        self.features_processor = features_processor
        self.label_processor = label_processor

    def fit(
            self,
            train_data: PairUnimodalDataset
    ):
        if self.features_processor is not None:
            self.features_processor.fit(train_data.a_inputs, train_data.b_inputs)

        if self.label_processor is not None:
            self.label_processor.fit(train_data.labels)

    def __call__(
            self,
            data: PairUnimodalDataset
    ):
        if self.features_processor is not None:
            data.a_inputs, data.b_inputs = self.features_processor(data.a_inputs, data.b_inputs)

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

    def get_collator_args(
            self
    ) -> Dict[str, Any]:
        collator_args = {}
        if self.features_processor is not None:
            collator_args = {**self.features_processor.get_collator_args(), **collator_args}
        if self.label_processor is not None:
            collator_args = {**self.label_processor.get_collator_args(), **collator_args}
        return collator_args


class PairMultimodalProcessor(Processor):
    def __init__(
            self,
            text_processor: ProcessorComponent = None,
            audio_processor: ProcessorComponent = None,
            label_processor: ProcessorComponent = None
    ):
        self.text_processor = text_processor
        self.audio_processor = audio_processor
        self.label_processor = label_processor

    def fit(
            self,
            train_data: PairMultimodalDataset
    ):
        if self.text_processor is not None:
            self.text_processor.fit(train_data.a_texts, train_data.b_texts)

        if self.audio_processor is not None:
            self.audio_processor.fit(train_data.a_audio, train_data.b_audio)

        if self.label_processor is not None:
            self.label_processor.fit(train_data.labels)

    def __call__(
            self,
            data: PairMultimodalDataset
    ):
        if self.text_processor is not None:
            data.a_texts, data.b_texts = self.text_processor(data.a_texts, data.b_texts)

        if self.audio_processor is not None:
            data.a_audio, data.b_audio = self.audio_processor(data.a_audio, data.b_audio)

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

    def get_collator_args(
            self
    ) -> Dict[str, Any]:
        collator_args = {}
        if self.text_processor is not None:
            collator_args = {**self.text_processor.get_collator_args(), **collator_args}
        if self.audio_processor is not None:
            collator_args = {**self.audio_processor.get_collator_args(), **collator_args}
        if self.label_processor is not None:
            collator_args = {**self.label_processor.get_collator_args(), **collator_args}
        return collator_args
