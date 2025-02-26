from typing import Any

import torch as th
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from mamkit.utility.collators import (
    encode_text_with_context_torch,
    encode_text_with_context_transformer,
    encode_text_with_context_transformer_output,
    encode_audio_with_context_torch
)


class CollatorComponent:

    def __call__(
            self,
            inputs: Any,
            context: Any = None
    ):
        return {'inputs': inputs, 'context': context}


class PairCollatorComponent:

    def __call__(
            self,
            a_inputs: Any,
            b_inputs: Any,
            a_context: Any = None,
            b_context: Any = None
    ):
        return {'a_inputs': a_inputs, 'b_inputs': b_inputs, 'a_context': a_context, 'b_context': b_context}


class UnimodalCollator:

    def __init__(
            self,
            features_collator: CollatorComponent,
            label_collator: CollatorComponent
    ):
        self.features_collator = features_collator
        self.label_collator = label_collator

    def __call__(
            self,
            batch
    ):
        inputs, labels, context = zip(*batch)
        if self.features_collator is None:
            features_collated = {'inputs': inputs, 'context': context}
        else:
            features_collated = self.features_collator(inputs=inputs, context=context)

        if self.label_collator is None:
            labels_collated = labels
        else:
            labels_collated = self.label_collator(labels)

        return features_collated, labels_collated


class PairUnimodalCollator:

    def __init__(
            self,
            features_collator: PairCollatorComponent,
            label_collator: CollatorComponent
    ):
        self.features_collator = features_collator
        self.label_collator = label_collator

    def __call__(
            self,
            batch
    ):
        a_features, b_features, labels, a_context, b_context = zip(*batch)
        if self.features_collator is None:
            features_collated = {'a_inputs': a_features, 'b_inputs': b_features,
                                 'a_context': a_context, 'b_context': b_context}
        else:
            features_collated = self.features_collator(a_inputs=a_features,
                                                       b_inputs=b_features,
                                                       a_context=a_context,
                                                       b_context=b_context)

        if self.label_collator is None:
            labels_collated = labels
        else:
            labels_collated = self.label_collator(labels)

        return features_collated, labels_collated


class MultimodalCollator:

    def __init__(
            self,
            text_collator: CollatorComponent,
            audio_collator: CollatorComponent,
            label_collator: CollatorComponent
    ):
        self.text_collator = text_collator
        self.audio_collator = audio_collator
        self.label_collator = label_collator

    def __call__(
            self,
            batch
    ):
        text_raw, audio_raw, labels, text_context, audio_context = zip(*batch)
        if self.text_collator is None:
            text_collated = {'inputs': text_raw, 'context': text_context}
        else:
            text_collated = self.text_collator(inputs=text_raw, context=text_context)

        if self.audio_collator is None:
            audio_collated = {'inputs': audio_raw, 'context': audio_context}
        else:
            audio_collated = self.audio_collator(inputs=audio_raw, context=audio_context)

        if self.label_collator is None:
            labels_collated = labels
        else:
            labels_collated = self.label_collator(labels)

        return {**{f'text_{key}': value for key, value in text_collated.items()},
                **{f'audio_{key}': value for key, value in audio_collated.items()}}, labels_collated


class PairMultimodalCollator:

    def __init__(
            self,
            text_collator: PairCollatorComponent,
            audio_collator: PairCollatorComponent,
            label_collator: CollatorComponent
    ):
        self.text_collator = text_collator
        self.audio_collator = audio_collator
        self.label_collator = label_collator

    def __call__(
            self,
            batch
    ):
        (a_text, b_text, a_audio, b_audio,
         labels,
         a_text_context, a_audio_context, b_text_context, b_audio_context) = zip(*batch)

        if self.text_collator is None:
            text_collated = {'a_inputs': a_text, 'b_inputs': b_text,
                             'a_context': a_text_context, 'b_context': b_text_context}
        else:
            text_collated = self.text_collator(a_inputs=a_text, b_inputs=b_text,
                                               a_context=a_text_context, b_context=b_text_context)

        if self.audio_collator is None:
            audio_collated = {'a_inputs': a_audio, 'b_inputs': b_audio,
                              'a_context': a_audio_context, 'b_context': b_audio_context}
        else:
            audio_collated = self.audio_collator(a_inputs=a_audio, b_inputs=b_audio,
                                                 a_context=a_audio_context, b_context=b_audio_context)

        if self.label_collator is None:
            labels_collated = labels
        else:
            labels_collated = self.label_collator(labels)

        return {**{f'text_{key}': value for key, value in text_collated.items()},
                **{f'audio_{key}': value for key, value in audio_collated.items()}}, labels_collated


class TextCollator(CollatorComponent):

    def __init__(
            self,
            tokenizer,
            vocab
    ):
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __call__(
            self,
            inputs,
            context=None
    ):
        texts, context = encode_text_with_context_torch(inputs=inputs,
                                                        context=context,
                                                        vocab=self.vocab,
                                                        tokenizer=self.tokenizer)
        return {'inputs': texts, 'context': context}


class PairTextCollator(PairCollatorComponent):

    def __init__(
            self,
            tokenizer,
            vocab
    ):
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __call__(
            self,
            a_inputs,
            b_inputs,
            a_context=None,
            b_context=None
    ):
        a_texts, a_context = encode_text_with_context_torch(inputs=a_inputs,
                                                            context=a_context,
                                                            vocab=self.vocab,
                                                            tokenizer=self.tokenizer)

        b_texts, b_context = encode_text_with_context_torch(inputs=b_inputs,
                                                            context=b_context,
                                                            vocab=self.vocab,
                                                            tokenizer=self.tokenizer)

        return {'a_inputs': a_texts, 'b_inputs': b_texts, 'a_context': a_context, 'b_context': b_context}


class TextTransformerCollator(CollatorComponent):

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
            inputs,
            context=None
    ):
        (input_ids, attention_mask,
         context_ids, context_mask) = encode_text_with_context_transformer(inputs=inputs,
                                                                           context=context,
                                                                           tokenizer=self.tokenizer,
                                                                           tokenizer_args=self.tokenizer_args,
                                                                           device=self.device)

        return {'inputs': input_ids, 'input_mask': attention_mask,
                'context': context_ids, 'context_mask': context_mask}


class PairTextTransformerCollator(PairCollatorComponent):

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
            a_inputs,
            b_inputs,
            a_context=None,
            b_context=None
    ):
        (a_input_ids, a_mask,
         a_context_ids, a_context_mask) = encode_text_with_context_transformer(inputs=a_inputs,
                                                                               context=a_context,
                                                                               tokenizer=self.tokenizer,
                                                                               tokenizer_args=self.tokenizer_args,
                                                                               device=self.device)

        (b_input_ids, b_mask,
         b_context_ids, b_context_mask) = encode_text_with_context_transformer(inputs=a_inputs,
                                                                               context=a_context,
                                                                               tokenizer=self.tokenizer,
                                                                               tokenizer_args=self.tokenizer_args,
                                                                               device=self.device)

        return {'a_inputs': a_input_ids, 'a_mask': a_mask,
                'b_inputs': b_input_ids, 'b_mask': b_mask,
                'a_context': a_context_ids, 'a_context_mask': a_context_mask,
                'b_context': b_context_ids, 'b_context_mask': b_context_mask}


class TextTransformerOutputCollator(CollatorComponent):

    def __call__(
            self,
            inputs,
            context=None
    ):
        (input_ids, attention_mask,
         context_ids, context_mask) = encode_text_with_context_transformer_output(inputs=inputs,
                                                                                  context=context)
        return {'inputs': input_ids, 'input_mask': attention_mask,
                'context': context_ids, 'context_mask': context_mask}


class PairTextTransformerOutputCollator(PairCollatorComponent):

    def __call__(
            self,
            a_inputs,
            b_inputs,
            a_context=None,
            b_context=None
    ):
        a_ids, a_mask, a_context_ids, a_context_mask = encode_text_with_context_transformer_output(inputs=a_inputs,
                                                                                                   context=a_context)

        b_ids, b_mask, b_context_ids, b_context_mask = encode_text_with_context_transformer_output(inputs=b_inputs,
                                                                                                   context=b_context)

        return {'a_inputs': a_ids, 'a_mask': a_mask,
                'b_inputs': b_ids, 'b_mask': b_mask,
                'a_context': a_context_ids, 'a_context_mask': a_context_mask,
                'b_context': b_context_ids, 'b_context_mask': b_context_mask}


class AudioCollator(CollatorComponent):

    def __call__(
            self,
            inputs,
            context=None
    ):
        (audio_features, attention_mask,
         context_features, context_mask) = encode_audio_with_context_torch(inputs=inputs,
                                                                           context=context)

        return {'inputs': audio_features, 'input_mask': attention_mask,
                'context': context_features, 'context_mask': context_mask}


class PairAudioCollator(PairCollatorComponent):

    def _parse_features(
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

        return features, attention_mask.to(th.float32)

    def __call__(
            self,
            a_inputs,
            b_inputs,
            a_context=None,
            b_context=None
    ):
        (a_audio_features, a_attention_mask,
         a_context_features, a_context_mask) = encode_audio_with_context_torch(inputs=a_inputs,
                                                                               context=a_context)

        (b_audio_features, b_attention_mask,
         b_context_features, b_context_mask) = encode_audio_with_context_torch(inputs=b_inputs,
                                                                               context=b_context)

        return {'a_inputs': a_audio_features, 'a_mask': a_attention_mask,
                'b_inputs': b_audio_features, 'b_mask': b_attention_mask,
                'a_context': a_context_features, 'a_context_mask': a_context_mask,
                'b_context': b_context_features, 'b_context_mask': b_context_mask}
