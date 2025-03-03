from typing import Any, Dict, Tuple, Optional

import torch as th
from transformers import AutoTokenizer

from mamkit.utility.collators import (
    encode_text_with_context_torch,
    encode_text_with_context_transformer,
    encode_text_with_context_transformer_output,
    encode_audio_with_context_torch
)


class CollatorComponent:
    """
    Base collator interface
    """

    def __call__(
            self,
            inputs: Any,
            context: Any = None
    ) -> Dict:
        """
        Args:
            inputs: any input feature
            context: any context features associated with inputs

        Returns:
            Collated outputs in dictionary form.
            All `CollatorComponent` instances must return the following base keys:
                inputs: for collated inputs
                context: for collated context

            Any input-specific additional feature inherits input suffixes as follows:
                input_*any-feature-name*  (e.g., input_attention for attention masks concerning inputs)
                context_*any-feature-name*
        """

        return {'inputs': inputs, 'context': context}


class PairCollatorComponent:
    """
    Base collator interface for pairwise collators
    """

    def __call__(
            self,
            a_inputs: Any,
            b_inputs: Any,
            a_context: Any = None,
            b_context: Any = None
    ):
        """
        Args:
            a_inputs: any input feature corresponding to input A
            b_inputs: any input feature corresponding to input B
            a_context: any context features associated with input A
            b_context: any context features associated with input B

        Returns:
            Collated outputs in dictionary form.
            All `PairCollatorComponent` instances must return the following base keys:
                a_inputs: for collated inputs corresponding to input A
                b_inputs: for collated inputs corresponding to input B
                a_context: for collated context associated with input A
                b_context: for collated context associated with input B

            Any input-specific additional feature inherits input suffixes as follows:
                a_input_*any-feature-name*  (e.g., input_attention for attention masks concerning inputs)
                b_input_*any-feature-name*
                a_context_*any-feature-name*
                b_context_*any-feature-name*
        """

        return {'a_inputs': a_inputs, 'b_inputs': b_inputs, 'a_context': a_context, 'b_context': b_context}


class UnimodalCollator:
    """
    Unimodal collator interface supporting a features collator and a label collator.
    """

    def __init__(
            self,
            features_collator: CollatorComponent,
            label_collator: CollatorComponent
    ):
        """
        Args:
            features_collator: any feature collator
            label_collator: any label collator
        """

        self.features_collator = features_collator
        self.label_collator = label_collator

    def __call__(
            self,
            batch
    ) -> Tuple[Dict, Dict]:
        """
        Executes features collator for inputs and label collator for output labels.

        Args:
            batch: any batched data in the form of nested lists of content, typically derived from torch.Dataset

        Returns:
            features_collated: Dict containing collated input features.
            labels_collated: Dict containing collated output features
        """

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
    """
    Pair Unimodal collator interface supporting a features collator and a label collator.
    """

    def __init__(
            self,
            features_collator: PairCollatorComponent,
            label_collator: CollatorComponent
    ):
        """
        Args:
            features_collator: any feature collator
            label_collator: any label collator
        """

        self.features_collator = features_collator
        self.label_collator = label_collator

    def __call__(
            self,
            batch
    ) -> Tuple[Dict, Dict]:
        """
        Executes features collator for inputs and label collator for output labels.

        Args:
            batch: any batched data in the form of nested lists of content, typically derived from torch.Dataset

        Returns:
            features_collated: Dict containing collated input features.
            labels_collated: Dict containing collated output features
        """

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
    """
    Multimodal collator interface supporting a text collator, an audio collator, and a label collator.
    """

    def __init__(
            self,
            text_collator: CollatorComponent,
            audio_collator: CollatorComponent,
            label_collator: CollatorComponent
    ):
        """
        Args:
            text_collator: any text feature collator
            audio_collator: any audio feature collator
            label_collator: any label collator
        """

        self.text_collator = text_collator
        self.audio_collator = audio_collator
        self.label_collator = label_collator

    def __call__(
            self,
            batch
    ) -> Tuple[Dict, Dict]:
        """
        Executes features collator for inputs and label collator for output labels.

        Args:
            batch: any batched data in the form of nested lists of content, typically derived from torch.Dataset

        Returns:
            features_collated: Dict containing collated text and audio features.
            Text features start with 'text_' prefix while audio features start with 'audio_' prefix.
            labels_collated: Dict containing collated output features
        """

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
    """
    Pair Multimodal collator interface supporting a text collator, an audio collator, and a label collator.
    """

    def __init__(
            self,
            text_collator: PairCollatorComponent,
            audio_collator: PairCollatorComponent,
            label_collator: CollatorComponent
    ):
        """
        Args:
            text_collator: any text feature collator
            audio_collator: any audio feature collator
            label_collator: any label collator
        """

        self.text_collator = text_collator
        self.audio_collator = audio_collator
        self.label_collator = label_collator

    def __call__(
            self,
            batch
    ) -> Tuple[Dict, Dict]:
        """
        Executes features collator for inputs and label collator for output labels.

        Args:
            batch: any batched data in the form of nested lists of content, typically derived from torch.Dataset

        Returns:
            features_collated: Dict containing collated text and audio features.
            Text features start with 'text_' prefix while audio features start with 'audio_' prefix.
            labels_collated: Dict containing collated output features
        """

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
    """
    Torch text collator that requires a tokenizer and its corresponding pre-defined vocabulary.
    """

    def __init__(
            self,
            tokenizer,
            vocab
    ):
        """
        Args:
            tokenizer: any torchtext tokenizer
            vocab: torchtext vocab
        """

        self.tokenizer = tokenizer
        self.vocab = vocab

    def __call__(
            self,
            inputs,
            context=None
    ) -> Dict:
        """
        Tokenizes and pads input texts via internal tokenizer and vocabulary.

        Args:
            inputs: input texts to tokenize and encode
            context: context texts to tokenize and encode

        Returns:
            Collated outputs in dictionary form.
            inputs: collated input ids
            context: collated context ids
        """

        texts, context = encode_text_with_context_torch(inputs=inputs,
                                                        context=context,
                                                        vocab=self.vocab,
                                                        tokenizer=self.tokenizer)
        return {'inputs': texts, 'context': context}


class PairTextCollator(PairCollatorComponent):
    """
    Pair torch text collator that requires a tokenizer and its corresponding pre-defined vocabulary.
    """

    def __init__(
            self,
            tokenizer,
            vocab
    ):
        """
        Args:
            tokenizer: any torchtext tokenizer
            vocab: torchtext vocab
        """

        self.tokenizer = tokenizer
        self.vocab = vocab

    def __call__(
            self,
            a_inputs,
            b_inputs,
            a_context=None,
            b_context=None
    ):
        """
        Args:
            a_inputs: input texts to tokenize and encode corresponding to input A
            b_inputs: input texts to tokenize and encode corresponding to input B
            a_context: context texts to tokenize and encode associated with input A
            b_context: context texts to tokenize and encode associated with input B

        Returns:
            Collated outputs in dictionary form.
            a_inputs: collated input ids corresponding to input A
            b_inputs: collated input ids corresponding to input B
            a_context: collated context ids associated with input A
            b_context: collated context ids associated with input B
        """

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
    """
    Text transformer collator
    """

    def __init__(
            self,
            model_card: str,
            tokenizer_args: Optional[Dict] = None,
    ):
        """
        Args:
            model_card: any huggingface model card
            tokenizer_args: any tokenizer-specific additional arguments
        """

        self.model_card = model_card
        self.tokenizer_args = tokenizer_args if tokenizer_args is not None else {}

        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_card)

    def __call__(
            self,
            inputs,
            context=None
    ):
        """
        Tokenizes and pads input texts via internal transformer tokenizer.

        Args:
            inputs: input texts to tokenize and encode
            context: context text to tokenize and encode

        Returns:
            Collated outputs in dictionary form.
            inputs: collated input ids
            input_mask: collated attention mask
            context: collated context ids
            context_mask: collated attention mask
        """

        (input_ids, attention_mask,
         context_ids, context_mask) = encode_text_with_context_transformer(inputs=inputs,
                                                                           context=context,
                                                                           tokenizer=self.tokenizer,
                                                                           tokenizer_args=self.tokenizer_args,
                                                                           device=self.device)

        return {'inputs': input_ids, 'input_mask': attention_mask,
                'context': context_ids, 'context_mask': context_mask}


class PairTextTransformerCollator(PairCollatorComponent):
    """
    Pair text transformer collator
    """

    def __init__(
            self,
            model_card,
            tokenizer_args=None,
    ):
        """
        Args:
            model_card: any huggingface model card
            tokenizer_args: any tokenizer-specific additional arguments
        """

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
        """
        Tokenizes and pads input texts via internal transformer tokenizer.

        Args:
            a_inputs: input texts to tokenize and encode corresponding to input A
            b_inputs: input texts to tokenize and encode corresponding to input B
            a_context: input text context to tokenize and encode associated with input A
            b_context: input text context to tokenize and encode associated with input B

        Returns:
            Collated outputs in dictionary form.
            a_inputs: collated input ids corresponding to input A
            a_input_mask: collated attention mask corresponding to input A
            a_context: collated context ids corresponding to input A
            a_context_mask: collated attention mask corresponding to input A
            b_inputs: collated input ids corresponding to input B
            b_input_mask: collated attention mask corresponding to input B
            b_context: collated context ids corresponding to input B
            b_context_mask: collated attention mask corresponding to input B
        """

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

        return {'a_inputs': a_input_ids, 'a_input_mask': a_mask,
                'b_inputs': b_input_ids, 'b_input_mask': b_mask,
                'a_context': a_context_ids, 'a_context_mask': a_context_mask,
                'b_context': b_context_ids, 'b_context_mask': b_context_mask}


class TextTransformerOutputCollator(CollatorComponent):
    """
    Text transformer output collator to simply pad transformer outputs.
    """

    # TODO: this should receive padding token id
    def __call__(
            self,
            inputs,
            context=None
    ):
        """
        Pads input texts via internal transformer tokenizer.

        Args:
            inputs: input texts to tokenize and encode
            context: input text context to tokenize and encode

        Returns:
            Collated outputs in dictionary form.
            inputs: collated input ids
            input_mask: collated attention mask
            context: collated context ids
            context_mask: collated attention mask
        """

        (input_ids, attention_mask,
         context_ids, context_mask) = encode_text_with_context_transformer_output(inputs=inputs,
                                                                                  context=context)
        return {'inputs': input_ids, 'input_mask': attention_mask,
                'context': context_ids, 'context_mask': context_mask}


class PairTextTransformerOutputCollator(PairCollatorComponent):
    """
    Pair text transformer output collator to simply pad transformer outputs.
    """

    def __call__(
            self,
            a_inputs,
            b_inputs,
            a_context=None,
            b_context=None
    ):
        """
        Pads input texts via internal transformer tokenizer.

        Args:
            a_inputs: input texts to tokenize and encode corresponding to input A
            b_inputs: input texts to tokenize and encode corresponding to input B
            a_context: input text context to tokenize and encode associated with input A
            b_context: input text context to tokenize and encode associated with input B

        Returns:
            Collated outputs in dictionary form.
            a_inputs: collated input ids corresponding to input A
            a_input_mask: collated attention mask corresponding to input A
            a_context: collated context ids corresponding to input A
            a_context_mask: collated attention mask corresponding to input A
            b_inputs: collated input ids corresponding to input B
            b_input_mask: collated attention mask corresponding to input B
            b_context: collated context ids corresponding to input B
            b_context_mask: collated attention mask corresponding to input B
        """

        a_ids, a_mask, a_context_ids, a_context_mask = encode_text_with_context_transformer_output(inputs=a_inputs,
                                                                                                   context=a_context)

        b_ids, b_mask, b_context_ids, b_context_mask = encode_text_with_context_transformer_output(inputs=b_inputs,
                                                                                                   context=b_context)

        return {'a_inputs': a_ids, 'a_input_mask': a_mask,
                'b_inputs': b_ids, 'b_input_mask': b_mask,
                'a_context': a_context_ids, 'a_context_mask': a_context_mask,
                'b_context': b_context_ids, 'b_context_mask': b_context_mask}


class AudioCollatorOutput(CollatorComponent):
    """
    Torch audio output collator to pad audio features
    """

    def __call__(
            self,
            inputs,
            context=None
    ):
        """
        Pads audio features

        Args:
            inputs: input audio features to pad
            context: context audio features to pad

        Returns:
            Collated outputs in dictionary form.
            inputs: collated input ids
            input_mask: collated attention mask
            context: collated context ids
            context_mask: collated attention mask
        """

        (audio_features, attention_mask,
         context_features, context_mask) = encode_audio_with_context_torch(inputs=inputs,
                                                                           context=context)

        return {'inputs': audio_features, 'input_mask': attention_mask,
                'context': context_features, 'context_mask': context_mask}


class PairAudioOutputCollator(PairCollatorComponent):
    """
    Pair torch audio output collator to pad audio features
    """

    def __call__(
            self,
            a_inputs,
            b_inputs,
            a_context=None,
            b_context=None
    ):
        """
        Pads audio features

        Args:
            a_inputs: input audio features corresponding to input A
            b_inputs: input audio features corresponding to input B
            a_context: input audio context associated with input A
            b_context: input audio context associated with input B

        Returns:
            Collated outputs in dictionary form.
            a_inputs: collated audio features corresponding to input A
            a_input_mask: collated audio attention mask corresponding to input A
            a_context: collated context audio features corresponding to input A
            a_context_mask: collated context audio attention mask corresponding to input A
            b_inputs: collated input audio features corresponding to input B
            b_input_mask: collated audio attention mask corresponding to input B
            b_context: collated context audio features corresponding to input B
            b_context_mask: collated audio attention mask corresponding to input B
        """

        (a_audio_features, a_attention_mask,
         a_context_features, a_context_mask) = encode_audio_with_context_torch(inputs=a_inputs,
                                                                               context=a_context)

        (b_audio_features, b_attention_mask,
         b_context_features, b_context_mask) = encode_audio_with_context_torch(inputs=b_inputs,
                                                                               context=b_context)

        return {'a_inputs': a_audio_features, 'a_input_mask': a_attention_mask,
                'b_inputs': b_audio_features, 'b_input_mask': b_attention_mask,
                'a_context': a_context_features, 'a_context_mask': a_context_mask,
                'b_context': b_context_features, 'b_context_mask': b_context_mask}
