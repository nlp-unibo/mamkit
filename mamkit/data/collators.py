import torch as th
from torch.nn.utils.rnn import pad_sequence
from torchaudio.backend.soundfile_backend import load
from torchaudio.functional import resample
from transformers import AutoTokenizer, AutoProcessor, AutoModel


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


class PairTextCollator(TextCollator):

    def __call__(
            self,
            texts
    ):
        a_texts, b_texts = texts
        a_texts = [th.tensor(self.vocab(self.tokenizer(text))) for text in a_texts]
        a_texts = pad_sequence(a_texts, padding_value=0, batch_first=True)

        b_texts = [th.tensor(self.vocab(self.tokenizer(text))) for text in b_texts]
        b_texts = pad_sequence(b_texts, padding_value=0, batch_first=True)

        return a_texts, b_texts


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


class PairTextTransformerCollator(TextTransformerCollator):

    def __call__(
            self,
            texts
    ):
        a_texts, b_texts = texts
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


class TextTransformerOutputCollator:

    def __call__(
            self,
            texts
    ):
        texts = pad_sequence([th.tensor(text, dtype=th.float32) for text in texts], padding_value=0.0, batch_first=True)
        attention_mask = texts[:, :, 0] != 0.0
        return texts, attention_mask.to(th.float32)


class PairTextTransformerOutputCollator(TextTransformerOutputCollator):

    def __call__(
            self,
            texts
    ):
        a_texts, b_texts = texts

        a_texts = pad_sequence([th.tensor(text, dtype=th.float32) for text in a_texts], padding_value=0.0,
                               batch_first=True)
        a_attention_mask = a_texts[:, :, 0] != 0.0

        b_texts = pad_sequence([th.tensor(text, dtype=th.float32) for text in b_texts], padding_value=0.0,
                               batch_first=True)
        b_attention_mask = b_texts[:, :, 0] != 0.0

        return (a_texts, a_attention_mask.to(th.float32)), (b_texts, b_attention_mask.to(th.float32))


class AudioTransformerCollator:

    def __init__(
            self,
            model_card,
            sampling_rate,
            downsampling_factor=None,
            aggregate=False,
            processor_args=None,
            model_args=None,
    ):
        self.model_card = model_card
        self.sampling_rate = sampling_rate
        self.processor_args = processor_args if processor_args is not None else {}
        self.model_args = model_args if model_args is not None else {}
        self.downsampling_factor = downsampling_factor
        self.aggregate = aggregate

        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(model_card)
        self.model = AutoModel.from_pretrained(model_card).to(self.device)

    def __call__(
            self,
            audio_files
    ):
        loaded_audio = []
        for audio_file in audio_files:
            if not audio_file.is_file():
                raise RuntimeError(f'Could not read file {audio_file}')
            audio, sampling_rate = load(audio_file.as_posix())
            if sampling_rate != self.sampling_rate:
                audio = resample(audio, sampling_rate, self.sampling_rate)
            audio = th.mean(audio, dim=0)
            loaded_audio.append(audio)

        loaded_audio = pad_sequence(loaded_audio, batch_first=True, padding_value=0.0)
        with th.inference_mode():
            features = self.processor(loaded_audio,
                                      sampling_rate=self.sampling_rate,
                                      return_tensors='pt',
                                      return_attention_mask=True,
                                      **self.processor_args)
            attention_mask = features.attention_mask
            features = features.input_values[0].to(self.device)
            features = self.model(features, **self.model_args).last_hidden_state

            if self.downsampling_factor is not None:
                features = th.nn.functional.interpolate(features.permute(0, 2, 1),
                                                        scale_factor=self.downsampling_factor,
                                                        mode='linear')
                features = features.permute(0, 2, 1)

        if self.aggregate:
            features = th.mean(features, dim=1, keepdim=True)

        return features, attention_mask


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

        return features, attention_mask.to(th.float32)


class PairAudioCollator(AudioCollator):

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
            features
    ):
        a_features, b_features = features

        a_features, a_attention_mask = self._parse_features(a_features)
        b_features, b_attention_mask = self._parse_features(b_features)

        return (a_features, a_attention_mask), (b_features, b_attention_mask)
