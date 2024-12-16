import torch as th
from torch.nn.utils.rnn import pad_sequence
from torchaudio.backend.soundfile_backend import load
from torchaudio.functional import resample
from transformers import AutoProcessor, AutoModel
from mamkit.components.collators import DataCollator

__all__ = [
    'AudioTransformerCollator',
    'AudioCollator',
    'PairAudioCollator'
]


class AudioTransformerCollator(DataCollator):

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
            batch
    ):
        loaded_audio = []
        for audio_file in batch:
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


class AudioCollator(DataCollator):

    def __call__(
            self,
            batch
    ):
        batch = [th.tensor(feature_set, dtype=th.float32) for feature_set in batch]
        batch = pad_sequence(batch, batch_first=True, padding_value=float('-inf'))
        batch[(batch == float('-inf'))] = 0

        if len(batch.shape) == 3:
            attention_mask = batch[:, :, 0] != float('-inf')
        else:
            attention_mask = th.ones((batch.shape[0]), dtype=th.int32)
            batch = batch[:, None, :]

        return batch, attention_mask.to(th.float32)


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
            batch
    ):
        a_features, b_features = batch

        a_features, a_attention_mask = self._parse_features(a_features)
        b_features, b_attention_mask = self._parse_features(b_features)

        return (a_features, a_attention_mask), (b_features, b_attention_mask)
