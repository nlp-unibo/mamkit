from torchaudio import load
from torchaudio.functional import resample
import torch as th
from torch.nn.utils.rnn import pad_sequence


def encode_text_torch(
        inputs,
        vocab,
        tokenizer
):
    texts = [th.tensor(vocab(tokenizer(text))) for text in inputs]
    texts = pad_sequence(texts, padding_value=0, batch_first=True)
    return texts


def encode_text_with_context_torch(
        inputs,
        vocab,
        tokenizer,
        context=None,
):
    texts = encode_text_torch(inputs=inputs, vocab=vocab, tokenizer=tokenizer)

    if context is not None:
        context = encode_text_torch(inputs=inputs, vocab=vocab, tokenizer=tokenizer)

    return texts, context


def encode_text_transformer(
        inputs,
        tokenizer,
        device,
        tokenizer_args={}
):
    tok_inputs = tokenizer(inputs,
                           padding=True,
                           return_tensors='pt',
                           **tokenizer_args).to(device)
    return tok_inputs


def encode_text_with_context_transformer(
        inputs,
        tokenizer,
        device,
        tokenizer_args={},
        context=None
):
    tok_inputs = encode_text_transformer(inputs=inputs,
                                         tokenizer=tokenizer,
                                         device=device,
                                         tokenizer_args=tokenizer_args)

    tok_context = {
        'input_ids': None,
        'attention_mask': None
    }
    if context is not None:
        tok_context = encode_text_transformer(inputs=context,
                                              tokenizer=tokenizer,
                                              device=device,
                                              tokenizer_args=tokenizer_args)

    return (tok_inputs['input_ids'], tok_inputs['attention_mask'],
            tok_context['input_ids'], tok_context['attention_mask'])


def encode_text_transformer_output(
        inputs
):
    texts = pad_sequence([th.tensor(text, dtype=th.float32) for text in inputs],
                         padding_value=0.0, batch_first=True)
    attention_mask = texts[:, :, 0] != 0.0
    attention_mask = attention_mask.to(th.float32)

    return texts, attention_mask


def encode_text_with_context_transformer_output(
        inputs,
        context=None
):
    texts, attention_mask = encode_text_transformer_output(inputs=inputs)

    context_texts, context_mask = None, None
    if context is not None:
        context_texts, context_mask = encode_text_transformer_output(inputs=context)

    return texts, attention_mask, context_texts, context_mask


def encode_audio_transformer(
        inputs,
        processor,
        model,
        device,
        model_args={},
        processor_args={},
        sampling_rate=16000,
        downsampling_factor=None,
        aggregate=False
):
    loaded_audio = []

    for audio_file in inputs:
        if not audio_file.is_file():
            raise RuntimeError(f'Could not read file {audio_file}')
        audio, sampling_rate = load(audio_file.as_posix())
        if sampling_rate != sampling_rate:
            audio = resample(audio, sampling_rate, sampling_rate)
        audio = th.mean(audio, dim=0)
        loaded_audio.append(audio)

    loaded_audio = pad_sequence(loaded_audio, batch_first=True, padding_value=0.0)
    with th.inference_mode():
        features = processor(loaded_audio,
                             sampling_rate=sampling_rate,
                             return_tensors='pt',
                             return_attention_mask=True,
                             **processor_args)
        attention_mask = features.attention_mask
        features = features.input_values[0].to(device)
        features = model(features, **model_args).last_hidden_state

        if downsampling_factor is not None:
            features = th.nn.functional.interpolate(features.permute(0, 2, 1),
                                                    scale_factor=downsampling_factor,
                                                    mode='linear')
            features = features.permute(0, 2, 1)

    if aggregate:
        features = th.mean(features, dim=1, keepdim=True)

    return features, attention_mask


def encode_audio_torch(
        inputs
):
    features = [th.tensor(feature_set, dtype=th.float32) for feature_set in inputs]
    features = pad_sequence(features, batch_first=True, padding_value=float('-inf'))
    features[(features == float('-inf'))] = 0

    if len(features.shape) == 3:
        attention_mask = features[:, :, 0] != float('-inf')
    else:
        attention_mask = th.ones((features.shape[0]), dtype=th.int32)
        features = features[:, None, :]

    return features, attention_mask.to(th.float32)


def encode_audio_with_context_torch(
        inputs,
        context=None
):
    input_features, attention_mask = encode_audio_torch(inputs=inputs)

    context_features, context_mask = None, None
    if context is not None:
        context_features, context_mask = encode_audio_torch(inputs=context)

    return input_features, attention_mask, context_features, context_mask
