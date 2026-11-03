import torch as th
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMStack(th.nn.Module):
    """A stack of bidirectional LSTMs, one per width in ``lstm_weights``.

    It returns the last hidden states of both directions of the top layer,
    concatenated, with shape ``[bs, 2 * lstm_weights[-1]]``.
    """

    def __init__(self, input_size: int, lstm_weights):
        super().__init__()
        self.lstm = th.nn.ModuleList()
        for weight in lstm_weights:
            self.lstm.append(
                th.nn.LSTM(
                    input_size=input_size,
                    hidden_size=weight,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            input_size = weight * 2

    def forward(self, x, lengths=None):
        """Runs the stack, packing ``x`` when ``lengths`` holds sequence lengths."""
        hidden = None
        inputs = x
        if lengths is not None:
            inputs = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        for lstm_module in self.lstm:
            inputs, hidden = lstm_module(inputs)
        # [bs, d * 2]
        return hidden[0].permute(1, 0, 2).reshape(x.shape[0], -1)
