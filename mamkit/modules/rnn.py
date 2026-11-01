import torch as th
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMStack(th.nn.Module):
    def __init__(self, input_size, lstm_weigths, return_hidden=True):
        super().__init__()

        self.return_hidden = return_hidden

        self.lstm = th.nn.ModuleList()
        for weight in lstm_weigths:
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

        if self.return_hidden:
            # [bs, d * 2]
            last_hidden = hidden[0]
            return last_hidden.permute(1, 0, 2).reshape(x.shape[0], -1)

        if lengths is not None:
            inputs, _ = pad_packed_sequence(
                inputs, batch_first=True, total_length=x.shape[1]
            )

        # [bs, T, d]
        return inputs
