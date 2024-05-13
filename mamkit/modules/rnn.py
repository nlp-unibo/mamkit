import torch as th


class LSTMStack(th.nn.Module):

    def __init__(
            self,
            input_size,
            lstm_weigths
    ):
        super().__init__()

        self.lstm = th.nn.ModuleList()
        for weight in lstm_weigths:
            self.lstm.append(th.nn.LSTM(input_size=input_size,
                                        hidden_size=weight,
                                        batch_first=True,
                                        bidirectional=True))
            input_size = weight * 2

    def forward(
            self,
            x
    ):
        inputs, attention_mask = x
        hidden = None
        for lstm_module in self.lstm:
            inputs, hidden = lstm_module(inputs)

        # [bs, d * 2]
        last_hidden = hidden[0]
        return last_hidden.permute(1, 0, 2).reshape(inputs.shape[0], -1)
