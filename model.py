import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(32, 3)
        self.hidden_size=16
        self.num_layers=2

    def forward(self, x):
        x = x.permute(0,2,1)
        # x = nn.utils.rnn.pack_padded_sequence(x, [9]*10, batch_first=True, enforce_sorted=True)
        # print(x.size())
        out, hn = self.rnn(x, None)
        # out = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        # print(out.size())
        out = self.linear(out[:,-1,:])
        # print(out.size())
        return out

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.linear.reset_parameters()

