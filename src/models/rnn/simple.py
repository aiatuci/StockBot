import torch
import torch.nn as nn

__all__ = ['SimpleRNN']


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_cells=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_cells = num_cells
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_cells, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size, bias=True)

    def forward(self, sequence, state=None):
        if state and len(state) == 2:
            h_0, c_0 = state
        else:
            h_0 = torch.zeros(self.num_cells, sequence.shape[0], self.hidden_size)
            c_0 = torch.zeros(self.num_cells, sequence.shape[0], self.hidden_size)
        output, (h_out, c_out) = self.lstm(sequence, (h_0, c_0))
        output = self.fc(output)
        return output


if __name__ == '__main__':
    import torch

    OUT_SHAPE = (1, 52, 7)
    inputs = torch.randn(1, 52, 7)
    rnn = SimpleRNN(7, 64, 1)
    output = rnn(inputs)
    assert output.shape == OUT_SHAPE, f"Received output of shape {output.shape}. Expected {OUT_SHAPE}."
