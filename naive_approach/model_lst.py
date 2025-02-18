import torch
import torch.nn as nn

class LST(nn.Module):
    """
    :param SOURCE_SIZE: input size
    :param TARGET_SIZE: target size
    :param hidden: number of hidden neurons
    :param layer: number of layers
    """
    def __init__(self, SOURCE_SIZE: int, TARGET_SIZE: int, hidden=200, layers=1, dtype=torch.float32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=SOURCE_SIZE, hidden_size=hidden, num_layers=layers, batch_first=True, dtype=dtype)
        self.linear = nn.Linear(hidden, TARGET_SIZE, dtype=dtype)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

