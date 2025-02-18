import torch
import torch.nn as nn

class LIN(nn.Module):
    def __init__(self, SOURCE_SIZE: int, TARGET_SIZE: int, hidden = 20, activation = nn.Sigmoid(), dtype=torch.float32):
        """
        :param SOURCE_SIZE: input size
        :param TARGET_SIZE: target size
        :param hidden: number of hidden neurons
        :param sctivation: non linearity
        """
        super(LIN, self).__init__()
        self.linear1 = nn.Linear(SOURCE_SIZE, hidden, dtype=dtype)
        self.act = activation
        self.linear2 = nn.Linear(hidden, TARGET_SIZE, dtype=dtype)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x
