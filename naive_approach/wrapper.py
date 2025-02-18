import torch.nn as nn


class WrappedModel(nn.Module):
    def __init__(self, model: nn.Module, trafo: nn.Module):
        super(WrappedModel, self).__init__()
        self.m = model
        self.t = trafo

    def forward(self, x):
        x = self.t.forward(x, 'norm')
        x = self.m.forward(x)
        x = self.t.forward(x, 'denorm')
        return x

