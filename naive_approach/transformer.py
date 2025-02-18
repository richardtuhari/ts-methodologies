import torch
import torch.nn as nn

eps=1e-5

class TransformDecoupled(nn.Module):
    def __init__(self, tail: int, eps=eps, affine=True, mapper=lambda x: x, mapper_inv=lambda x: x):
        """
        :param tail: input size, when normalizing truncate to this size
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        :param mapper: a function to introduce custom nonlinearity
        :param mapper_inv: the inverse of mapper
        """
        super(TransformDecoupled, self).__init__()
        self.tail = tail
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.f = mapper
            self.fi = mapper_inv
            self._init_params()

    @property
    def _vars(self):
        if self.affine:
            return (self.mean.item(), self.stdev.item(), self.affine_weight.item(), self.affine_bias.item(), self.affine_weight_old, self.affine_bias_old)
        else:
            return (self.mean.item(), self.stdev.item())

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x[:, -self.tail:])
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(1))
        self.affine_bias = nn.Parameter(torch.zeros(1))

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            self.affine_weight_old = self.affine_weight.item()
            self.affine_bias_old = self.affine_bias.item()
            x = self.f(x) * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = self.fi(x) - self.affine_bias_old
            x = x / (self.affine_weight_old + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class Transform(nn.Module):
    def __init__(self, tail: int, eps=eps, affine=True, mapper=lambda x: x, mapper_inv=lambda x: x):
        """
        :param tail: input size, when normalizing truncate to this size
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        :param mapper: a function to introduce custom nonlinearity
        :param mapper_inv: the inverse of mapper
        """
        super(Transform, self).__init__()
        self.tail = tail
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.f = mapper
            self.fi = mapper_inv
            self._init_params()

    @property
    def _vars(self):
        if self.affine:
            return (self.mean.item(), self.stdev.item(), self.affine_weight.item(), self.affine_bias.item())
        else:
            return (self.mean.item(), self.stdev.item())

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x[:, -self.tail:])
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(1))
        self.affine_bias = nn.Parameter(torch.zeros(1))

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = self.f(x) * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = self.fi(x) - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class TransformNoZDecoupled(nn.Module):
    def __init__(self, tail: int, eps=eps, mapper=lambda x: x, mapper_inv=lambda x: x):
        """
        :param tail: input size, when normalizing truncate to this size
        :param eps: a value added for numerical stability
        :param mapper: a function to introduce custom nonlinearity
        :param mapper_inv: the inverse of mapper
        """
        super(TransformNoZDecoupled, self).__init__()
        self.tail = tail
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1))
        self.affine_bias = nn.Parameter(torch.zeros(1))
        self.f = mapper
        self.fi = mapper_inv

    @property
    def _vars(self):
        return (self.affine_weight.item(), self.affine_bias.item(), self.affine_weight_old, self.affine_bias_old)

    def forward(self, x, mode:str):
        if mode == 'norm':
            x = self._normalize(x[:, -self.tail:])
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _normalize(self, x):
        self.affine_weight_old = self.affine_weight.item()
        self.affine_bias_old = self.affine_bias.item()
        x = x * self.affine_weight
        x = x + self.affine_bias
        return self.f(x)

    def _denormalize(self, x):
        x = self.fi(x) - self.affine_bias_old
        x = x / (self.affine_weight_old + self.eps*self.eps)
        return x

class TransformNoZ(nn.Module):
    def __init__(self, tail: int, eps=eps, mapper=lambda x: x, mapper_inv=lambda x: x):
        """
        :param tail: input size, when normalizing truncate to this size
        :param eps: a value added for numerical stability
        :param mapper: a function to introduce custom nonlinearity
        :param mapper_inv: the inverse of mapper
        """
        super(TransformNoZ, self).__init__()
        self.tail = tail
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1))
        self.affine_bias = nn.Parameter(torch.zeros(1))
        self.f = mapper
        self.fi = mapper_inv

    @property
    def _vars(self):
        return (self.affine_weight.item(), self.affine_bias.item())

    def forward(self, x, mode:str):
        if mode == 'norm':
            x = self._normalize(x[:, -self.tail:])
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _normalize(self, x):
        x = x * self.affine_weight
        x = x + self.affine_bias
        return self.f(x)

    def _denormalize(self, x):
        x = self.fi(x) - self.affine_bias
        x = x / (self.affine_weight + self.eps*self.eps)
        return x

class TransformFutureDecoupled(nn.Module):
    def __init__(self, tail: int, head: int, eps=eps, affine=True, mapper=lambda x: x, mapper_inv=lambda x: x):
        """
        :param tail: input size, when normalizing truncate to this size
        :param head: output size, when normalizing dont care about the bias, suppose we already know the future
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(TransformFutureDecoupled, self).__init__()
        self.tail = tail
        self.head = head
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    @property
    def _vars(self):
        if self.affine:
            return (self.mean.item(), self.stdev.item(), self.affine_weight.item(), self.affine_bias.item(), self.affine_weight_old, self.affine_bias_old)
        else:
            return (self.mean.item(), self.stdev.item())

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x[:, -self.tail-self.head:-self.head])
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(1))
        self.affine_bias = nn.Parameter(torch.zeros(1))

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            self.affine_weight_old = self.affine_weight.item()
            self.affine_bias_old = self.affine_bias.item()
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias_old
            x = x / (self.affine_weight_old + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class TransformFuture(nn.Module):
    def __init__(self, tail: int, head: int, eps=eps, affine=True, mapper=lambda x: x, mapper_inv=lambda x: x):
        """
        :param tail: input size, when normalizing truncate to this size
        :param head: output size, when normalizing dont care about the bias, suppose we already know the future
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(TransformFuture, self).__init__()
        self.tail = tail
        self.head = head
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    @property
    def _vars(self):
        if self.affine:
            return (self.mean.item(), self.stdev.item(), self.affine_weight.item(), self.affine_bias.item())
        else:
            return (self.mean.item(), self.stdev.item())

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x[:, -self.tail-self.head:-self.head])
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(1))
        self.affine_bias = nn.Parameter(torch.zeros(1))

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

