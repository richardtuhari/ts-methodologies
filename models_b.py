import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Callable, Optional

'''This file contains the models used in out study. The code for NBeats and DLinear was copied from their original GitHub repositories, 
and only necessary modifications were made, to ensure compatibility with our framework.'''


#LST
#----------------------------------

class LST(nn.Module):
    def __init__(self, SOURCE_SIZE, TARGET_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size=SOURCE_SIZE, hidden_size=200, num_layers=1, batch_first=True)
        self.linear = nn.Linear(200, TARGET_SIZE)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

#LIN
#-----------------------------------

class LIN(nn.Module):
    def __init__(self, SOURCE_SIZE, TARGET_SIZE):
        super(LIN, self).__init__()
        self.linear1 = nn.Linear(SOURCE_SIZE, 20)
        #self.act = nn.ReLU()
        self.act = nn.Sigmoid()
        self.linear2 = nn.Linear(20, TARGET_SIZE)
        self.m = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


#NBEATS
#----------------------------

class GenericBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta):
        return theta[:,:self.backcast_size], theta[:,-self.forecast_size:]

class NBeatsBlock(nn.Module):
    def __init__(self, SOURCE_SIZE, TARGET_SIZE, basis_function, layers, layer_size):
        super().__init__()
        theta_size = SOURCE_SIZE + TARGET_SIZE
        self.basis_function = basis_function
        self.basis_parameters = nn.Linear(in_features = layer_size, out_features = theta_size)
        self.layers = nn.ModuleList([nn.Linear(in_features = SOURCE_SIZE, out_features = layer_size)] + 
                                   [nn.Linear(in_features = layer_size, out_features = layer_size) for _ in range(layers - 1)])

    def forward(self, x):
        inp = x
        for layer in self.layers:
            inp = layer(inp)
            inp = torch.relu(inp)
        basis_parameters = self.basis_parameters(inp)
        return self.basis_function(basis_parameters)

class NBeats(nn.Module):
    def __init__(self, SOURCE_SIZE, TARGET_SIZE, layers, layer_size, stacks):
        super().__init__()
        basis_function = GenericBasis(backcast_size=SOURCE_SIZE, forecast_size=TARGET_SIZE)
        self.blocks = nn.ModuleList([NBeatsBlock(SOURCE_SIZE=SOURCE_SIZE,
                                               TARGET_SIZE=TARGET_SIZE,
                                               basis_function=GenericBasis(backcast_size=SOURCE_SIZE,
                                                                           forecast_size=TARGET_SIZE),
                                               layers=layers,
                                               layer_size=layer_size)
                                   for _ in range(stacks)])

    def forward(self, x):
        residuals = x.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        return forecast


#DLinear
#---------------------------

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, SOURCE_SIZE, TARGET_SIZE, INDIVIDUAL, N_VARIATE):
        super(DLinear, self).__init__()
        self.seq_len = SOURCE_SIZE
        self.pred_len = TARGET_SIZE

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = INDIVIDUAL
        self.channels = N_VARIATE

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len)
                                         
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = x[:,:,None]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1)[:,:,0] # to [Batch, Output length, Channel]



#SegRNN
#-------------------------------

class SegRNN(nn.Module):  
    def __init__(self, SOURCE_SIZE, TARGET_SIZE, SEGMENT_SIZE, N_VARIATE, HIDDEN_SIZE, DROPOUT):
        super(SegRNN, self).__init__()
        
        # get parameters
        self.seq_len = SOURCE_SIZE
        self.pred_len = TARGET_SIZE
        self.enc_in = N_VARIATE
        self.d_model = HIDDEN_SIZE
        self.dropout = DROPOUT
        
        self.seg_len = SEGMENT_SIZE
        self.seg_num_x = self.seq_len//self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len
        
        
        self.valueEmbedding = nn.Sequential(
         nn.Linear(self.seg_len, self.d_model),
         nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                           batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
        self.predict = nn.Sequential(
         nn.Dropout(self.dropout),
         nn.Linear(self.d_model, self.seg_len)
        )

    def forward(self, x):
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        x = x[:,:,None]
        batch_size = x.size(0)
        
        # normalization and permute     b,s,c -> b,c,s
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1) # b,c,s
        
        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))
        
        # encoding
        _, hn = self.rnn(x) # bc,n,d  1,bc,d
        
        # m,d//2 -> 1,m,d//2 -> c,m,d//2
        # c,d//2 -> c,1,d//2 -> c,m,d//2
        # c,m,d -> cm,1,d -> bcm, 1, d
        pos_emb = torch.cat([
         self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
         self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size,1,1)
        
        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)) # bcm,1,d  1,bcm,d
        
        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)
        
        # permute and denorm
        y = y.permute(0, 2, 1) + seq_last
        
        return y[:,:,0]


#Predictor
#----------------------------------

class predictor(torch.nn.Module):
    """
A model that mimics a series expansion.
With more than one terms F~ projections try learn differences
    """
    def __init__(self, SOURCE_SIZE, TERMS=2, l=14, activation = torch.nn.ReLU, p_in = 0., p_hidden = 0.):
        """
Input parameters
----
input_size: the window size of the past
l: the width of the hidden layers
terms: how fine grained the model is (default 1)
activation: the type of non-linearity (default ReLU)
p_in: input layer dropout (default 0)
p_hidden: hidden layer dropout (default 0)
        """
        terms = TERMS
        input_size = SOURCE_SIZE
        assert input_size % terms == 0, f'input_size must be divisible by terms: {input_size}%{terms}'
        h = input_size // terms
        super().__init__()
        self.w = h
        self.order = terms
        
        self.a = activation()
        
        # common dense mlp
        self.sharedMLP = torch.nn.Sequential(
            torch.nn.Linear(h, l),
            activation(),
            torch.nn.Dropout(p = p_in),
            
            torch.nn.Linear(l, l),
            activation(),
            torch.nn.Dropout(p = p_hidden),

            torch.nn.Linear(l, l),
            activation(),
            torch.nn.Dropout(p = p_hidden),
            
            torch.nn.Linear(l, l)
        )

        # first order: F
        self.layer_p = torch.nn.Linear(l, 1)
        
        # next chunk projectors: F~
        self.layer_pw = torch.nn.Linear(l, h) #TODO: what if all differ?
    
        # differential layers and projectors (simgle layer MLP)
        self.layer_pd = torch.nn.ModuleList([ torch.nn.Linear(h, h) for _ in range(1, terms) ])
        self.layer_pp = torch.nn.ModuleList([ torch.nn.Linear(h, 1) for _ in range(1, terms) ])
        
    def forward(self, x):                   # order*w -> 1
        p = 0
        for i in range(self.order, 1, -1):
            I = -i*self.w
            J = (1-i)*self.w
            K = (2-i)*self.w
            # split data
            x_fp = x[:, I:J]                        # w
            x_np = x[:, J:K] if K<0 else x[:, J:]   # w
            # predict near past from far past
            x_fpp = self.sharedMLP(x_fp)            # l
            x_fpp = self.layer_pw(x_fpp)            # w
            x_fpp = self.a(x_fpp)                   # w
            # difference of predicted and actual near past
            diff = x_np - x_fpp                     # w
            diff = self.layer_pd[2-i](diff)         # l
            diff = self.a(diff)                     # l
            # add correction
            p += self.layer_pp[2-i](diff)           # 1
        # first order term
        i_m1 = x[:,-self.w:]
        near_past = self.sharedMLP(i_m1)
        p += self.layer_p(near_past)
        return p


#NHITS
#---------------------------


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size, interpolation_mode):
        super().__init__()
        assert (interpolation_mode in ['linear', 'nearest']) or ('cubic' in interpolation_mode)
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(self, theta):
        backcast = theta[:,:self.backcast_size]
        knots = theta[:,self.backcast_size:]

        if self.interpolation_mode in ['nearest', 'linear']:
            knots = knots[:,None,:]
            forecast = F.interpolate(knots, size = self.forecast_size, mode = self.interpolation_mode)
            forecast = forecast[:,0,:]

        elif 'cubic' in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split('-')[-1])
            knots = knots[:,None,None,:]
            forecast = t.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots)/batch_size))

            for i in range(n_batches):
                forecast_i = F.interpolate(knots[i*batch_size:(i+1)*batch_size], size=self.forecast_size, mode='bicubic')
                forecast[i*batch_size:(i+1)*batch_size] += forecast_i[:,0,0,:]

        return backcast, forecast

class NHitsBlock(nn.Module):
    def __init__(self, n_time_in, n_time_out, n_theta, n_theta_hidden, n_pool_kernel_size, basis, n_layers, activation):
        super().__init__()
        self.n_time_in = n_time_in
        self.n_time_out = n_time_out
        self.n_pool_kernel_size = n_pool_kernel_size
        n_time_in_pooled = int(np.ceil(n_time_in/n_pool_kernel_size))
        n_theta_hidden = [n_time_in_pooled] + n_theta_hidden

        activ = getattr(nn, activation)()

        self.pooling_layer = nn.MaxPool1d(kernel_size=self.n_pool_kernel_size,
                                              stride=self.n_pool_kernel_size, ceil_mode=True)

        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(nn.Linear(in_features=n_theta_hidden[i], out_features=n_theta_hidden[i+1]))
            hidden_layers.append(activ)
        output_layer = [nn.Linear(in_features=n_theta_hidden[-1], out_features=n_theta)]
        layers = hidden_layers+output_layer
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pooling_layer(x)
        x = x.squeeze(1)

        theta = self.layers(x)
        backcast, forecast = self.basis(theta)
        return backcast, forecast

class NHits(nn.Module):
    def __init__(self, SOURCE_SIZE, TARGET_SIZE, n_stacks = 3, 
                        n_blocks = [1,1,1], n_layers = [2,2,2], n_theta_hidden = 3*[2*[512]], 
                        n_pool_kernel_size = [16,8,1], n_freq_downsample = [9,6,1], activation = 'ReLU'):
        super().__init__()
        self.n_time_in = SOURCE_SIZE
        self.n_time_out = TARGET_SIZE
        self.n_stacks = n_stacks
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_theta_hidden = n_theta_hidden
        self.n_pool_kernel_size = n_pool_kernel_size
        self.n_freq_downsample = n_freq_downsample
        self.activation = activation

        stacks = self.create_blocks()
        self.blocks = nn.ModuleList(stacks)

    def create_blocks(self):
        blocks = []
        for i in range(self.n_stacks):
            for block_id in range(self.n_blocks[i]):
                n_theta = self.n_time_in + max(self.n_time_out//self.n_freq_downsample[i], 1)
                basis = IdentityBasis(self.n_time_in, self.n_time_out, 'linear')
                block = NHitsBlock(self.n_time_in, self.n_time_out, n_theta, 
                                   self.n_theta_hidden[i], self.n_pool_kernel_size[i], 
                                   basis, self.n_layers[i], self.activation)
                blocks.append(block)
        return blocks

    def forward(self, x): 
        residual = x.flip(dims=(-1,))
        forecast = x[:,-1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return forecast
