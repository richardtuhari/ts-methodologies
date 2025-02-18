#! /usr/bin/python3.11

import numpy as np
import torch
import torch.nn as nn
import argparse

from model_mlp import LIN
from model_lst import LST
from transformer import *
from wrapper import WrappedModel

pre_train=1000
warm_epochs=100
retrain_epochs=1
lr=1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype=torch.float32
try:
    print("#", device, torch.cuda.get_device_name(0), flush=True)
except:
    pass

def smape(y_pred, y_true, epsilon=eps):
    """
    Computes the Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: SMAPE score.
    """
    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_true) + torch.abs(y_pred) + epsilon) / 2
    return torch.mean(numerator / denominator)


class SMAPELoss(torch.nn.Module):
    def __init__(self, epsilon=eps):
        super(SMAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        return smape(y_pred, y_true, self.epsilon)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--history", action = "store", type=int, default=5,
                    help = "history size")
    parser.add_argument("-S", "--sourcesize", action = "store", type=int, default=5,
                    help = "model's input size")
    parser.add_argument("-P", "--targetsize", action = "store", type=int, default=1,
                    help = "prediction window")
    parser.add_argument("-i", "--dataset", action = "store", default="amzn.npy", 
                     help = "dataset (numpy binary)")
    parser.add_argument("-o", "--trainloss", action = "store", default=None,
                     help = "where trainloss data to store")
    parser.add_argument("-l", "--testloss", action = "store", default=None,
                     help = "where testloss data to store")
    parser.add_argument("-T", "--transformer", action = "store", required=True, 
                     choices=['Z_', 'Z_oS', 'Z_oFoS', 'Z_oS-', 'Z_oFoS-', 'Z', 'ZoS', 'ZoFoS', 'ZoS-', 'ZoFoS-', 'S', 'SoF', 'S-', 'S-oF'],
                     help = "what sort of transformation to use on dataset. (Z: Z-score transform, F: tanh, S: scaler, _: future bias, -: decoupled")
    parser.add_argument("-m", "--model", action = "store", default='LIN',
                     choices=['LIN', 'LST'],
                     help = "what model to use")
    args = parser.parse_args()

    data=np.load(args.dataset)
    print("# loaded", data.shape, flush=True)
    S=torch.tensor(data, dtype=dtype).to(device)
    print("# tensor created", S.shape, flush=True)

    mapper=torch.tanh
    mapper_inv=lambda x: torch.atanh(torch.clamp(x, min=eps-1, max=1-eps))

    if args.model=='LIN':
        model=LIN(SOURCE_SIZE=args.sourcesize, TARGET_SIZE=args.targetsize, dtype=dtype)
    elif args.model=='LIST':
        model=LST(SOURCE_SIZE=args.sourcesize, TARGET_SIZE=args.targetsize, dtype=dtype)

    offset_future=args.targetsize if 'Z_' in args.transformer else 0
    if args.transformer=='Z_':
        trafo=TransformFuture(tail=args.sourcesize, head=args.targetsize, affine=False)
    elif args.transformer=='Z_oS':
        trafo=TransformFuture(tail=args.sourcesize, head=args.targetsize, affine=True)
    elif args.transformer=='Z_oFoS':
        trafo=TransformFuture(tail=args.sourcesize, head=args.targetsize, affine=True, mapper=mapper, mapper_inv=mapper_inv)
    elif args.transformer=='Z_oS-':
        trafo=TransformFutureDecoupled(tail=args.sourcesize, head=args.targetsize, affine=True)
    elif args.transformer=='Z_oFoS-':
        trafo=TransformFutureDecoupled(tail=args.sourcesize, head=args.targetsize, affine=True, mapper=mapper, mapper_inv=mapper_inv)

    elif args.transformer=='Z':
        trafo=Transform(tail=args.sourcesize, affine=False)
    elif args.transformer=='ZoS':
        trafo=Transform(tail=args.sourcesize, affine=True)
    elif args.transformer=='ZoFoS':
        trafo=Transform(tail=args.sourcesize, affine=True, mapper=mapper, mapper_inv=mapper_inv)
    elif args.transformer=='ZoS-':
        trafo=TransformDecoupled(tail=args.sourcesize, affine=True)
    elif args.transformer=='ZoFoS-':
        trafo=TransformDecoupled(tail=args.sourcesize, affine=True, mapper=mapper, mapper_inv=mapper_inv)

    elif args.transformer=='S':
        trafo=TransformNoZ(tail=args.sourcesize)
    elif args.transformer=='SoF':
        trafo=TransformNoZ(tail=args.sourcesize, mapper=mapper, mapper_inv=mapper_inv)
    elif args.transformer=='S-':
        trafo=TransformNoZDecoupled(tail=args.sourcesize)
    elif args.transformer=='S-oF':
        trafo=TransformNoZDecoupled(tail=args.sourcesize, mapper=mapper, mapper_inv=mapper_inv)

    super_model = WrappedModel(model=model, trafo=trafo).to(device)
    optimizer = torch.optim.Adam(super_model.parameters(), lr=lr)
    #loss = nn.MSELoss()
    loss = SMAPELoss()

# pretrain
    print("# pretrain")
    super_model.train()
    optimizer.zero_grad()
    train_loss=[]
    for epoch in range(warm_epochs):
        tl = 0.0
        for i in range(pre_train):
            X=S[i:i+args.history+offset_future].reshape(1, args.history+offset_future)
            Y=S[i+args.history:i+args.history+args.targetsize].reshape(1,args.targetsize)
    
            Y_ = super_model(X)
            l = loss(Y, Y_)
    
            l.backward()
            optimizer.step()
            tl += l.item()
        train_loss.append(tl / pre_train)

# test
    print("# test")

    test_loss=[]
    super_model.eval()
    with torch.no_grad():
        X=S[pre_train:pre_train+args.history+offset_future].reshape(1, args.history+offset_future)
        Y=S[pre_train+args.history:pre_train+args.history+args.targetsize].reshape(1,args.targetsize)
    
        Y_ = super_model(X)
        l = loss(Y, Y_)
        test_loss.append(l.item())
    print (Y.item(), Y_.item(), l.item(), *trafo._vars)

# walk forward
    print("# walkforward")

    for i in range(pre_train, S.shape[0] - args.history - args.targetsize):
        # train
        super_model.train()
        optimizer.zero_grad()
        for epoch in range(retrain_epochs):
            tl = 0.0
            for j in range(pre_train):
                X=S[i-j:i-j+args.history+offset_future].reshape(1, args.history+offset_future)
                Y=S[i-j+args.history:i-j+args.history+args.targetsize].reshape(1,args.targetsize)
    
                Y_ = super_model(X)
                l = loss(Y, Y_)
    
                l.backward()
                optimizer.step()
                tl += l.item()
            train_loss.append(tl / pre_train)
        # evaluate
        super_model.eval()
        with torch.no_grad():
            X=S[i+1:i+1+args.history+offset_future].reshape(1, args.history+offset_future)
            Y=S[i+1+args.history:i+1+args.history+args.targetsize].reshape(1,args.targetsize)
    
            Y_ = super_model(X)
            l = loss(Y, Y_)
            test_loss = l.item()
        print (Y.item(), Y_.item(), l.item(), *trafo._vars)

    if args.trainloss:
        np.save(args.trainloss, np.array(train_loss))
    if args.testloss:
        np.save(args.testloss, np.array(test_loss))
