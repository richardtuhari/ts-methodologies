import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

import pandas as pd
from tqdm import tqdm
from math import isclose
import matplotlib.pyplot as plt

from datapreprocessor import DataPreprocessor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    #print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def identitiy(x, m, s):
    return x
    
def zscore(x, m, s):
    r = x - m
    return r/s

def cumTrans(x, mean, std):
    cumulated = np.cumsum(x)
    X = np.array(list(range(len(cumulated))))
    a = np.dot(X, cumulated)/np.dot(X,X)
    y = a*X
    return cumulated-y

def train(use, transform, device, SOURCE_SIZE, TARGET_SIZE, HISTORY_SIZE, dataset_path, k_folds, num_epochs, loss_function, MODEL, BIAS):
    results = {}
    test_losses = []
    sliding_fold_train_losses = []
    sliding_fold_val_losses = []
    testpreds = pd.DataFrame([])
    test_loss = 0
    DAYS = len(pd.read_csv(dataset_path))
    for l in tqdm(range(DAYS-HISTORY_SIZE-TARGET_SIZE)):
        preprocessor = DataPreprocessor(use, transform, device, SOURCE_SIZE, TARGET_SIZE, None, dataset_path, start=0+l, present=HISTORY_SIZE+l, bias=BIAS)
        original_dataset, transformed_dataset = preprocessor.create_training_dataset()
        original_dataset_cp, transformed_dataset_cp = original_dataset, transformed_dataset

        original_dataset = [[original_dataset[i][0], original_dataset[i][1]] for i in range(len(original_dataset)-1)]
        transformed_dataset = [[transformed_dataset[i][0], transformed_dataset[i][1]] for i in range(len(transformed_dataset)-1)]

        # K-fold Cross Validator
        kfold = KFold(n_splits=k_folds, shuffle=True)
         
        fold_train_losses, fold_val_losses = [], []
        for fold, (train_ids, test_ids) in enumerate(kfold.split(original_dataset)):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader = torch.utils.data.DataLoader(list(zip(transformed_dataset,original_dataset)),batch_size=16, sampler=train_subsampler)
            valloader = torch.utils.data.DataLoader(list(zip(transformed_dataset,original_dataset)),batch_size=16, sampler=val_subsampler)

            # Init the neural network (and reinit)
            model = MODEL
            model.apply(reset_weights)
            if l>0:
                model.load_state_dict(torch.load(f'./model-fold-{last_best}.pth'))
            model = model.to(device)
            
            # Initialize optimizer (and reinit)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            train_losses, val_losses = [], []
            for epoch in range(0, num_epochs):
                model.train()
                train_loss = 0.0
                for i, data in enumerate(trainloader, 0):  
                    (inputs, targets), (o_inputs, o_targets) = data
                    inputs, targets = inputs.to(device).to(torch.float32), targets.to(device).to(torch.float32)
                    o_inputs, o_targets = o_inputs.to(device).to(torch.float32), o_targets.to(device).to(torch.float32)

                    optimizer.zero_grad()
                    outputs = model(inputs)

                    if BIAS == 'SRC_BIAS':
                        o_data = o_inputs
                    if BIAS == 'TRG_BIAS':
                        o_data = torch.cat((o_inputs,o_targets),dim=1)
                    m = o_data.mean(dim=1)
                    s = o_data.std(dim=1, unbiased=False)

                    loss = loss_function((outputs.T*s+m).T, o_targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
        
                
                train_loss /= i
                train_losses.append(train_loss)
        
                model.eval()
                val_loss = 0.0
                correct, total = 0, 0
                with torch.no_grad():
                    for i, data in enumerate(valloader, 0):    
                        (inputs, targets), (o_inputs, o_targets) = data
                        inputs, targets = inputs.to(device).to(torch.float32), targets.to(device).to(torch.float32)
                        o_inputs, o_targets = o_inputs.to(device).to(torch.float32), o_targets.to(device).to(torch.float32)
                        
                        outputs = model(inputs)

                        if BIAS == 'SRC_BIAS':
                            o_data = o_inputs
                        if BIAS == 'TRG_BIAS':
                            o_data = torch.cat((o_inputs,o_targets),dim=1)
                        m = o_data.mean(dim=1)
                        s = o_data.std(dim=1, unbiased=False)

                        loss = loss_function((outputs.T*s+m).T, o_targets)
        
                        C = 0
                        C = ((abs(outputs-targets)).sum(axis=1) < .05*min((targets).sum(axis=1))).sum()
                        correct += C
                        total += len(targets)
                        
                        val_loss += loss.item()
                        
                val_loss /= i+1
                
                val_losses.append(val_loss)
            results[fold] = 100.0 * (correct / total)        
        
                
            save_path = f'./model-fold-{fold}.pth'
            torch.save(model.state_dict(), save_path)
            fold_train_losses.append(train_losses)
            fold_val_losses.append(val_losses)
    
    
        sliding_fold_train_losses.append(fold_train_losses)
        sliding_fold_val_losses.append(fold_val_losses)
        
        last_best = max(results, key=results.get)
        model = MODEL
        model.load_state_dict(torch.load(f'./model-fold-{last_best}.pth'))

        o_test_input, o_test_target = original_dataset_cp[-1:]
        test_input, test_target = transformed_dataset_cp[-1:]
        
        o_test_input, o_test_target = o_test_input.to(device).to(torch.float32), o_test_target.to(device).to(torch.float32)
        test_input, test_target = test_input.to(device).to(torch.float32), test_target.to(device).to(torch.float32)
        test_output = model(test_input)
        
        if BIAS == 'SRC_BIAS':
            o_data = o_test_input
        if BIAS == 'TRG_BIAS':
            o_data = torch.cat((o_test_input,o_test_target),dim=1)
        m = o_data.mean(dim=1)
        s = o_data.std(dim=1, unbiased=False)

        df1 = pd.DataFrame((test_output.T*s+m).T.cpu().detach().numpy())
        df2 = pd.DataFrame(o_test_target.cpu().detach().numpy())
        result_df = pd.concat([df1, df2], axis=1, ignore_index=True)
        testpreds = pd.concat([testpreds, result_df], axis=0, ignore_index=True)
        test_loss = loss_function((test_output.T*s+m).T, o_test_target)
        test_losses.append(test_loss.item())

    return test_losses, sliding_fold_train_losses, sliding_fold_val_losses, model, testpreds
