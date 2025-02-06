import os
import torch
import itertools

from models_b import LST, LIN, predictor, SegRNN, DLinear, NHits, NBeats, PatchTST
from train import *

from time import time
import pickle


# Set your run list with any model and with any parameter combination
D = {
        'LST': {'DATASET_PATH':  ['cut/AMZN.csv','cut/electricity.csv','cut/ETTh1.csv','cut/ETTh2.csv', 'cut/ETTm1.csv', 'cut/ETTm2.csv', 'cut/exchange_rate.csv', 'cut/ILINet.csv', 'cut/traffic.csv', 'cut/weather.csv'], 'HISTORY_SIZE':[500], "NUM_EPOCHS":[1], 'SOURCE_SIZE': [5], 'TARGET_SIZE':[1], 'BIAS':['SRC_BIAS','TRG_BIAS']},
        #'DLinear': {'DATASET_PATH':  ['cut/AMZN.csv','cut/electricity.csv','cut/ETTh1.csv','cut/ETTh2.csv', 'cut/ETTm1.csv', 'cut/ETTm2.csv', 'cut/exchange_rate.csv', 'cut/ILINet.csv', 'cut/traffic.csv', 'cut/weather.csv'], 'HISTORY_SIZE':[100,500,1000], "NUM_EPOCHS":[1,2,5,10], 'SOURCE_SIZE': [10,14], 'TARGET_SIZE':[1,2,5,7], 'INDIVIDUAL':[False], 'N_VARIATE':[1], 'BIAS':['SRC_BIAS', 'TRG_BIAS']},
        #'NBeats': {'DATASET_PATH':  ['cut/AMZN.csv','cut/electricity.csv','cut/ETTh1.csv','cut/ETTh2.csv', 'cut/ETTm1.csv', 'cut/ETTm2.csv', 'cut/exchange_rate.csv', 'cut/ILINet.csv', 'cut/traffic.csv', 'cut/weather.csv'], 'HISTORY_SIZE':[100], "NUM_EPOCHS":[10], 'SOURCE_SIZE': [10,14], 'TARGET_SIZE':[1,2,5,7], 'stacks':[10], 'layers':[4], 'layer_size':[512], 'BIAS':['SRC_BIAS', 'TRG_BIAS']},       
        #'LIN': {'DATASET_PATH':  ['cut/AMZN.csv'], 'HISTORY_SIZE':[500,1000], "NUM_EPOCHS":[1,2,5,10], 'SOURCE_SIZE': [5,7,10], 'TARGET_SIZE':[1,2,3], 'BIAS':['SRC_BIAS', 'TRG_BIAS']},
        #'SegRNN': {'DATASET_PATH':  ['cut/AMZN.csv','cut/electricity.csv','cut/ETTh1.csv','cut/ETTh2.csv', 'cut/ETTm1.csv', 'cut/ETTm2.csv', 'cut/exchange_rate.csv', 'cut/ILINet.csv', 'cut/traffic.csv', 'cut/weather.csv'], 'HISTORY_SIZE':[100], "NUM_EPOCHS":[10], 'SOURCE_SIZE': [10,15], 'TARGET_SIZE':[1,2,5,7], 'SEGMENT_SIZE':[5], 'N_VARIATE':[1], 'HIDDEN_SIZE':[512], 'DROPOUT':[0], 'BIAS':['SRC_BIAS', 'TRG_BIAS']},
        #'PatchTST': {'DATASET_PATH':  ['data/AMZN.csv','data/standardized_preprocessed_data.csv','data/weather.csv','data/exchange_rate.csv','data/ETTh1.csv','data/ETTh2.csv','data/ETTm2.csv'], 'HISTORY_SIZE':[100,200], "NUM_EPOCHS":[1,2], 'SOURCE_SIZE': [7,14,28], 'TARGET_SIZE':[1,2]},
        #'NHits': {'DATASET_PATH':  ['cut/AMZN.csv','cut/electricity.csv','cut/ETTh1.csv','cut/ETTh2.csv', 'cut/ETTm1.csv', 'cut/ETTm2.csv', 'cut/exchange_rate.csv', 'cut/ILINet.csv', 'cut/traffic.csv', 'cut/weather.csv'], 'HISTORY_SIZE':[100,500], "NUM_EPOCHS":[1,10], 'SOURCE_SIZE': [10,14], 'TARGET_SIZE':[1,2,5,7], 'BIAS':['SRC_BIAS', 'TRG_BIAS']},
}


lut = {
    'LST': (LST, ['SOURCE_SIZE', 'TARGET_SIZE']),
    'LIN': (LIN, ['SOURCE_SIZE', 'TARGET_SIZE']),
    'predictor': (predictor, ['SOURCE_SIZE', 'TERMS']),
    'SegRNN' : (SegRNN, ['SOURCE_SIZE', 'TARGET_SIZE', 'SEGMENT_SIZE', 'N_VARIATE', 'HIDDEN_SIZE', 'DROPOUT']),
    'DLinear' : (DLinear, ['SOURCE_SIZE', 'TARGET_SIZE', 'INDIVIDUAL', 'N_VARIATE']),
    'NBeats' : (NBeats, ['SOURCE_SIZE', 'TARGET_SIZE', 'stacks', 'layers', 'layer_size']),
    'NHits' : (NHits, ['SOURCE_SIZE', 'TARGET_SIZE']),
    'PatchTST' : (PatchTST, ['SOURCE_SIZE', 'TARGET_SIZE'])
}


dataset_label = lambda pth: pth.split('/')[-1].split('.')[0]

def smape(target, output):
            n = len(output)
            return sum(sum((2/n)*((target-output).abs())/(torch.max(target.abs()+output.abs(),torch.full_like(output, 0.0001)))))
            

#def smape(y_pred, y_true, epsilon=0.0001):
#    """
#    Computes the Symmetric Mean Absolute Percentage Error (SMAPE).
#
#    Args:
#        y_pred (torch.Tensor): Predicted values.
#        y_true (torch.Tensor): True values.
#        epsilon (float): Small value to avoid division by zero.
#
#    Returns:
#        torch.Tensor: SMAPE score.
#    """
#    numerator = torch.abs(y_pred - y_true)
#    denominator = (torch.abs(y_true) + torch.abs(y_pred) + epsilon) / 2
#    return torch.mean(numerator / denominator)


for m, parset in D.items():
    for id_s, element in enumerate(itertools.product(*parset.values())):
        M, M_PARS = lut[m]
        PARAM_DICT = dict(zip(parset.keys(), element))
        _ds=dataset_label(PARAM_DICT['DATASET_PATH'])
        t0=time()

        R = dict(filter(lambda x: x[0] in M_PARS ,PARAM_DICT.items()))
        current_set = (id_s, m, dict(zip(parset.keys(), element)))
        # data config
        use = ['Date', 'data']
        num_epochs = current_set[2]['NUM_EPOCHS']
        dataset_path = current_set[2]['DATASET_PATH']
        SOURCE_SIZE = current_set[2]['SOURCE_SIZE']
        TARGET_SIZE = current_set[2]['TARGET_SIZE']
        HISTORY_SIZE = current_set[2]['HISTORY_SIZE']
        BIAS = current_set[2]['BIAS']
        transform = zscore

        # train config
        k_folds = 5

        loss_function = smape

        MODEL = M(**R)
        test_losses, sliding_fold_train_losses, sliding_fold_val_losses, model, testpreds = train(use, transform, device, SOURCE_SIZE, TARGET_SIZE, HISTORY_SIZE, dataset_path, k_folds, num_epochs, loss_function, MODEL, BIAS)

        #saving procedure
        filename = f'{m}-{_ds}-{t0}-{time()}'
        pickle_path = os.path.join("parameters", f'{filename}.pickle')
        with open(pickle_path, 'wb') as fo:
            pickle.dump(PARAM_DICT, fo, protocol=pickle.HIGHEST_PROTOCOL)
        
        #test_losses
        path = os.path.join("saved_test_losses", f'{filename}.csv')
        with open(path, "w") as f:
            for s in test_losses:
                f.write(str(s) +"\n")
                
        #model
        path = os.path.join("saved_models", f'{filename}.pth')
        torch.save(model.state_dict(), path)

        #sliding_losses
        path = os.path.join("saved_sliding_fold_train_losses", f'{filename}.csv')
        with open(path, "w") as f:
            for train_losses in sliding_fold_train_losses:
                for s in train_losses:
                    f.write(str(s[0]) +",")
                f.write("\n")

        path = os.path.join("saved_sliding_fold_val_losses", f'{filename}.csv')
        with open(path, "w") as f:
            for train_losses in sliding_fold_val_losses:
                for s in train_losses:
                    f.write(str(s[0]) +",")
                f.write("\n")
                
        #pedictionss
        path = os.path.join("saved_testpreds", f'{filename}.csv')
        testpreds.to_csv(path, index=False)




