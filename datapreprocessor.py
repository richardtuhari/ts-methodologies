import torch
import numpy as np
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DataPreprocessor(torch.utils.data.Dataset):
    """
    input: dataset_path, use = [timestamp, data], transform, device, source_size ,target_size

    output: torch Dataset
    """
    
    def __init__(self, use, transform, device, source_size, target_size, dataset = None, dataset_path = None, start=0, present=100, bias = 'SRC_BIAS'):
        self.dataset_path = dataset_path
        self.data = dataset
        self.use = use
        self.start = start
        self.present = present
        self.source_size = source_size
        self.target_size = target_size
        self.transform = transform
        self.dataset_mean = 0
        self.dataset_std = 0
        self.bias = bias

    def create_training_dataset(self):
        
        if self.dataset_path:
            dataset = self._load_dataset()
        else: 
            dataset = pd.DataFrame(self.data)
            dataset.rename(columns={0:use[-1]}, inplace=True)
     
        dataset = np.array(dataset[self.use[-1]].to_list())
        self.dataset_mean = dataset.mean()
        self.dataset_std = dataset.std()
        input_sequences, target_sequences, transformed_input_sequences, transformed_target_sequences  = self._create_sequence_pairs(dataset,self.source_size, self.target_size)
        training_dataset = self._convert_to_dataset(input_sequences, target_sequences)
        transformed_training_dataset = self._convert_to_dataset(transformed_input_sequences, transformed_target_sequences)

        return(training_dataset,transformed_training_dataset)

    def get_meanandstd(self):
        return self.dataset_mean, self.dataset_std

    def _load_dataset(self):
            return pd.read_csv(self.dataset_path, usecols=self.use)

    def _create_sequence_pairs(self, dataset, source_size, target_size):
        input_sequences, target_sequences, transformed_input_sequences, transformed_target_sequences = np.empty((0,source_size), float), np.empty((0,target_size), int), np.empty((0,source_size), int),np.empty((0,target_size), int)
        for i in range(self.start, self.present+target_size-source_size):
            EOH = i+source_size
            input_seq = dataset[i:EOH]
            target_seq = dataset[EOH:EOH+target_size]
            padded_input_seq = self._pad_input_sequence(input_seq)
            padded_target_seq = self._pad_target_sequence(target_seq)
            input_sequences = np.append(input_sequences, padded_input_seq.reshape(1, source_size), axis=0)
            target_sequences = np.append(target_sequences, padded_target_seq.reshape(1, target_size), axis=0)
            #local transform

            if self.bias == 'SRC_BIAS':
                self.dataset_mean = dataset[i:EOH].mean()
                self.dataset_std = dataset[i:EOH].std()
            if self.bias == 'TRG_BIAS':
                self.dataset_mean = dataset[i:EOH+target_size].mean()
                self.dataset_std = dataset[i:EOH+target_size].std()
                
            transformed_dataset =  self.transform(dataset[i:EOH+target_size], self.dataset_mean, self.dataset_std)
            transformed_input_seq = transformed_dataset[:source_size]
            transformed_target_seq = transformed_dataset[-target_size:]
            padded_transformed_input_seq = self._pad_input_sequence(transformed_input_seq)
            padded_transformed_target_seq = self._pad_target_sequence(transformed_target_seq)
            transformed_input_sequences = np.append(transformed_input_sequences, padded_transformed_input_seq.reshape(1, source_size), axis=0)
            transformed_target_sequences = np.append(transformed_target_sequences, padded_transformed_target_seq.reshape(1, target_size), axis=0)
        return input_sequences, target_sequences, transformed_input_sequences, transformed_target_sequences
            
    def _pad_input_sequence(self, sequence):
        return np.pad(sequence, (0,(self.source_size - len(sequence))), 'constant', constant_values=(0, 0))

    def _pad_target_sequence(self, sequence):
        return np.pad(sequence, (0,(self.target_size - len(sequence))), 'constant', constant_values=(0, 0))

    def _convert_to_dataset(self, input_sequences, target_sequences):
        return torch.utils.data.TensorDataset(torch.tensor(input_sequences), torch.tensor(target_sequences))
