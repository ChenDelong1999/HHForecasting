import pandas as pd
import numpy as np
import torch


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, forecast_range, dataset, mode, train_test_split_ratio, sample_length):

        assert mode in ['train', 'test']
        assert forecast_range >= 1
        self.mode = mode
        self.sample_length = sample_length
        self.forecast_range = forecast_range
        self.overlapping_split = 20

        csv_file = f'dataset/{dataset}/data.csv'
        print('--- reading', csv_file)
        df = pd.read_csv(csv_file)
        # print(df)
        df = df.dropna()
        data = np.array(df)

        X = data[:-self.forecast_range, :]
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))  # normalization
        Y = data[self.forecast_range:, 0]

        spilt = int(train_test_split_ratio * len(X))
        if self.mode == 'train':
            self.inputs = X[:spilt, :]
            self.targets = Y[:spilt]
        elif self.mode == 'test':
            self.inputs = X[spilt:, :]
            self.targets = Y[spilt:]

        print(f'--- loaded [{dataset}] [{self.mode}] set: '
              f'input shape: {self.inputs.shape}, '
              f'target shape: {self.targets.shape}, len(dataset)={len(self)}')

    def __getitem__(self, index):
        start = int(index / self.overlapping_split * self.sample_length)  # 20-overlapping samples
        raindrop = self.inputs[start:start + self.sample_length]
        runoff = self.targets[start:start + self.sample_length]

        return torch.from_numpy(raindrop), torch.from_numpy(runoff)

    def __len__(self):
        return int(len(self.inputs) / self.sample_length * self.overlapping_split) - self.overlapping_split

    def get_input_size(self):
        return self.inputs.shape[1]
