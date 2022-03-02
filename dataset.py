import pandas as pd
import numpy as np
import torch


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self,
                 forecast_range,
                 dataset,
                 mode,
                 train_test_split_ratio=0.7,
                 sample_length=72,
                 training_set_scale=1,
                 training_set_start=0,
                 few_shot_num=10):

        assert mode in ['train', 'test']
        assert forecast_range >= 1
        self.mode = mode
        self.sample_length = sample_length
        self.forecast_range = forecast_range
        self.training_set_scale = training_set_scale
        self.overlapping_split = 20

        csv_file = f'dataset/{dataset}/data.csv'
        print('--- reading', csv_file)
        df = pd.read_csv(csv_file)
        df = df.dropna()
        data = np.array(df)

        raindrop = data[:-self.forecast_range, 1:]
        raindrop = (raindrop - np.min(raindrop, axis=0)) / (np.max(raindrop, axis=0) - np.min(raindrop, axis=0))
        runoff_history = data[:-self.forecast_range, 0]
        runoff = data[self.forecast_range:, 0]

        train_test_spilt = int(train_test_split_ratio * len(raindrop))
        training_subset = int(train_test_spilt * self.training_set_scale)
        if few_shot_num is not None:
            start_range = int((train_test_spilt - training_subset) / (few_shot_num - 1))
            offset = int(training_set_start * start_range)
        else:
            offset = 0

        if self.mode == 'train':
            self.raindrop = raindrop[offset:training_subset+offset, :]
            self.runoff_history = runoff_history[offset:training_subset+offset]
            self.runoff = runoff[offset:training_subset+offset]
        elif self.mode == 'test':
            self.raindrop = raindrop[train_test_spilt:, :]
            self.runoff_history = runoff_history[train_test_spilt:]
            self.runoff = runoff[train_test_spilt:]

        print(f'--- loaded [{dataset}] [{self.mode}] set: '
              f'input shape: {self.raindrop.shape}, '
              f'target shape: {self.runoff.shape}, len(dataset)={len(self)}')

    def __getitem__(self, index):
        start = int(index / self.overlapping_split * self.sample_length)  # 20-overlapping samples
        return torch.from_numpy(self.raindrop[start:start + self.sample_length]), \
               torch.from_numpy(self.runoff_history[start:start + self.sample_length]), \
               torch.from_numpy(self.runoff[start:start + self.sample_length])

    def __len__(self):
        return int(len(self.raindrop) / self.sample_length * self.overlapping_split) - self.overlapping_split

    def get_input_size(self):
        return self.raindrop.shape[1]
