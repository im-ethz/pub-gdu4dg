import numpy as np
import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, df):

        self.inputs = torch.tensor(np.stack(df['inputs'].values), dtype=torch.float32)
        self.hypo = torch.tensor(np.stack(df['hypoglycemia'].values), dtype=torch.float32)
        self.hyper = torch.tensor(np.stack(df['hyperglycemia'].values), dtype=torch.float32)
        self.nocturnal = torch.tensor(np.stack(df['nocturnal_sampled'].values), dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx].view(-1, 1), self.hypo[idx], self.hyper[idx], self.nocturnal[idx]


class NocturnalDataset(Dataset):
    def __init__(self, df, window, period_length=1, scale=False, sampled_labels=False, standardise=False, params=None):

        assert isinstance(period_length, int), "period_length must be int"
        self.sampling_frequency = period_length
        self.points_per_window = int(window * 60 / period_length) + 1

        self.sampled_labels = sampled_labels

        inputs = np.stack(df.prenocturnal.values)

        if scale:
            # the scaling is done as in other papers
            inputs = inputs * 0.01 * 18

        if standardise:

            if params:
                self.mu = params[0]
                self.sigma = params[1]
            else:
                self.mu = np.mean(inputs)
                self.sigma = np.std(inputs)

            inputs = (inputs - self.mu) / self.sigma

        self.inputs = torch.tensor(inputs, dtype=torch.float32)

        if sampled_labels:
            self.targets = torch.tensor(np.stack(df['nocturnal_resampled'].values))
        else:
            self.targets = torch.tensor(np.stack(df['augmented_nocturnal'].values))

        if scale:
            self.targets = self.targets * 0.01 * 18

        self.mask = torch.tensor(np.stack(df['mask'].values))
        self.lengths = torch.tensor(np.stack(df['length'].values))

        self.times = list(df['nocturnal_times'].values)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        prenocturnal = self.inputs[idx, -self.points_per_window::self.sampling_frequency].view(-1, 1)
        targets = self.targets[idx]
        mask = self.mask[idx]
        length = self.lengths[idx]
        # times = self.times[idx]
        return prenocturnal, targets, mask, length


class ShortTermDataset(Dataset):
    def __init__(self, df, input_length, horizon, scale=False):

        if input_length % 5 != 0:
            raise UserWarning(f"The input length is {input_length} but should be divisible by"
                              f"the sampling period length, i.e. a multiple of 5")
        if horizon % 5 != 0:
            raise UserWarning(f"The prediction horizon is {horizon} but should be divisible by"
                              f" the sampling period length, i.e. a multiple of 5.")

        points_input = int(input_length / 5)
        points_prediction = int(horizon / 5)

        glucose_values = np.stack(df['resampled_values'].values)

        self.inputs = torch.tensor(glucose_values[:, -(points_input + points_prediction):-points_prediction],
                                   dtype=torch.float32)
        self.targets = torch.tensor(glucose_values[:, -points_prediction:], dtype=torch.float32)

        if scale:
            self.inputs = self.inputs * 0.01 * 18
            self.targets = self.targets * 0.01 * 18

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx].view(-1, 1), self.targets[idx]
