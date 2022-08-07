import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset


class DataBuilder:

    def __init__(self, data, train=True):
        self.x_train, self.x_test, self.standardizer, self.path = data
        if train:
            self.x = torch.from_numpy(self.x_train).float()
        else:
            self.x = torch.from_numpy(self.x_test).float()

        self.len = self.x.shape[0]
        del self.x_train
        del self.x_test

    def __getitem__(self, item):
        return self.x[item]

    def __len__(self):
        return self.len