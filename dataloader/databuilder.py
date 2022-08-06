import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset


class DataBuilder:
    @staticmethod
    def load_and_standardize_data(path):
        df = pd.read_pickle(path + ".pkl", compression="zip")
        df = df.values.reshape(-1, df.shape[1])
        x_train = df[df[:, 6] % 2 != 0]
        x_test = df[df[:, 6] % 2 == 0]
        scalar = preprocessing.StandardScaler()
        x_train = scalar.fit_transform(x_train)
        x_test = scalar.transform(x_test)

        return x_train, x_test, scalar, path

    def __init__(self, path, train=True):
        self.x_train, self.x_test, self.standardizer, self.path = self.load_and_standardize_data(path)
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

    def train_data(self, batch_size):
        train_set = DataBuilder(self.path, train=True)
        train_loader = DataLoader(dataset= train_set, batch_size=batch_size)
        return train_loader

    def test_data(self, batch_size):
        test_set = DataBuilder(self.path, train=False)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size)
        return test_loader
