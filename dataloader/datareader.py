import pandas as pd
from sklearn import preprocessing


class DataReader:

    def __init__(self, cfg):
        self.cfg = cfg

    def load_and_standardize_data(self):
        path = self.cfg.data.path + self.cfg.data.folder
        df = pd.read_pickle(path + ".pkl", compression="zip")
        df = df[df["time"] <= self.cfg.data.frame]
        df = df.values.reshape(-1, df.shape[1])
        x_train = df[df[:, 6] % 2 != 0]
        x_test = df[df[:, 6] % 2 == 0]
        print("train length: ", len(x_train))
        print("test length: ", len(x_test))
        scalar = preprocessing.StandardScaler()
        x_train = scalar.fit_transform(x_train)
        x_test = scalar.transform(x_test)

        return x_train, x_test, scalar, path
