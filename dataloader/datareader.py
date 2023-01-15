import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class DataReader:

    def __init__(self, cfg):
        self.cfg = cfg

    def load_and_standardize_data(self, random_split=False):
        path = self.cfg.data.path + self.cfg.data.folder
        df = pd.read_pickle(path + ".pkl", compression="zip")
        df = df[df["time"] >= self.cfg.data.start_frame]
        df = df[df["time"] <= self.cfg.data.end_frame]
        df = df.values.reshape(-1, df.shape[1])
        if random_split:
            x_train, x_test = train_test_split(df, test_size=self.cfg.data.test_size , random_state=42)
        else:
            x_train = df[df[:, self.cfg.data.feature_no] % self.cfg.data.split_fraction != 0]
            x_test = df[df[:, self.cfg.data.feature_no] % self.cfg.data.split_fraction == 0]
        print("train length: ", len(x_train))
        print("test length: ", len(x_test))
        scalar = preprocessing.StandardScaler()
        x_train = scalar.fit_transform(x_train)
        x_test = scalar.transform(x_test)

        return x_train, x_test, scalar, path
