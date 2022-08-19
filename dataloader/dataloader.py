import torch
from torch.utils.data import DataLoader as Mapper


class DataLoader:

    def __init__(self, databuilder, batch_size):
        self.databuilder = databuilder
        self.batch_size = batch_size

    def loader(self):
        loader = Mapper(dataset=self.databuilder, batch_size=self.batch_size, shuffle=True)
        return loader
