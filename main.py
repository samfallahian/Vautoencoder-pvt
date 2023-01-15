import pandas as pd

from configs import config as cfg
from utils.config import Config, Log
from dataloader.databuilder import DataBuilder
from dataloader.dataloader import DataLoader
from dataloader.datareader import DataReader
from executor.executor import Executor
from datetime import datetime


def main():
    logger = Log()
    conf = Config.from_json(cfg.CFG)

    data_reader = DataReader(conf)

    data = data_reader.load_and_standardize_data(random_split=True)

    train_data_set = DataBuilder(data=data, train=True)
    test_data_set = DataBuilder(data=data, train=False)

    train_loader = DataLoader(databuilder=train_data_set, batch_size=conf.train.batch_size).loader()
    test_loader = DataLoader(databuilder=test_data_set, batch_size=conf.train.batch_size).loader()

    input_feature = train_loader.dataset.x.shape[1]

    print(f"Start training: {datetime.now()}")
    exe = Executor(cfg=conf.train, input_feature=input_feature)
    train_losses = []
    test_losses = []
    # recon_values= []

    for epoch in range(1, conf.train.epochs + 1):
        train = exe.train(epoch, train_loader)
        train_losses.append(train)
        test = exe.test(epoch, test_loader)
        test_losses.append(test)
        # Comparing real data with reconstructed data
        # recon = exe.reconstructor(test_loader)
        # recon_values.append(recon)

    print(f"Finish training: {datetime.now()}")

    logger.write_file(data=cfg.CFG, file_name="config", cfg=conf)
    logger.write_file(data=train_losses, file_name="train", cfg=conf)
    logger.write_file(data=test_losses, file_name="test", cfg=conf)

    print(f"Saving model: {datetime.now()}")
    exe.save_model(file_name=conf.data.folder)

    # Generate new data
    new_data = exe.generator(test_loader)


if __name__ == "__main__":
    main()
