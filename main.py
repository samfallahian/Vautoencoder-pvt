from configs import config as cfg
from utils.config import Config
from dataloader.databuilder import DataBuilder
from executor.executor import Executor
from datetime import datetime
import json
import os.path

def main():
    conf = Config.from_json(cfg.CFG)
    path = conf.data.path + "test" #3p6

    data = DataBuilder(path)

    train_loader = data.train_data(batch_size=conf.train.batch_size)
    test_loader = data.test_data(batch_size=conf.train.batch_size)

    input_feature = train_loader.dataset.x.shape[1]

    exe = Executor(cfg=conf.train, input_feature=input_feature)
    train_losses=[]
    test_losses=[]

    for epoch in range(1, conf.train.epochs + 1):
        train = exe.train(epoch, train_loader)
        train_losses.append(train)
        test = exe.test(epoch, test_loader)
        test_losses.append(test)

    with open(os.path.join("logs", f"{datetime.now()}-train.json"), "w") as f:
        json.dump(train_losses, f, indent=4)

    with open(os.path.join("logs", f"{datetime.now()}-test.json"), "w") as f:
        json.dump(train_losses, f, indent=4)


if __name__ == "__main__":
    main()

    # H = 50
    # H2 = 12
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = autoencoder.Autoencoder(D_in, H, H2).to(device)
    # custom_loss = autoencoder.CustomLoss()
# conf = Config.from_json(cfg.CFG)
# path = conf.data.path + "test"
#
# data = DataBuilder(path)
#
# trainloader = data.train_data(batch_size=conf.train.batch_size)
# testloader = data.test_data(batch_size=conf.train.batch_size)

# print(type(a.train_data(batch_size=conf.train.batch_size).dataset.x))
# print(a.test_data(batch_size=conf.train.batch_size).dataset.len)
# print(a.test_data(batch_size=conf.train.batch_size).dataset.x)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# D_in = trainloader.dataset.x.shape[1]
#
# H = 50
# H2 = 12
# model = autoencoder.Autoencoder(D_in, H, H2).to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# loss_mse = autoencoder.CustomLoss()


# epochs = 1500
# log_interval = 50
# val_losses = []
# train_losses = []
# test_losses = []


# def train(epoch):
#     model.train()
#     train_loss = 0
#     for batch_idx, data in enumerate(trainloader):
#         data = data.to(device)
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data) # forward
#         loss = loss_mse(recon_batch, data, mu, logvar) # calling custom loss
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#     if epoch % 200 == 0:
#         print('====> Epoch: {} Average training loss: {:.4f}'.format(
#             epoch, train_loss / len(trainloader.dataset)))
#         train_losses.append(train_loss / len(trainloader.dataset))


# def test(epoch):
#     with torch.no_grad():
#         test_loss = 0
#         for batch_idx, data in enumerate(testloader):
#             data = data.to(device)
#             optimizer.zero_grad()
#             recon_batch, mu, logvar = model(data)
#             loss = loss_mse(recon_batch, data, mu, logvar)
#             test_loss += loss.item()
#             if epoch % 200 == 0:
#                 print('====> Epoch: {} Average test loss: {:.4f}'.format(
#                     epoch, test_loss / len(testloader.dataset)))
#             test_losses.append(test_loss / len(testloader.dataset))
