import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from model import autoencoder
from datetime import datetime
import pandas as pd


class Executor:
    def __init__(self, cfg, input_feature):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device: ", self.device)
        self.model = autoencoder.Autoencoder(input_feature, cfg.H, cfg.H2, latent_dim=cfg.latent_dim,
                                             dropout=cfg.dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.custom_loss = autoencoder.CustomLoss()
        self.cfg = cfg

    def train(self, epoch, loader):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(data)  # forward
            loss = self.custom_loss(recon_batch, data, mu, log_var)  # calling custom loss
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        if epoch % self.cfg.print_iteration == 0:
            print('====> Epoch: {} Time: {} Average training loss: {:.4f}'.format(
                epoch, datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), train_loss / len(loader.dataset)))
        return {"epoch": epoch, "time": str(datetime.now()), "loss": train_loss / len(loader.dataset)}

    def test(self, epoch, loader):
        with torch.no_grad():
            test_loss = 0
            for batch_idx, data in enumerate(loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, log_var = self.model(data)
                loss = self.custom_loss(recon_batch, data, mu, log_var)
                test_loss += loss.item()
            if epoch % self.cfg.print_iteration == 0:
                print('====> Epoch: {} Time: {} Average test loss: {:.4f}'.format(
                    epoch, datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), test_loss / len(loader.dataset)))
        return {"epoch": epoch, "time": str(datetime.now()), "loss": test_loss / len(loader.dataset)}

    def save_model(self):
        torch.save(self.model.state_dict(), "saved_models/" + f"{datetime.now()}-model.pth")

    def reconstructor(self, loader):
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)

        scaler = loader.dataset.standardizer
        recon_row = scaler.inverse_transform(recon_batch[0].cpu().numpy().reshape(1, -1))
        real_row = scaler.inverse_transform(loader.dataset.x[0].cpu().numpy().reshape(1, -1))
        # print("recon_row", recon_row)
        # print("real_row", real_row)
        # df = pd.DataFrame(np.stack((recon_row, real_row)), columns=["x","y","z","vx","vy","vz"])
        # print(df)

    def generator(self, loader):
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
        sigma = torch.exp(logvar / 2)
        no_samples = 20
        q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
        z = q.rsample(sample_shape=torch.Size([no_samples]))
        print(z.shape)
        print(z[:5])
        with torch.no_grad():
            pred = self.model.decode(z).cpu().numpy()
        scaler = loader.dataset.standardizer
        fake_data = scaler.inverse_transform(pred)
        df_fake = pd.DataFrame(fake_data, columns=["x","y","z","vx","vy","vz","time"])
        df_fake['time'] = np.round(df_fake['time']).astype(int)
        df_fake['time'] = np.where(df_fake['time'] < 1, 1, df_fake['time'])
        print(df_fake.head(10))
