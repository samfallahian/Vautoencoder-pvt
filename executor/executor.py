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

    def save_model(self, file_name):
        # Saving Model Weights
        torch.save(self.model.state_dict(), "saved_models/" + f"{datetime.now()}-model_state-{file_name}.pth")
        # Saving Models with Shapes
        torch.save(self.model, "saved_models/" + f"{datetime.now()}-model-{file_name}.pth")

    def load_model(self, file_name):
        load_model = self.model
        load_model.load_state_dict(torch.load("saved_models/" + f"{file_name}.pth"))
        print(load_model.eval())

    def reconstructor(self, loader):
        scaler = loader.dataset.standardizer
        df_data = []
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                recon_row = scaler.inverse_transform(recon_batch[0].cpu().numpy().reshape(1, -1))
                real_row = scaler.inverse_transform(loader.dataset.x[0].cpu().numpy().reshape(1, -1))
                df_data.append(real_row)
                df_data.append(recon_row)
        return df_data

    def generator(self, loader):
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
        sigma = torch.exp(logvar / 2)
        no_samples = 5000
        q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
        z = q.rsample(sample_shape=torch.Size([no_samples]))
        with torch.no_grad():
            pred = self.model.decode(z).cpu().numpy()
        scaler = loader.dataset.standardizer
        fake_data = scaler.inverse_transform(pred)
        df_fake = self.create_data_frame(fake_data)
        print('====> Generated data:')
        print(df_fake.head())
        return df_fake

    @staticmethod
    def create_data_frame(arr):
        # df = pd.DataFrame(arr, columns=["x", "y", "z", "vx", "vy", "vz", "time"])
        # df = pd.DataFrame(arr, columns=["vx", "vy", "vz", "time","loc_point"])
        # df = pd.DataFrame(arr, columns=["x", "y", "z", "vx", "vy", "vz", "time", "distance"])
        try:
            df = pd.DataFrame(arr,
                              columns=["vx", "vy", "vz", "time","transformed_x", "transformed_y", "transformed_z", "distance"])
            df['time'] = np.round(df['time']).astype(int)
            df['time'] = np.where(df['time'] < 1, 1, df['time'])
            df['transformed_x'] = np.round(df['transformed_x']).astype(int)
            df['transformed_y'] = np.round(df['transformed_y']).astype(int)
            df['transformed_z'] = np.round(df['transformed_z']).astype(int)
            df['distance'] = np.round(df['distance']).astype(int)
            df.to_csv("logs/" + f"{datetime.now()}-generated-data.csv", index=False)
            return df
        except:
            print("An exception occurred")
            return pd.DataFrame([1,1,1,1,1,1,1],
                              columns=["vx", "vy", "vz", "transformed_x", "transformed_y", "transformed_z", "distance"])


