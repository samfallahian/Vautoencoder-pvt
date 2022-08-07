
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from model import autoencoder
from datetime import datetime


class Executor:
    def __init__(self, cfg, input_feature):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = autoencoder.Autoencoder(input_feature, cfg.H, cfg.H2).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.custom_loss= autoencoder.CustomLoss()

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
        if epoch % 100 == 0:
            print('====> Epoch: {} Time: {} Average training loss: {:.4f}'.format(
                epoch, datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), train_loss / len(loader.dataset)))
        return {"epoch": epoch, "time": str(datetime.now()) , "loss": train_loss / len(loader.dataset)}

    def test(self, epoch, loader):
        with torch.no_grad():
            test_loss = 0
            for batch_idx, data in enumerate(loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, log_var = self.model(data)
                loss = self.custom_loss(recon_batch, data, mu, log_var)
                test_loss += loss.item()
            if epoch % 100 == 0:
                print('====> Epoch: {} Time: {} Average test loss: {:.4f}'.format(
                        epoch, datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), test_loss / len(loader.dataset)))
        return {"epoch": epoch, "time": str(datetime.now()) ,"loss": test_loss / len(loader.dataset)}

    def save_model(self):
        torch.save(self.model.state_dict(), "saved_models/"+f"{datetime.now()}-model.pth")



