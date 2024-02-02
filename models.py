import torch
from torch import nn, optim, autograd as ag
import numpy as np
import copy
import argparse
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

class CosMAML():
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, X_train, y_train, n_epochs, 
                          batch_size, optimiser, loss_fn, loss_rec=True):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        loss_record = []
        for _ in range(n_epochs):
            for X_batch, y_batch in train_loader:
                optimiser.zero_grad()
                y_pred = self.model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                if loss_rec:
                    loss_record.append(loss.item())
                loss.backward()
                optimiser.step()
        self.model = self.model
        if loss_rec:
            return loss_record   

    def predict(self, X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_pred = self.model(X_test)
        return y_pred.detach().cpu().numpy()
    
    # def maml_train():