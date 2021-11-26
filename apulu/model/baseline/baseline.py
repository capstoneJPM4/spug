import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report

class baseline(object):
        
    def __init__(self, df, companies, time_range):
        
        class StockDataset(Dataset):
            def __init__(self, X, y, transform=None, target_transform=None):
                self.X = X
                self.y = y
                self.transform = transform
                self.target_transform = target_transform

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return self.X[idx].T, self.y[idx].T

        class model_class(nn.Module):
            def __init__(self, IMPUT_DIM):
                super(model_class, self).__init__()
                self.fc1 = nn.Linear(IMPUT_DIM, 32)
                self.fc2 = nn.Linear(32, 16)
                self.fc3 = nn.Linear(16, 1)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                x = self.fc3(x)
                return torch.sigmoid(x)
        
        self.df = df
        self.df["Year"] = self.df.Date.dt.year
        self.df["Quarter"] = self.df.Date.dt.quarter
        self.df["Month"] = self.df.Date.dt.month
        self.df["Week"] = self.df.Date.dt.isocalendar().week
        self.companies = companies
        self.time_range = time_range
        
        self.train_df = df[df.Date < "2020-10-01"].copy()
        self.val_df = df[df.Date >= "2020-10-01"].copy()

        self.train_X, self.train_y = self._split_data(self.train_df, companies, time_range)
        self.val_X, self.val_y = self._split_data(self.val_df, companies, time_range)

        self.train_ds = StockDataset(self.train_X, self.train_y)
        self.val_ds = StockDataset(self.val_X, self.val_y)

        self.train_dl = DataLoader(self.train_ds, batch_size = 1, shuffle=True)
        self.val_dl = DataLoader(self.val_ds, batch_size = 1, shuffle=True)
        self.IMPUT_DIM = self.train_X.shape[1]
        
        self.model = model_class(self.IMPUT_DIM)
        
        self.EPOCH = 50
        self.LEARN_RATE = 0.0001
        self.OPTIM = torch.optim.Adam(self.model.parameters(), self.LEARN_RATE)
        self.LOSS = nn.BCELoss()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _split_data(self, df, companies, time_range):
        """Split data based on range (Quarter, Month, Week)"""
        
        if time_range not in ("Quarter", "Month", "Week"):
            raise ValueError("Please indicate data range (Quarter, Month, or Week)")

        n = len(companies)
        X = [df.groupby(["Year", time_range]).get_group(x)[companies].to_numpy() \
            for x in df.groupby(["Year", time_range]).groups]
        len_data = max([len(X[i]) for i in range(len(X))])
        X = [np.pad(X[i], ((len_data-X[i].shape[0],0), (0,0)), "mean") for i in range(len(X))]
        curr_mean = np.array([X[i].mean(axis=0) for i in range(len(X))])
        y = (np.diff(curr_mean, axis=0) > 0).astype(int)
        X = X[:-1]

        return np.array(X).astype(np.float32), np.array(y).astype(np.float32)
    
    def _training_loop(self, epochs, optimizer, model, loss_f, train_loader, val_loader, device):
        """Train baseline model"""
        
        for epoch in tqdm(range(epochs), position=0, leave=True):
            loss_train = 0.0
            for x, y in train_loader:
                out = model(x.to(device=device))
                loss = loss_f(out, y.to(device=device).unsqueeze(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss.item()

            if epoch % 10 == 0 or epoch == epochs - 1:
                loss_val = 0.0         
                for x, y in val_loader:
                    out = model(x.to(device=device))
                    loss = loss_f(out, y.to(device=device).unsqueeze(-1))
                    loss_val += loss.item()
                print(
                    epoch, "Training Loss:", loss_val/len(val_loader),
                    "Validation Loss:", loss_train/len(train_loader)
                )
    
    def _val_loop(self, model, val_loader, device):
        
        preds = []
        trues = []
        
        for x, y in val_loader:
            out = model(x.to(device=device)).detach().numpy().reshape(len(self.companies),)
            out = (out > 0.5).astype(int).tolist()
            y = y.detach().numpy().reshape(len(self.companies),).tolist()
            preds += out
            trues += y
        
        return classification_report(trues, preds)
                
    def train(self):
        self._training_loop(self.EPOCH, self.OPTIM, self.model, self.LOSS, self.train_dl, self.val_dl, self.DEVICE)
        
    def validate(self):
        return self._val_loop(self.model, self.val_dl, self.DEVICE)