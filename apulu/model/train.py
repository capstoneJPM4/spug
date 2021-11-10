import collections
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from tqdm import tqdm
from model import GNN

class Train():
    def __init__(self, dataset, feature_size, epochs = 10, learning_rate = 0.01):
        self.data = dataset
        self.fs = feature_size
        
        self.model = GNN(self.data, self.fs)
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate, weight_decay = decay)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def train_network(self, epochs = 10, learning_rate = 0.01):
        data = self.data
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            output = self.model(data, data.edge_index, data.edge_attr)
            loss = self.loss_func(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            train_losses += [loss]
            
            train_correct = torch.argmax(output[data.train_mask], dim=1) == data.y[data.train_mask]
            train_acc = int(train_correct.sum()) / int(data.train_mask.sum())
            train_accs += [train_acc]

            test_correct = torch.argmax(output[data.test_mask], dim=1) == data.y[data.test_mask]
            test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
            test_accs += [test_acc]

            print(f"Epoch {epoch + 1}/{epochs}, Train_Loss: {loss:.4f}, Train_Accuracy: {train_acc:.4f}, Test_Accuracy: {test_acc:.4f}")

        plt.plot(train_losses)
        plt.show()

        plt.plot(train_accs)
        plt.plot(test_accs) 
        plt.show()
