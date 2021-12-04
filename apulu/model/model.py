import torch
from torch_geometric_temporal import GConvGRU
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_dims, dropout=0.3):
        super(GCN, self).__init__()
        self.conv = GCNConv(in_channels=input_size, out_channels=hidden_dims)
        self.linear = torch.nn.Linear(in_features=hidden_dims, out_features=2)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        h = self.conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
        h = self.dropout(h)
        h = torch.relu(h)
        h = self.linear(h)
        h = torch.sigmoid(h)
        return h


class RecurrentGCN(torch.nn.Module):
    def __init__(self, input_size, hidden_dims, dropout=0.3):
        super(RecurrentGCN, self).__init__()
        self.conv_gru = GConvGRU(input_size, hidden_dims, 3)
        self.linear = torch.nn.Linear(hidden_dims, 2)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        h = self.conv_gru(x, edge_index, edge_weight)
        h = self.dropout(h)
        h = torch.tanh(h)
        h = self.linear(h)
        h = torch.sigmoid(h)
        return h
