import torch
import torch.nn.functional as F 
from torch.nn import Linear
from torch_geometric.nn import TransformerConv, GCNConv

torch.manual_seed(42)

class GNN(torch.nn.Module):
    def __init__(self, dataset, feature_size):
        super(GNN, self).__init__()
        torch.manual_seed(10)

        input_size = dataset.num_features
        output_size = dataset.num_classes
        self.conv1 = GCNConv(in_channels=input_size, out_channels=feature_size)
        self.conv2 = GCNConv(in_channels=feature_size, out_channels=feature_size)
        self.d1 = torch.nn.Linear(feature_size, output_size)

    def forward(self, x, edge_index):
        # First Message Passing Layer
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)

        # Output layer 
        x = F.softmax(self.out(x), dim=1)
        return x

