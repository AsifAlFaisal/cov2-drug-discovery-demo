
from torch_geometric.nn import InstanceNorm, GENConv
from torch.nn import Linear, LSTMCell
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool

class Cov2GEN(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Cov2GEN, self).__init__()
        torch.manual_seed(0)
        self.hidden_dim = hidden_dim
        self.genc1 = GENConv(hidden_dim, hidden_dim)
        self.in1 = InstanceNorm(hidden_dim)
        self.genc2 = GENConv(hidden_dim, hidden_dim)
        self.in2 = InstanceNorm(hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, edge_attr, batch):
        torch.manual_seed(0)
        node_emb = LSTMCell(x.shape[1], self.hidden_dim)
        torch.manual_seed(0)
        edge_emb = LSTMCell(edge_attr.shape[1], self.hidden_dim)
        _, x = node_emb(x)
        _, edge_attr = edge_emb(edge_attr)
        x = F.relu(self.genc1(x, edge_index, edge_attr))
        x = self.in1(x)
        x = F.relu(self.genc2(x, edge_index, edge_attr))
        x = self.in2(x)
        x = global_max_pool(x, batch)        
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.lin(x)
        return torch.sigmoid(x)

