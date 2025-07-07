import torch.nn as nn
import torch.nn.functional as F
from ekan import KAN, KANLinear

from torch_geometric.nn import GINEConv, GCNConv, GINConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn.conv import EdgeConv


def make_mlp(num_features, hidden_dim, out_dim, hidden_layers, batch_norm=True):
    if hidden_layers>=2:
        if batch_norm:
            list_hidden = [nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim))]
        else:
            list_hidden = [nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU())]
        for _ in range(hidden_layers-2):
            if batch_norm:
                list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)))
            else:
                list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, out_dim, nn.ReLU())))
    else:
        list_hidden = [nn.Sequential(nn.Linear(num_features, out_dim), nn.ReLU())]
    MLP = nn.Sequential(*list_hidden)
    return(MLP)
def make_kan(num_features, hidden_dim, out_dim, hidden_layers, grid_size, spline_order):
    sizes = [num_features] + [hidden_dim]*(hidden_layers-1) + [out_dim]
    return(KAN(layers_hidden=sizes, grid_size=grid_size, spline_order=spline_order))

class KAGIN_cls(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, num_classes, hidden_layers, grid_size, spline_order, dropout):
        super(KAGIN_cls, self).__init__()
        self.n_layers = gnn_layers
        lst = list()
        lst.append(GINConv(make_kan(num_features, hidden_dim, hidden_dim, hidden_layers, grid_size, spline_order)))
        for i in range(gnn_layers-1):
            lst.append(GINConv(make_kan(hidden_dim, hidden_dim, hidden_dim, hidden_layers, grid_size, spline_order)))
        self.conv = nn.ModuleList(lst)
        lst = list()
        for i in range(gnn_layers):
            lst.append(nn.BatchNorm1d(hidden_dim))
        self.bn = nn.ModuleList(lst)
        self.kan = make_kan(hidden_dim, hidden_dim, num_classes, hidden_layers, grid_size, spline_order)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, edge_attr=None, edge_weight=None, node_level: bool = True):

        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = self.bn[i](x)
            x = self.dropout(x)
        if node_level:
            x = self.kan(x)
            return F.log_softmax(x, dim=1)
        x = global_add_pool(x, batch)
        x = self.kan(x)
        return F.log_softmax(x, dim=1)

class KANLayer(KANLinear):
    def __init__(self, input_dim, output_dim, grid_size=4, spline_order=3):
        super(KANLayer, self).__init__(in_features=input_dim, out_features=output_dim, grid_size=grid_size,
                                       spline_order=spline_order)


class KAGCN_Layer(GCNConv):
    def __init__(self, in_feat: int,
                 out_feat: int,
                 grid_size: int = 4,
                 spline_order: int = 3):
        super(KAGCN_Layer, self).__init__(in_feat, out_feat)
        self.lin = KANLayer(in_feat, out_feat, grid_size, spline_order)

class KAGCN_cls(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, num_classes, grid_size, spline_order, dropout):
        super(KAGCN_cls, self).__init__()
        self.n_layers = gnn_layers
        lst = list()
        lst.append(KAGCN_Layer(num_features, hidden_dim, grid_size, spline_order))
        for _ in range(gnn_layers-1):
            lst.append(KAGCN_Layer(hidden_dim, hidden_dim, grid_size, spline_order))
        self.conv = nn.ModuleList(lst)
        self.readout = make_kan(hidden_dim, hidden_dim, num_classes, 1, grid_size, spline_order)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, batch, edge_attr=None, edge_weight=None, node_level: bool = True):
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = F.silu(x)
            x = self.dropout(x)
        if node_level:
            x = self.readout(x)
            return F.log_softmax(x, dim=1)
        x = global_mean_pool(x, batch)
        x = self.readout(x)
        return F.log_softmax(x, dim=1)

# function creating KAN-based EdgeCNN layer
def get_KAEdgeCNN_layer(in_channels: int, out_channels: int, grid_size:int, spline_order:int,
                  **kwargs):
    if grid_size is None:
        raise ValueError("grid size must be provided")
    if spline_order is None:
        raise ValueError("spline order must be provided")
    kan = make_kan(
        num_features=2 * in_channels,
        hidden_dim=out_channels,
        out_dim=out_channels,
        hidden_layers=1,
        grid_size=grid_size,
        spline_order=spline_order
    )
    return EdgeConv(kan, **kwargs)

class KAEdge_cls(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, num_classes, hidden_layers, grid_size, spline_order, dropout):
        super(KAEdge_cls, self).__init__()
        self.n_layers = gnn_layers
        lst = list()
        lst.append(get_KAEdgeCNN_layer(in_channels=num_features, out_channels=hidden_dim, grid_size=grid_size, spline_order=spline_order))
        for i in range(gnn_layers-1):
            lst.append(get_KAEdgeCNN_layer(in_channels=hidden_dim, out_channels=hidden_dim, grid_size=grid_size, spline_order=spline_order))
        self.conv = nn.ModuleList(lst)
        lst = list()
        for i in range(gnn_layers):
            lst.append(nn.BatchNorm1d(hidden_dim))
        self.bn = nn.ModuleList(lst)
        self.kan = make_kan(hidden_dim, hidden_dim, num_classes, hidden_layers, grid_size, spline_order)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, edge_attr=None, edge_weight=None, node_level: bool = True):
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = self.bn[i](x)
            x = F.silu(x)
            x = self.dropout(x)
        if node_level:
            x = self.kan(x)
            return F.log_softmax(x, dim=1)
        x = global_add_pool(x, batch)
        x = self.kan(x)
        return F.log_softmax(x, dim=1)
