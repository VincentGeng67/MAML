import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv

from HGPlayers import GCN, HGPSLPool
import torch.nn as nn
from    copy import deepcopy


class HGPModel(torch.nn.Module):
    def __init__(self, args):
        super(HGPModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb
        
        
        
        
        
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)
        
        
        

    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)
        

        return x

    def setpara(self,parameter,init=False):
        if parameter is None:
            parameter=parameter
        elif init:
#         else:
            self.conv1.weight=parameter[0]
            self.conv1.bias=parameter[1]
            self.conv2.weight=parameter[2]
            self.conv2.bias=parameter[3]
            self.conv3.weight=parameter[4]
            self.conv3.bias=parameter[5]
            self.pool1.att=parameter[6]
            self.pool2.att=parameter[7]
            self.lin1.weight=parameter[8]
            self.lin1.bias=parameter[9]
            self.lin2.weight=parameter[10]
            self.lin2.bias=parameter[11]
            self.lin3.weight=parameter[12] 
            self.lin3.bias=parameter[13] 
        else:
            
            self.conv1.weight.data=parameter[0]
            self.conv1.bias.data=parameter[1]
            self.conv2.weight.data=parameter[2]
            self.conv2.bias.data=parameter[3]
            self.conv3.weight.data=parameter[4]
            self.conv3.bias.data=parameter[5]
            self.pool1.att.data=parameter[6]
            self.pool2.att.data=parameter[7]
            self.lin1.weight.data=parameter[8]
            self.lin1.bias.data=parameter[9]
            self.lin2.weight.data=parameter[10]
            self.lin2.bias.data=parameter[11]
            self.lin3.weight.data=parameter[12] 
            self.lin3.bias.data=parameter[13]  
        
        
        
    
    def weight_bias(self):
        wblist=nn.ParameterList()
        wblist.append(self.conv1.weight)
        wblist.append(self.conv1.bias)
        wblist.append(self.conv2.weight)
        wblist.append(self.conv2.bias)
        wblist.append(self.conv3.weight)
        wblist.append(self.conv3.bias)
        wblist.append(self.pool1.att)
        wblist.append(self.pool2.att)
        wblist.append(self.lin1.weight)
        wblist.append(self.lin1.bias)
        wblist.append(self.lin2.weight)
        wblist.append(self.lin2.bias)
        wblist.append(self.lin3.weight)
        wblist.append(self.lin3.bias)
        
        
        return wblist