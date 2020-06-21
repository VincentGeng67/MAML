import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from SAGlayers import SAGPool
import torch.nn as nn
from HGPlayers import GCN, HGPSLPool



class SAGNet(torch.nn.Module):
    def __init__(self,args):
        super(SAGNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        print('tq333333')
        self.t1=HGPSLPool(self.nhid, self.pooling_ratio, True, True, True, 1.0)
        self.t2=GCN(self.nhid, self.nhid)
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

    def forward(self, x, edge_index, batch):


        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
    
    def setpara(self,parameter,init=False):
        if parameter is None:
            parameter=parameter
        elif init:
#         else:
            self.conv1.weight=parameter[0]
            self.conv1.bias=parameter[1]
            self.pool1.score_layer.weight=parameter[2]
            self.pool1.score_layer.bias=parameter[3]
            self.conv2.weight=parameter[4]
            self.conv2.bias=parameter[5]
            self.pool2.score_layer.weight=parameter[6]
            self.pool2.score_layer.bias=parameter[7]
            self.conv3.weight=parameter[8]
            self.conv3.bias=parameter[9]
            self.pool3.score_layer.weight=parameter[10]
            self.pool3.score_layer.bias=parameter[11]
            self.lin1.weight=parameter[12]
            self.lin1.bias=parameter[13]
            self.lin2.weight=parameter[14]
            self.lin2.bias=parameter[15]
            self.lin3.weight=parameter[16] 
            self.lin3.bias=parameter[17] 
        else:
            
            self.conv1.weight.data=parameter[0]
            self.conv1.bias.data=parameter[1]
            self.pool1.score_layer.weight.data=parameter[2]
            self.pool1.score_layer.bias.data=parameter[3]
            self.conv2.weight.data=parameter[4]
            self.conv2.bias.data=parameter[5]
            self.pool2.score_layer.weight.data=parameter[6]
            self.pool2.score_layer.bias.data=parameter[7]
            self.conv3.weight.data=parameter[8]
            self.conv3.bias.data=parameter[9]
            self.pool3.score_layer.weight.data=parameter[10]
            self.pool3.score_layer.bias.data=parameter[11]
            self.lin1.weight.data=parameter[12]
            self.lin1.bias.data=parameter[13]
            self.lin2.weight.data=parameter[14]
            self.lin2.bias.data=parameter[15]
            self.lin3.weight.data=parameter[16] 
            self.lin3.bias.data=parameter[17] 
        
        
        
    
    def weight_bias(self):
        wblist=nn.ParameterList()
        wblist.append(self.conv1.weight)
        wblist.append(self.conv1.bias)
        wblist.append(self.pool1.score_layer.weight)
        wblist.append(self.pool1.score_layer.bias)
        wblist.append(self.conv2.weight)
        wblist.append(self.conv2.bias)
        wblist.append(self.pool2.score_layer.weight)
        wblist.append(self.pool2.score_layer.bias)
        wblist.append(self.conv3.weight)
        wblist.append(self.conv3.bias)
        wblist.append(self.pool3.score_layer.weight)
        wblist.append(self.pool3.score_layer.bias)
        wblist.append(self.lin1.weight)
        wblist.append(self.lin1.bias)
        wblist.append(self.lin2.weight)
        wblist.append(self.lin2.bias)
        wblist.append(self.lin3.weight)
        wblist.append(self.lin3.bias)
        
        
        return wblist