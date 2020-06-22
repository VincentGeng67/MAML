import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from gin_layer import GINLayer, ApplyNodeFunc, MLP

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from gin_layer import GINLayer, ApplyNodeFunc, MLP

class GINNet(nn.Module):
    
    def __init__(self, net_params,parameter=None,init=False):
        super().__init__()
        in_dim = net_params["in_dim"]
        hidden_dim = net_params["hidden_dim"]
        n_classes = net_params["n_classes"]
        dropout =net_params["dropout"]
        self.n_layers = net_params["L"]
        n_mlp_layers = net_params["n_mlp_GIN"]            # GIN
        learn_eps = net_params['learn_eps_GIN']              # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN'] # GIN
        readout = net_params['readout']                      # this is graph_pooling_type
        graph_norm = net_params['graph_norm']      
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']   
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        for layer in range(self.n_layers):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))

        # Linear function for graph poolings (readout) of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(self.n_layers+1):
            self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))
        
        if parameter is None:
            parameter=parameter
        elif init:
#         else:
            self.ginlayers[0].apply_func.mlp.linears[0].weight=parameter[0]
            self.ginlayers[0].apply_func.mlp.linears[0].bias=parameter[1]
            self.ginlayers[0].apply_func.mlp.linears[1].weight=parameter[2]
            self.ginlayers[0].apply_func.mlp.linears[1].bias=parameter[3]
            self.ginlayers[1].apply_func.mlp.linears[0].weight=parameter[4]
            self.ginlayers[1].apply_func.mlp.linears[0].bias=parameter[5]
            self.ginlayers[1].apply_func.mlp.linears[1].weight=parameter[6]
            self.ginlayers[1].apply_func.mlp.linears[1].bias=parameter[7]
            self.ginlayers[2].apply_func.mlp.linears[0].weight=parameter[8]
            self.ginlayers[2].apply_func.mlp.linears[0].bias=parameter[9]
            self.ginlayers[2].apply_func.mlp.linears[1].weight=parameter[10]
            self.ginlayers[2].apply_func.mlp.linears[1].bias=parameter[11]
            self.ginlayers[3].apply_func.mlp.linears[0].weight=parameter[12]
            self.ginlayers[3].apply_func.mlp.linears[0].bias=parameter[13]
            self.ginlayers[3].apply_func.mlp.linears[1].weight=parameter[14]
            self.ginlayers[3].apply_func.mlp.linears[1].bias=parameter[15]
            self.embedding_h.weight=parameter[16] 
            self.embedding_h.bias=parameter[17] 
            self.linears_prediction[0].weight=parameter[18] 
            self.linears_prediction[0].bias=parameter[19]
            self.linears_prediction[1].weight=parameter[20] 
            self.linears_prediction[1].bias=parameter[21]
            self.linears_prediction[2].weight=parameter[22] 
            self.linears_prediction[2].bias=parameter[23]
        else:
            
            self.ginlayers[0].apply_func.mlp.linears[0].weight.data=parameter[0]
            self.ginlayers[0].apply_func.mlp.linears[0].bias.data=parameter[1]
            self.ginlayers[0].apply_func.mlp.linears[1].weight.data=parameter[2]
            self.ginlayers[0].apply_func.mlp.linears[1].bias.data=parameter[3]
            self.ginlayers[1].apply_func.mlp.linears[0].weight.data=parameter[4]
            self.ginlayers[1].apply_func.mlp.linears[0].bias.data=parameter[5]
            self.ginlayers[1].apply_func.mlp.linears[1].weight.data=parameter[6]
            self.ginlayers[1].apply_func.mlp.linears[1].bias.data=parameter[7]
            self.ginlayers[2].apply_func.mlp.linears[0].weight.data=parameter[8]
            self.ginlayers[2].apply_func.mlp.linears[0].bias.data=parameter[9]
            self.ginlayers[2].apply_func.mlp.linears[1].weight.data=parameter[10]
            self.ginlayers[2].apply_func.mlp.linears[1].bias.data=parameter[11]
            self.ginlayers[3].apply_func.mlp.linears[0].weight.data=parameter[12]
            self.ginlayers[3].apply_func.mlp.linears[0].bias.data=parameter[13]
            self.ginlayers[3].apply_func.mlp.linears[1].weight.data=parameter[14]
            self.ginlayers[3].apply_func.mlp.linears[1].bias.data=parameter[15]
            self.embedding_h.weight.data=parameter[16] 
            self.embedding_h.bias.data=parameter[17] 
            self.linears_prediction[0].weight.data=parameter[18] 
            self.linears_prediction[0].bias.data=parameter[19]
            self.linears_prediction[1].weight.data=parameter[20] 
            self.linears_prediction[1].bias.data=parameter[21]
            self.linears_prediction[2].weight.data=parameter[22] 
            self.linears_prediction[2].bias.data=parameter[23]
        
#         self.ginlayers=parameter[0]
#         self.embedding_h=parameter[1]
#         self.linears_prediction=parameter[2]

    
    
        if readout == 'sum':
            self.pool = SumPooling()
        elif readout == 'mean':
            self.pool = AvgPooling()
        elif readout == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError
        
#         print('old0',self.n_layers)


            
        
    def forward(self, g, h, e, snorm_n, snorm_e):
        
#         self.embedding_h.requires_grad = True
#         print(self.embedding_h.grad_fn)
#         print('inputh',h)
        h = self.embedding_h(h)
#         h.requires_grad = True
#         print('self.embedding',self.embedding_h)
#         print('self.embeddingw',self.embedding_h.weight)
#         print('self.embeddingb',self.embedding_h.bias)
#         print('omg',h.requires_grad)
#         print('outputh',h )
#         h.requires_grad = True
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]
        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)

        score_over_layer = 0
        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
#             pooled_h.requires_grad = True
            score_over_layer += self.linears_prediction[i](pooled_h)
        return score_over_layer
    

        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        
        
        return loss
    
    def setpara(self,parameter,init=False):
        if parameter is None:
            parameter=parameter
        elif init:
#         else:
            self.ginlayers[0].apply_func.mlp.linears[0].weight=parameter[0]
            self.ginlayers[0].apply_func.mlp.linears[0].bias=parameter[1]
            self.ginlayers[0].apply_func.mlp.linears[1].weight=parameter[2]
            self.ginlayers[0].apply_func.mlp.linears[1].bias=parameter[3]
            self.ginlayers[1].apply_func.mlp.linears[0].weight=parameter[4]
            self.ginlayers[1].apply_func.mlp.linears[0].bias=parameter[5]
            self.ginlayers[1].apply_func.mlp.linears[1].weight=parameter[6]
            self.ginlayers[1].apply_func.mlp.linears[1].bias=parameter[7]
            self.ginlayers[2].apply_func.mlp.linears[0].weight=parameter[8]
            self.ginlayers[2].apply_func.mlp.linears[0].bias=parameter[9]
            self.ginlayers[2].apply_func.mlp.linears[1].weight=parameter[10]
            self.ginlayers[2].apply_func.mlp.linears[1].bias=parameter[11]
            self.ginlayers[3].apply_func.mlp.linears[0].weight=parameter[12]
            self.ginlayers[3].apply_func.mlp.linears[0].bias=parameter[13]
            self.ginlayers[3].apply_func.mlp.linears[1].weight=parameter[14]
            self.ginlayers[3].apply_func.mlp.linears[1].bias=parameter[15]
            self.embedding_h.weight=parameter[16] 
            self.embedding_h.bias=parameter[17] 
            self.linears_prediction[0].weight=parameter[18] 
            self.linears_prediction[0].bias=parameter[19]
            self.linears_prediction[1].weight=parameter[20] 
            self.linears_prediction[1].bias=parameter[21]
            self.linears_prediction[2].weight=parameter[22] 
            self.linears_prediction[2].bias=parameter[23]
        else:
            
            self.ginlayers[0].apply_func.mlp.linears[0].weight.data=parameter[0]
            self.ginlayers[0].apply_func.mlp.linears[0].bias.data=parameter[1]
            self.ginlayers[0].apply_func.mlp.linears[1].weight.data=parameter[2]
            self.ginlayers[0].apply_func.mlp.linears[1].bias.data=parameter[3]
            self.ginlayers[1].apply_func.mlp.linears[0].weight.data=parameter[4]
            self.ginlayers[1].apply_func.mlp.linears[0].bias.data=parameter[5]
            self.ginlayers[1].apply_func.mlp.linears[1].weight.data=parameter[6]
            self.ginlayers[1].apply_func.mlp.linears[1].bias.data=parameter[7]
            self.ginlayers[2].apply_func.mlp.linears[0].weight.data=parameter[8]
            self.ginlayers[2].apply_func.mlp.linears[0].bias.data=parameter[9]
            self.ginlayers[2].apply_func.mlp.linears[1].weight.data=parameter[10]
            self.ginlayers[2].apply_func.mlp.linears[1].bias.data=parameter[11]
            self.ginlayers[3].apply_func.mlp.linears[0].weight.data=parameter[12]
            self.ginlayers[3].apply_func.mlp.linears[0].bias.data=parameter[13]
            self.ginlayers[3].apply_func.mlp.linears[1].weight.data=parameter[14]
            self.ginlayers[3].apply_func.mlp.linears[1].bias.data=parameter[15]
            self.embedding_h.weight.data=parameter[16] 
            self.embedding_h.bias.data=parameter[17] 
            self.linears_prediction[0].weight.data=parameter[18] 
            self.linears_prediction[0].bias.data=parameter[19]
            self.linears_prediction[1].weight.data=parameter[20] 
            self.linears_prediction[1].bias.data=parameter[21]
            self.linears_prediction[2].weight.data=parameter[22] 
            self.linears_prediction[2].bias.data=parameter[23]
        
        
        
    
    def weight_bias(self):
        wblist=nn.ParameterList()
        wblist.append(self.ginlayers[0].apply_func.mlp.linears[0].weight)
        wblist.append(self.ginlayers[0].apply_func.mlp.linears[0].bias)
        wblist.append(self.ginlayers[0].apply_func.mlp.linears[1].weight)
        wblist.append(self.ginlayers[0].apply_func.mlp.linears[1].bias)
        wblist.append(self.ginlayers[1].apply_func.mlp.linears[0].weight)
        wblist.append(self.ginlayers[1].apply_func.mlp.linears[0].bias)
        wblist.append(self.ginlayers[1].apply_func.mlp.linears[1].weight)
        wblist.append(self.ginlayers[1].apply_func.mlp.linears[1].bias)
        wblist.append(self.ginlayers[2].apply_func.mlp.linears[0].weight)
        wblist.append(self.ginlayers[2].apply_func.mlp.linears[0].bias)
        wblist.append(self.ginlayers[2].apply_func.mlp.linears[1].weight)
        wblist.append(self.ginlayers[2].apply_func.mlp.linears[1].bias)
        wblist.append(self.ginlayers[3].apply_func.mlp.linears[0].weight)
        wblist.append(self.ginlayers[3].apply_func.mlp.linears[0].bias)
        wblist.append(self.ginlayers[3].apply_func.mlp.linears[1].weight)
        wblist.append(self.ginlayers[3].apply_func.mlp.linears[1].bias)
        wblist.append(self.embedding_h.weight)
        wblist.append(self.embedding_h.bias) 
        wblist.append(self.linears_prediction[0].weight) 
        wblist.append(self.linears_prediction[0].bias)
        wblist.append(self.linears_prediction[1].weight) 
        wblist.append(self.linears_prediction[1].bias)
        wblist.append(self.linears_prediction[2].weight)
        wblist.append(self.linears_prediction[2].bias)
        
        return wblist