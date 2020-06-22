


import  torch
from    torch import nn

import  numpy as np
import dgl

from tqdm import tqdm_notebook as tqdm
import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from gin_net import GINNet

import torch.optim as optim
from torch.utils.data import DataLoader
from gnnf import gpu_setup,  train_epoch, evaluate_network, gnn_model, init_parameters
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from gin_layer import GINLayer, ApplyNodeFunc, MLP





class Learner(nn.Module):
    """
    """

    def __init__(self,way):
        """
        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()
        with open("superpixels_graph_classification_GIN_MNIST.json", "r") as read_file:
            self.config= json.load(read_file)

        MODEL_NAME='GIN'
        config=self.config
        params = config['params']
        self.params=params
        # this dict contains all tensors needed to be optimized
#         self.vars = nn.ParameterList()
        # running_mean and running_var
#         self.vars_bn = nn.ParameterList()
        net_params = config['net_params']
        device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
        net_params['device'] = device
        net_params['gpu_id'] = config['gpu']['id']
        net_params['batch_size'] = params['batch_size']
        net_params['in_dim'] = 3
        net_params['in_dim_edge'] = 1
        net_params['n_classes'] = 10
#         net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
        self.net_params=net_params
    
        
        

#         w = nn.Parameter(torch.FloatTensor(self.net_params["L"]))
    
#         print('w',w)
#         print('l',self.net_params["L"])
#         self.vars_bn = nn.ParameterList()

        in_dim = net_params["in_dim"]
        hidden_dim = net_params["hidden_dim"]
        n_classes = net_params["n_classes"]
        dropout =net_params["dropout"]
        n_layers = net_params["L"]
        n_mlp_layers = net_params["n_mlp_GIN"]            # GIN
        learn_eps = net_params['learn_eps_GIN']              # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN'] # GIN
        readout = net_params['readout']                      # this is graph_pooling_type
        graph_norm = net_params['graph_norm']      
        batch_norm = net_params['batch_norm']
        residual = net_params['residual'] 
        random.seed(params['seed'])
        np.random.seed(params['seed'])
        torch.manual_seed(params['seed'])
        device=net_params['device']
        if device == 'cuda':
            torch.cuda.manual_seed(params['seed'])
        
        self.vars =[]
        ginlayers = torch.nn.ModuleList()
        
        embedding_h = nn.Linear(in_dim, hidden_dim)
        
        for layer in range(n_layers):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            
            ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))

        # Linear function for graph poolings (readout) of output of each layer
        # which maps the output of different layers into a prediction score
        linears_prediction = torch.nn.ModuleList()

        for layer in range(n_layers+1):
            linears_prediction.append(nn.Linear(hidden_dim, n_classes))
            
        self.vars.append(ginlayers)
        self.vars.append(embedding_h)
        self.vars.append(linears_prediction)
        
        self.model = gnn_model(MODEL_NAME, self.net_params)
        self.model = self.model.to(device)
        self.vars_bn=list(self.model.parameters())
        
        self.varstest = nn.ParameterList()
        self.modelbn=nn.ParameterList()
        self.varstest.append(self.model.ginlayers[0].apply_func.mlp.linears[0].weight)
        self.varstest.append(self.model.ginlayers[0].apply_func.mlp.linears[0].bias)
        self.varstest.append(self.model.ginlayers[0].apply_func.mlp.linears[1].weight)
        self.varstest.append(self.model.ginlayers[0].apply_func.mlp.linears[1].bias)
        self.varstest.append(self.model.ginlayers[1].apply_func.mlp.linears[0].weight)
        self.varstest.append(self.model.ginlayers[1].apply_func.mlp.linears[0].bias)
        self.varstest.append(self.model.ginlayers[1].apply_func.mlp.linears[1].weight)
        self.varstest.append(self.model.ginlayers[1].apply_func.mlp.linears[1].bias)
        self.varstest.append(self.model.ginlayers[2].apply_func.mlp.linears[0].weight)
        self.varstest.append(self.model.ginlayers[2].apply_func.mlp.linears[0].bias)
        self.varstest.append(self.model.ginlayers[2].apply_func.mlp.linears[1].weight)
        self.varstest.append(self.model.ginlayers[2].apply_func.mlp.linears[1].bias)
        self.varstest.append(self.model.ginlayers[3].apply_func.mlp.linears[0].weight)
        self.varstest.append(self.model.ginlayers[3].apply_func.mlp.linears[0].bias)
        self.varstest.append(self.model.ginlayers[3].apply_func.mlp.linears[1].weight)
        self.varstest.append(self.model.ginlayers[3].apply_func.mlp.linears[1].bias)
        self.varstest.append(self.model.embedding_h.weight)
        self.varstest.append(self.model.embedding_h.bias) 
        self.varstest.append(self.model.linears_prediction[0].weight) 
        self.varstest.append(self.model.linears_prediction[0].bias)
        self.varstest.append(self.model.linears_prediction[1].weight) 
        self.varstest.append(self.model.linears_prediction[1].bias)
        self.varstest.append(self.model.linears_prediction[2].weight)
        self.varstest.append(self.model.linears_prediction[2].bias)
        
        
#         check=list(self.vars_bn)
#         for i in range(len(check)):
#             print('check',i,check[i])
#         print('firstcheck2',self.vars[1].weight,self.vars[1].bias)
#         self.vars_bn.append(self.vars[1].weight)
#         self.vars_bn.append(self.vars[1].bias)



    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info
    
    
    
    



#     def forward(self, x, vars=None, bn_training=True):
    def forward(self, dataset,setname, vars=None, bn_training=True,init=False):
        
        
        trainset, valset, testset = dataset.train, dataset.val, dataset.test
        MODEL_NAME='GIN'
            
        if vars is None:
            vars=self.varstest
        
#         print('check1',init,vars[0])
        self.model.setpara(vars,init)
        self.modelbn=self.model.weight_bias()
        net_params=self.net_params
        params = self.params     
        device=net_params['device']
        optimizer = optim.Adam(self.model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
        
        
        
        epoch_train_losses, epoch_val_losses = [], []
        epoch_train_accs, epoch_val_accs = [], [] 
    
        drop_last = True if MODEL_NAME == 'DiffPool' else False
        if(setname=='train'):
            train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
            pred_loader=DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
        else:
            train_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
            pred_loader=DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
#         val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
#         test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)

    # At any point you can hit Ctrl + C to break out of training early.

        acclist=[]
        with tqdm(range(1)) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)
                    
#                     start = time.time()
#                 print('check1',model.embedding_h.weight)
#                     score,loss = train_epoch(model, optimizer, device, train_loader)
                score, epoch_train_acc, optimizer,pred,loss = train_epoch(self.model, optimizer, device, pred_loader, epoch)
#                     epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)

#                     epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_acc)
                acclist.append(epoch_train_acc)
#                 print(epoch,epoch_train_acc)



                


        return score,pred,loss


    def zero_grad(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.varstest:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()
                        
    
        
    
  
        
        
        
    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """


        return self.varstest


    
    
    