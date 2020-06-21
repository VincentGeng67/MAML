

import  torch
from    torch import nn
import argparse
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
from    copy import deepcopy
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader
# from torch.utils.data import DataLoader
from torch_geometric import utils
import torch.nn.functional as F
from torch.utils.data import random_split,Subset



from HGPmodel import HGPModel

import torch.optim as optim
from gnnf import gpu_setup,  train_epoch, evaluate_network, gnn_model, init_parameters
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling





class HGPLearner(nn.Module):
    """
    """

    def __init__(self,way):
        """
        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(HGPLearner, self).__init__()
        parser = argparse.ArgumentParser()
        

        parser.add_argument('--seed', type=int, default=777, help='random seed')
        parser.add_argument('--batch_size', type=int, default=512, help='batch size')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
        parser.add_argument('--nhid', type=int, default=110, help='hidden size')
        parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
        parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
        parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
        parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
        parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
        parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
        parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
        parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
        parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

        args, unknown = parser.parse_known_args()
        args.device = 'cpu'
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            args.device = 'cuda:0'

        self.args=args
        self.device=args.device
        self.way=way

        self.device=args.device


        
        args.num_classes = way
        args.num_features = 1
        self.model = HGPModel(args).to(args.device)
        self.varstest=self.model.weight_bias()



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
    def forward(self,dataset, setname, vars=None, bn_training=True,init=False):
        
        args=self.args
        trainset=dataset[0]
        testset=dataset[1]
            
        if vars is None:
            vars=self.varstest
        
        self.model.setpara(vars,init)
#         self.modelbn=self.varstest
        self.modelbn=self.model.weight_bias()
#         print('parameter1',list(self.model.pool1.sparse_attention.parameters()))
#         print('parameter2',list(self.model.pool1.calc_information_score.parameters()))
#         print('origin',self.model.parameters)
#         for ch in range(len(self.modelbn)):
#             print('check para ',ch,type(self.modelbn[ch]))
        
        device = self.device
        if(setname=='train'):
            pred_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            self.model.train()
        else:
            pred_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
            self.model.eval()
            

        for i, data in enumerate(pred_loader):
            data = data.to(args.device)
            out = self.model(data)
            pred = out.max(dim=1)[1]
            sz=len(data.y)
            ways=[]
            zeta=[]
            for j in range(sz):
                if data.y[j] in ways:
                    ways=ways
                else:
                    ways.append(data.y[j])
            for k in data.y:
                for h in range(len(ways)):
                    if ways[h]==k:
                        zeta.append(int(h))
            zeta=torch.LongTensor(zeta)
            loss = F.nll_loss(out, zeta)
        print('pred',pred)
        print('z',zeta)
        correct = (pred==zeta).float().sum().item()
        correct=correct/sz
#         check=deepcopy(self.model)
#         print('success')
        return out,pred,loss,correct


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


    
    
    