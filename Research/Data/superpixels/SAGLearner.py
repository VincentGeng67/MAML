

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
from torch.nn import Parameter
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader
# from torch.utils.data import DataLoader
from torch_geometric import utils
import torch.nn.functional as F
from torch.utils.data import random_split,Subset



from SAGNet import SAGNet

import torch.optim as optim
from gnnf import gpu_setup,  train_epoch, evaluate_network, gnn_model, init_parameters
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling





class SAGLearner(nn.Module):
    """
    """

    def __init__(self,way):
        """
        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(SAGLearner, self).__init__()
        parser = argparse.ArgumentParser()
        

        parser.add_argument('--seed', type=int, default=777,
                    help='seed')
        parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size')
        parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
        parser.add_argument('--nhid', type=int, default=110,
                    help='hidden size')
        parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
        parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
        parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
        parser.add_argument('--epochs', type=int, default=2,
                    help='maximum number of epochs')
        parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
        parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')

        args, unknown = parser.parse_known_args()
        args.device = 'cpu'
        self.way=way
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            args.device = 'cuda:0'

        self.args=args
        self.device=args.device

        self.att = Parameter(torch.Tensor(1, 1* 2))
        
        args.num_classes = way
        args.num_features = 1
        self.model = SAGNet(args).to(args.device)
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
        self.modelbn=self.model.weight_bias()
        
        
        
        
        device = self.device
        if(setname=='train'):
            pred_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            self.model.train()
        else:
            pred_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
            self.model.eval()
            

        for i, data in enumerate(pred_loader):
            data = data.to(args.device)
            out = self.model(data.x, data.edge_index, data.batch)
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


    
    
    