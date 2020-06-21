

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

def test(model,loader,device):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.max(dim=1)[1]
        print('y,',data.y)
        print('pred',pred)
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)




class SAGLearnerori(nn.Module):
    """
    """

    def __init__(self,dataset):
        """
        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(SAGLearnerori, self).__init__()
        parser = argparse.ArgumentParser()
        

        parser.add_argument('--seed', type=int, default=777,
                    help='seed')
        parser.add_argument('--batch_size', type=int, default=128,
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
        parser.add_argument('--epochs', type=int, default=100000,
                    help='maximum number of epochs')
        parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
        parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')

        args, unknown = parser.parse_known_args()
        args.device = 'cpu'
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            args.device = 'cuda:0'

        self.args=args
        self.device=args.device
        self.dataset=dataset

        
        args.num_classes = dataset.num_classes
        args.num_features = dataset.num_features
        



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
    def forward(self,vars=None, bn_training=True,init=False):
        args=self.args
        num_training = int(len(self.dataset)*0.8)
        num_val = int(len(self.dataset)*0.1)
        num_test = len(self.dataset) - (num_training+num_val)
        training_set,validation_set,test_set = random_split(self.dataset,[num_training,num_val,num_test])



        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
        test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
        model = SAGNet(args).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)    
        
        
        
        
        device = self.device
        min_loss = 1e10
        patience = 0

        for epoch in range(args.epochs):
            model.train()
            for i, data in enumerate(train_loader):
                data = data.to(args.device)
                out = model(data.x, data.edge_index, data.batch)
                loss = F.nll_loss(out, data.y)
                print("Training loss:{}".format(loss.item()))
                loss.backward()
                print('grad',list(model.parameters())[0].grad)
                
                optimizer.step()
                print('para',list(model.parameters())[0])
                optimizer.zero_grad()
            val_acc,val_loss = test(model,val_loader,device)
            print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
            if val_loss < min_loss:
                torch.save(model.state_dict(),'latest.pth')
                print("Model saved at epoch{}".format(epoch))
                min_loss = val_loss
                patience = 0
            else:
                patience += 1
            if patience > args.patience:
                break 

        model = Net(args).to(args.device)
        model.load_state_dict(torch.load('latest.pth'))
        test_acc,test_loss = test(model,test_loader,test)
        loss=test_loss
        correct=test_acc
        print("Test accuarcy:{}".fotmat(test_acc))
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


    
    
    