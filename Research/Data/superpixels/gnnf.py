import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm
from gin_net import GINNet
from gin_layer import GINLayer, ApplyNodeFunc, MLP

def accuracy(scores, targets):
#     print('t1',scores)
    scores = scores.detach().argmax(dim=1)

    acc = (scores==targets).float().sum().item()

    return acc

def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        
        device = torch.device("cuda")
    else:
        
        device = torch.device("cpu")
    return device
    
# def view_model_param(MODEL_NAME, net_params,parameter):
#     models = {
#         'GIN': GIN,
#     }
        
#     model = models[MODEL_NAME](net_params,parameter)
#     total_param = 0
# #     for param in model.parameters():
# #          total_param += np.prod(list(param.data.size()))
# #     return total_param
#     return net_params
    
def gnn_model(MODEL_NAME, net_params,parameter=None,init=False):
    models = {
         'GIN': GIN,
        }
        
    return models[MODEL_NAME](net_params,parameter,init)

def GIN(net_params,parameter=None,init=False):
    return GINNet(net_params,parameter,init)
    
def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    score=[]
    losslist=[]
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_n = batch_snorm_n.to(device)         # num x 1
#         print('g1',model.ginlayers[0].apply_func.mlp.linears[0].weight.grad)
#         optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
        
#         print('target',batch_labels)
        loss = model.loss(batch_scores, batch_labels)
#         loss.backward()
#         print('g2',model.ginlayers[0].apply_func.mlp.linears[0].weight.grad)


#         print('grad',model.ginlayers[0].apply_func.mlp.linears[0].weight.grad)
#         for param in model.parameters():
#             print(param.grad.data.sum())
#         print('loss2',loss)
#         print('para2',model.embedding_h.weight)
#         optimizer.step()
#         print('para',model.ginlayers[0].apply_func.mlp.linears[0].weight)
#         print('loss3',loss)
#         print('para3',model.embedding_h.weight)
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
        score.append(batch_scores)
        losslist.append(loss)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
#     it=len(score[0])
#     pred=[]
#     for sco in (score[0]):
#         sco=sco.tolist()
#         pred.append(sco.index(max(sco)))
#     pred=torch.Tensor(pred)

    pred = batch_scores.detach().argmax(dim=1)
#     print('pred',pred)
#     print('labelin',batch_labels)
    return batch_scores, epoch_train_acc, optimizer,pred,loss
#     return score, losslist

def init_parameters(inparams,innet_params):
    params=inparams
    net_params=innet_params


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
        
    rvars =[]
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
            
    rvars.append(ginlayers)
    rvars.append(embedding_h)
    rvars.append(linears_prediction)
    return rvars

def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)

            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc