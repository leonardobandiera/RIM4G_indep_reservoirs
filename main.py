import ctypes 
#import learn_ising 
from learn_ising import *
from sklearn.decomposition import PCA
#import torch.serialization
#torch.serialization.weights_only = False



import torch
import torch

# Salvo la funzione originale
_orig_load = torch.load

# Creo una versione modificata che forza weights_only=False
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)

# Rimpiazzo torch.load globalmente
torch.load = _patched_load
from torch.nn.functional import one_hot
from torch_geometric.datasets import WikipediaNetwork, WebKB, Actor, Planetoid
from graphesn import StaticGraphReservoir, initializer, Readout
from graphesn.util import graph_spectral_norm
import warnings 
import numpy as np
import subprocess
import itertools
from argparse import ArgumentParser
import sys
import os
warnings.filterwarnings('ignore')
import time 
import pandas as pd 
from sklearn.preprocessing import StandardScaler

from scipy.special import digamma
from sklearn.metrics import pairwise_distances
import torch.nn as nn 
import torch.optim as optim 
import random 
import copy 
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import ToUndirected



def get_dataset(root, name): 
    if name in ['chameleon','squirrel']: 
        return WikipediaNetwork(root=root, name=name, geom_gcn_preprocess=True)
    elif name in ['cornell', 'texas', 'wisconsin']:
        return WebKB(root=root, name=name)
    elif name in ['cora', 'citeseer', 'pubmed']:
        return Planetoid(root=root, name=name, split='geom-gcn')
    elif name == 'actor':
        return Actor(root=root)
    else:
        raise ValueError(f'Unknown dataset `{name}`')
        

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
    
def uncoupled_features(emb, q=0.9):

    H = emb.astype(np.float64)

    singular_values = np.linalg.svd( # computation of singular values
        H, compute_uv= False
    )
    cumulative = np.cumsum(singular_values) # cumulative sum of singular values
    cumulative /= cumulative[-1] #normalization

    uq = np.searchsorted(cumulative, q) + 1 # finds the first value above the threshold

    return uq
    
            
    
def ridge_regression(dataset,data,emb,dimension,dev=0):

    if dev==0:
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        print(device)
    else:
        device = 'cuda:1'
    #ld = 0
    y = one_hot(data.y, dataset.num_classes).float().to(device)
    x = torch.Tensor(emb).to(device)

    best_acc_test = 0
    bast_acc_val = 0

    test_acc_list =[]
    val_acc_list = []
    std_val_acc = [] 
    std_test_acc = []

    #readout = Readout(num_features=dimension,num_targets=dataset.num_classes).to(device)





    ld_list = ld_list=np.linspace(0.001, 25, 240)

    for ld in ld_list:
       
        lista_score_train = []

        lista_score_val = []

        lista_score_test = []

        

        for i in range(data.train_mask.shape[1]):
            readout = Readout(num_features=x.shape[1],num_targets=dataset.num_classes).to(device)
            readout.fit((x[data.train_mask[:,i]], y[data.train_mask[:,i]]), ld)
            y_pred = readout(x.to(device))
            lista_score_val.append((y_pred[data.val_mask[:,i]].argmax(dim=-1).to('cpu') == data.y[data.val_mask[:,i]]).float().to('cpu').mean() * 100)
            lista_score_test.append((y_pred[data.test_mask[:,i]].argmax(dim=-1).to('cpu') == data.y[data.test_mask[:,i]]).float().to('cpu').mean() * 100)

        lista_score_test = np.array(lista_score_test)
        lista_score_val = np.array(lista_score_val)
        mean_test = np.mean(lista_score_test)
        mean_val = np.mean(lista_score_val)
        test_acc_list.append(mean_test)
        val_acc_list.append(mean_val)
        std_val_acc.append(np.std(lista_score_val)) 
        std_test_acc.append(np.std(lista_score_test))

    #best_acc = max(test_acc_list)

    #return best_acc

    best_idx = max(range(len(val_acc_list)), key=lambda i: val_acc_list[i])
    best_val = val_acc_list[best_idx]
    best_test = test_acc_list[best_idx]
    std_val = std_val_acc[best_idx]
    std_test =std_test_acc[best_idx]

    return best_val, best_test,std_val,std_test

def main(args):
    data_ = str(args.dataset)
    dataset = get_dataset(root='./Dataset___', name=data_)
    dimension = int(args.dimension)
    save_dir = str(args.save_dir)
    #dir_ = "bias_"+str(args.bias)+"_no_init_"+str(matrix_type)+"_"+str(dimension)+"_"+str(w_connections)+"_"+str(args.random_state_w)
    #tot_dir = os.path.join(save_dir,dir_)

    #if not os.path.exists(tot_dir):
        #os.makedirs(tot_dir)

    emb = os.path.join(save_dir, "embedding2.npy")
    emb = np.load(emb)
    emb = emb.T
    emb = emb.astype("float32")
    data = dataset[0]

    pca = PCA(n_components=dimension, svd_solver="randomized")
    pca.fit(emb)
    emb = pca.components_
    print(emb.shape)
    emb = emb.T
    print(f"fatto!")
    ridge_val, ridge_test, std_val, std_test = ridge_regression(dataset, data, emb, dimension,args.dev)
    print(ridge_test)
    file_with_acc = os.path.join(save_dir,"ridge_val_accuracy"+str(dimension)+".txt") 
    file_ridge_test = os.path.join(save_dir,"ridge_test_accuracy"+str(dimension)+".txt")    
    energy_file =os.path.join(save_dir,"sim_energy"+str(dimension)+".txt") 
    file_with_val_std = os.path.join(save_dir,"std_val"+str(dimension)+".txt") 
    file_with_test_std = os.path.join(save_dir,"std_test"+str(dimension)+".txt")
    file_uncoupled = os.path.join(save_dir, "uncoupled_features"+str(dimension)+".txt")

if __name__ == "__main__": 
    parser = ArgumentParser() 
    parser.add_argument("--dataset", default= 'cora')
    parser.add_argument("--save_dir",default='./') 
    parser.add_argument("--dimension",type=int,default=4096)
    parser.add_argument("--dev", type=int, default=0)
    args =parser.parse_args() 
    main(args)
    pca.fit(emb)
    emb = pca.components_
    print(emb.shape)
