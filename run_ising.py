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
    
            
    
def ridge_regression(dataset,data,emb,dimension, n_components=None):


    device = 'cpu'
    #ld = 0
    y = one_hot(data.y, dataset.num_classes).float().to(device)

    if n_components is not None:

        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(emb)
        pca = PCA(n_components=n_components)
        emb = pca.fit_transform(emb_scaled)

        print(f"PCA: {dimension} -> {emb.shape[1]}")

    x = torch.Tensor(emb, dtype=torch.float32).to(device)

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

    np.random.seed(0) 
    
    data_ = str(args.dataset) 
    
    dataset = get_dataset(root='./Dataset___',name=data_)
    
    data = dataset[0] 
    
    dimension = int(args.dimension) 
    
    matrix_type = int(args.w_type) 
    
    w_connections = int(args.w_connections)
    
    save_dir = str(args.save_dir)    
    
    dir_ = "bias_"+str(args.bias)+"_no_init_"+str(matrix_type)+"_"+str(dimension)+"_"+str(w_connections)+"_"+str(args.random_state_w)
    
    tot_dir = os.path.join(save_dir,dir_)
    
    if not os.path.exists(tot_dir):
        os.makedirs(tot_dir) 

    file_with_acc = os.path.join(tot_dir,"ridge_val_accuracy.txt") 
    file_ridge_test = os.path.join(tot_dir,"ridge_test_accuracy.txt")    
    energy_file =os.path.join(tot_dir,"sim_energy.txt") 
    file_with_val_std = os.path.join(tot_dir,"std_val.txt") 
    file_with_test_std = os.path.join(tot_dir,"std_test.txt")
    file_uncoupled = os.path.join(tot_dir, "uncoupled_features.txt")

    

    adj_matrix_name = "Adj_"+data_+".txt"
    adj_matrix_name =adj_matrix_name.encode('utf-8')
    
    model = Ising(
        data, 
        adj_matrix_name,
        data.num_nodes,
        args.steps,
        args.temperature,
        w_connections,
        args.random_state_w,
        args.random_state_iteration,
        dimension,
        args.bias,
        nthreads=args.n_threads,
        matrix_type=matrix_type,
        continue_=args.continue_,
        root_=tot_dir,
        data_=data_)
        
        
        
        
          
   
   
    num_loop = int((args.steps -args.initial_iter)/args.iter_) 
 
    initial_energy = model.calculate_energy()
    with open(energy_file,"a") as f: 
            f.write(str(initial_energy))
            f.write("\n")
            f.flush() 
            os.fsync(f.fileno())
    best_val = 0 
    best_ridge=0
   
    iter_ = args.initial_iter   
    
    for i in range(num_loop): 
         
        energy = model.simulate(iter_*data.num_nodes,initial_energy)
        _,emb = model.save_to_numpy(save_file=False)   #mettere True se si vuole salvare l'embedding su disco
        u90 = uncoupled_features(emb)

        components_list = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16]

        for nc in components_list:
            ridge_val, ridge_test,std_val,std_test= ridge_regression(dataset,data,emb,dimension,n_components=nc)

            print(f"PCA={nc}"
                  f"Val={ridge_val:.2f} "
                  f"Test={ridge_test:.2f}"
                )
            
        if args.iter_<3: 
            iter_=3
        else: 
            iter_=args.iter_  
        with open(file_with_acc,"a") as f: 
            f.write(str(ridge_val))
            f.write("\n")
            f.flush() 
            os.fsync(f.fileno())
        
        with open(energy_file,"a") as f: 
            f.write(str(energy))
            f.write("\n")
            f.flush() 
            os.fsync(f.fileno())  
        with open(file_with_val_std,"a") as f:
            f.write(str(std_val))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        with open(file_with_test_std,"a") as f:
            f.write(str(std_test))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        with open(file_ridge_test,"a") as f:
            f.write(str(ridge_test))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        with open(file_uncoupled, "a") as f:
            f.write(str(u90))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())


        if (initial_energy==energy): 
            break 
        initial_energy=energy


    model.clean_memory()
              
           
           
            
            
     
    
    
    
                            
 
                   
                    
                
                    
                        
if __name__ == "__main__": 
    parser = ArgumentParser() 
    parser.add_argument("--dataset", default= 'chameleon')
    parser.add_argument("--save_dir",default='./results') 
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument("--w_connections",type=int,default=128)
    parser.add_argument("--random_state_w",type=int,default=5)
    parser.add_argument("--random_state_iteration",type=int,default=5)
    parser.add_argument("--dimension",type=int,default=4096)
    parser.add_argument("--w_type",type=int,default=2)
    parser.add_argument("--n_threads",type=int,default=1)
    parser.add_argument("--continue_",type=int,default=0)  
    parser.add_argument("--bias",type=float,default=0)
    parser.add_argument("--hops",type=int,default=0)
    parser.add_argument("--initial_iter",type=int,default=700) 
    parser.add_argument("--iter_",type=int,default=20)   
    parser.add_argument("--threshold",type=float,default=2)
    parser.add_argument("--temperature",type=float,default=0)  
    args =parser.parse_args() 
    main(args)             
            
        
            
    
    
    
    
    
    
    
    
        
        

                    
                    
        
            
    

    
