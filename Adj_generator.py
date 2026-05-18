import numpy as np 
import torch
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.transforms import ToUndirected
from torch.nn.functional import one_hot
from torch_geometric.datasets import WikipediaNetwork, WebKB, Actor, Planetoid
from graphesn import StaticGraphReservoir, initializer, Readout
from graphesn.util import *
from torch_geometric.utils import to_undirected
import time
import warnings 
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import argparse


def get_dataset(root, name):
    if name in ['chameleon', 'squirrel']:
        return WikipediaNetwork(root=root, name=name, geom_gcn_preprocess=True)
    elif name in ['cornell', 'texas', 'wisconsin']:
        return WebKB(root=root, name=name)
    elif name in ['cora', 'citeseer', 'pubmed']:
        return Planetoid(root=root, name=name, split='geom-gcn')
    elif name == 'actor':
        return Actor(root=root)
        
    elif name == 'arxiv': 
        return NodePropPredDataset(name="ogbn-arxiv", transform=ToUndirected())
        
    else:
        raise ValueError(f'Unknown dataset `{name}`')
        
        
        
        
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='cora')

args = parser.parse_args()

    

name = str(args.dataset)



dataset = get_dataset(root='./Dataset___',name=name)


data = dataset[0]

adj = to_undirected(data.edge_index) 

with open("./Adj_"+name+".txt", "w") as file:
    file.write(str(data.num_nodes) + "\n")
    for i in range(data.num_nodes):
        x = adj[0,adj[1,:] == i].numpy()
        file.write(str(len(x))+ " ")
        file.write(" ".join(map(str, x)) + "\n")

