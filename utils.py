import torch
import pandas as pd
import dgl
from torch_geometric.data import Data
import random
import numpy as np


def prepare_text(dataset):
    """return the text of all nodes, and the text of labels"""
    if dataset in ['cornell', 'texas', 'wisconsin', 'washington']:
        df = pd.read_csv(f'./dataset/{dataset}/{dataset.capitalize()}.csv')
        raw_texts = df.raw_text.to_list()
    else:
        raw_texts_path = f'dataset/{dataset}/raw_texts.pt'
        raw_texts = torch.load(raw_texts_path)
    
    label_desc = pd.read_csv(f'dataset/{dataset}/categories.csv')
    labels = []
    num_label = len(label_desc)
    num_columns = label_desc.shape[1] 
    for row in range(num_label):
        label = label_desc.iloc[row][0]
        labels.append(label)
    return raw_texts, labels

def prepare_graph(dataset: str, encoder_name: str = 'llmicl_class_aware'):
    """return the graph and the embeddings (shape=(num_nodes, embedding_dim))"""
    if dataset in ['cornell', 'texas', 'wisconsin', 'washington']:
        dgl_graph = dgl.load_graphs(f'./dataset/{dataset}/{dataset.capitalize()}.pt')[0][0]
        edge_index = torch.stack(dgl_graph.edges())
        graph_data = Data(edge_index = edge_index, y = dgl_graph.ndata['label'])
        graph_data.test_id = torch.arange(len(graph_data.y))
        graph_data.train_id = torch.arange(len(graph_data.y))
        graph_data.val_id = torch.arange(len(graph_data.y))
    else:
        graph_data = torch.load(f'./dataset/{dataset}/processed_data.pt')
    embs = torch.load(f'./dataset/{dataset}/{encoder_name}_x.pt')
    if dataset in ['wikics']:
        pkg = torch.where(graph_data.test_mask)
        pkg2 = torch.where(graph_data.train_mask)
        pkg3 = torch.where(graph_data.val_mask)
        graph_data.test_id = pkg[0]
        graph_data.train_id = pkg2[0]
        graph_data.val_id = pkg3[0]
    return graph_data, embs

def prepare_model(model_type: str, num_classes: int, num_layers: int = 2, args=None):
    model_type = model_type.lower()
    if model_type == 'gcn':
        from torch_geometric.nn.models import GCN
        model = GCN(in_channels=4096, out_channels=num_classes, hidden_channels=512, num_layers=num_layers, jk='cat', dropout=args.dropout)
    elif model_type == 'gin':
        from torch_geometric.nn.models import GIN
        model = GIN(in_channels=4096, out_channels=num_classes, hidden_channels=512, num_layers=num_layers, jk='cat', dropout=args.dropout)
    elif model_type == 'glognn':
        from model.mlpnorm import get_mlpnorm
        model = get_mlpnorm(args.dataset, feature_dim=4096, hidden_dim=512, class_dim=num_classes)
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented.")
    return model

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

