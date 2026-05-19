import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, GATConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_scatter import scatter_mean
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
from tqdm import tqdm
import time
from torch_sparse import SparseTensor

import pickle
import math
linalg_device = 'cpu'

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.prelu = nn.PReLU()
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.prelu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class TAMModel(nn.Module):
    """
    Truncated Affinity Maximization Model for Graph Anomaly Detection
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0, readout='avg'):
        super(TAMModel, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels, dropout)
        self.fc1 = nn.Linear(out_channels, hidden_channels)
        self.fc2 = nn.Linear(out_channels, hidden_channels)
        self.readout = readout
        self.fc4 = nn.Linear(in_channels*2, out_channels)
        self.fc5 = nn.Linear(in_channels*2, 1)
        
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        z1 = self.fc1(z)
        z2 = self.fc2(z)
        
        return z, z1, z2

    def encode1(self, x, edge_index):
        num_nodes = x.size(0)
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        x_nei = adj_t @ x
        x = torch.cat((x, x_nei), 1)
        z = self.fc4(x)
        return z
    
    def encode(self, x, edge_index):
        num_nodes = x.size(0)
        x_nei = x.mean(0).unsqueeze(0).repeat(num_nodes, 1)
        x = torch.cat((x, x_nei), 1)
        z = self.fc4(x)
        return z


    
    def get_embedding(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            z, _, _ = self.forward(x, edge_index)
        return z

    def reg_edge(self, emb, adj):
        emb = F.normalize(emb, p=2, dim=-1)
        sim = torch.mm(emb, emb.t())
        adj_inv = (1 - adj)
        sim = sim * adj_inv
        sim_sum = torch.sum(sim, 1)
        row_sum = torch.sum(adj_inv, 1)
        r_inv = torch.pow(row_sum, -1)
        r_inv[torch.isinf(r_inv)] = 0.
        sim_sum = sim_sum * r_inv
        return torch.sum(sim_sum)
    
    def neg_all(self, emb, anamaly_idx, tem=0.3):
        emb = F.normalize(emb, p=2, dim=-1)
        sim = torch.mm(emb, emb.t()) / tem
        exp_sim = torch.exp(sim)
        mask = torch.eq(anamaly_idx.unsqueeze(1), anamaly_idx.unsqueeze(0)).int()
        neg_mask = 1 - mask
        value_neg = (exp_sim*neg_mask).sum(dim=1, keepdim=True)
        value_pos = -((torch.log(exp_sim / (1e-8 + (exp_sim + value_neg))) * mask).sum(1)) / mask.sum(0)
        loss = value_pos.mean()
        return loss
    
    def ana_class(self, x, edge_index, anamaly_idx):
        num_nodes = x.size(0)
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
        #adj_t = adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        x_nei = adj_t @ x
        x = torch.cat((x, x_nei), 1)
        x = self.fc5(x)
        loss = nn.BCEWithLogitsLoss()(x, anamaly_idx.float().unsqueeze(-1))
        return loss
    
    def ana_infer(self, x, edge_index):
        num_nodes = x.size(0)
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
        #adj_t = adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        x_nei = adj_t @ x
        x = torch.cat((x, x_nei), 1)
        x = self.fc5(x)
        return x
    
    def neg_all1(self, emb, anamaly_idx, tem=0.3):
        emb = F.normalize(emb, p=2, dim=-1)
        sim = torch.mm(emb, emb.t())
        mask = torch.eq(anamaly_idx.unsqueeze(1), anamaly_idx.unsqueeze(0)).int()
        neg_mask = 1 - mask
        loss = ((sim * neg_mask).sum(1) / (neg_mask.sum(1))).mean()
        return loss
    
    def max_message(self, feature, adj_matrix):
        feature = F.normalize(feature, p=2, dim=-1)
        sim_matrix = torch.mm(feature, feature.t())
        sim_matrix = sim_matrix * adj_matrix
        
        sim_matrix[torch.isinf(sim_matrix)] = 0
        sim_matrix[torch.isnan(sim_matrix)] = 0
        
        row_sum = torch.sum(adj_matrix, 0)
        r_inv = torch.pow(row_sum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        
        message = torch.sum(sim_matrix, 1)
        message = message * r_inv
        
        return -torch.sum(message), message
    
    def inference(self, feature, adj_matrix):
        feature = F.normalize(feature, p=2, dim=-1)
        sim_matrix = torch.mm(feature, feature.t())
        sim_matrix = sim_matrix * adj_matrix
        
        row_sum = torch.sum(adj_matrix, 1)
        r_inv = torch.pow(row_sum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        message = torch.sum(sim_matrix, 1)
        message = message * r_inv
        return message
    
    def inference_new(self, feature, adj_matrix):
        feature = F.normalize(feature, p=2, dim=-1)
        sim_matrix = torch.mm(feature, feature.t())
        message = torch.sum(sim_matrix, 1)
        return message
    
    @staticmethod
    def normalize_score(scores):
        return (scores - scores.min()) / (scores.max() - scores.min()+1e-2)


class GATSCL(nn.Module):
    """
    Truncated Affinity Maximization Model for Graph Anomaly Detection (with GAT)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads = 8, dropout=0., readout='avg', type = 0):
        super(GATSCL, self).__init__()
        self.gat = GATConv(in_channels, hidden_channels//heads, heads=heads, concat=True, dropout=dropout)
        if type == 0:
            self.fc = nn.Linear(in_channels * 3, out_channels)
        elif type == 3:
            self.fc = nn.Linear(in_channels * 1, out_channels)
        else:
            self.fc = nn.Linear(in_channels * 2, out_channels)
        self.fc1 = nn.Linear(out_channels, out_channels)
        self.type = type
        # self.fc2 = nn.Linear(out_channels, out_channels)
        self.readout = readout
        self.dropout = dropout
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
    
    def forward(self, x, edge_index):
        z = self.gat(x, edge_index)
        z1 = self.fc1(z)
        z2 = self.fc2(z)
        
        return z, z1, z2

    def encode1(self, x, edge_index):
        num_nodes = x.size(0)
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
        #adj_t = adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        x_nei = adj_t @ x
        x = torch.cat((x, x_nei), 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        z = self.fc4(x)
        return z
    
    def encode2(self, x, edge_index):
        num_nodes = x.size(0)
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
        #adj_t = adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        x_nei = adj_t @ x
        x_graph = x.mean(0).unsqueeze(0).repeat(num_nodes, 1)
        x = torch.cat((x, x_nei), 1)
        x = torch.cat((x, x_graph), 1)
        z = self.fc(x)
        return z
    
    def encode(self, x, edge_index):
        num_nodes = x.size(0)
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
        #adj_t = adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        x_nei = adj_t @ x
        x_graph = x.mean(0).unsqueeze(0).repeat(num_nodes, 1)
        if self.type == 0:
            x = torch.cat((x, x_nei), 1)
            x = torch.cat((x, x_graph), 1)
        elif self.type == 1:
            x = torch.cat((x, x_nei), 1)
        elif self.type == 2:
            x = torch.cat((x, x_graph), 1)
        elif self.type == 3:
            x = x
        # import ipdb; ipdb.set_trace()
        z = self.fc(x)
        return z
    
    def get_embedding(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            z, _, _ = self.forward(x, edge_index)
        return z
        
    def reg_edge(self, emb, adj):
        emb = F.normalize(emb, p=2, dim=-1)
        sim = torch.mm(emb, emb.t())
        adj_inv = (1 - adj)
        sim = sim * adj_inv
        sim_sum = torch.sum(sim, 1)
        row_sum = torch.sum(adj_inv, 1)
        r_inv = torch.pow(row_sum, -1)
        r_inv[torch.isinf(r_inv)] = 0.
        sim_sum = sim_sum * r_inv
        return torch.sum(sim_sum)
    
    def neg_all1(self, emb, anamaly_idx, tem=0.3):
        emb = self.bn(emb)
        emb = F.normalize(emb, p=2, dim=-1)
        sim = torch.mm(emb, emb.t()) / tem
        exp_sim = torch.exp(sim)
        mask = torch.eq(anamaly_idx.unsqueeze(1), anamaly_idx.unsqueeze(0)).int()
        #loss = (sim * (1-mask)).mean() - (sim * mask).mean()
        neg_mask = 1 - mask
        value_neg = (exp_sim*neg_mask).sum(dim=1, keepdim=True)
        value_pos = -((torch.log(exp_sim / (1e-8 + (exp_sim + value_neg))) * mask).sum(1)) / mask.sum(0)
        loss = value_pos.mean()
        return loss

    def neg_all(self, emb, anamaly_idx, tem=0.3):
        emb = F.relu(emb)
        emb = self.fc1(emb)
        emb = F.normalize(emb, p=2, dim=-1)
        sim = torch.mm(emb, emb.t()) / tem
        exp_sim = torch.exp(sim)
        mask = torch.eq(anamaly_idx.unsqueeze(1), anamaly_idx.unsqueeze(0)).int()
        #loss = (sim * (1-mask)).mean() - (sim * mask).mean()
        neg_mask = 1 - mask
        value_neg = (exp_sim*neg_mask).sum(dim=1, keepdim=True)
        value_pos = -((torch.log(exp_sim / (1e-8 + (exp_sim + value_neg))) * mask).sum(1)) / mask.sum(0)
        loss = value_pos.mean()
        return loss
    
    def ana_infer(self, x, edge_index):
        num_nodes = x.size(0)
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
        #adj_t = adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        x_nei = adj_t @ x
        x = torch.cat((x, x_nei), 1)
        x = self.fc5(x)
        return x
    
    def neg_all2(self, emb, anamaly_idx, tem=0.3):
        #import ipdb; ipdb.set_trace()
        emb = F.normalize(emb, p=2, dim=-1)
        sim = torch.mm(emb, emb.t())
        mask = torch.eq(anamaly_idx.unsqueeze(1), anamaly_idx.unsqueeze(0)).int()
        neg_mask = 1 - mask
        loss = ((sim * neg_mask).sum(1) / (neg_mask.sum(1))).mean()
        return loss
    
    def max_message(self, feature, adj_matrix):
        feature = F.normalize(feature, p=2, dim=-1)
        sim_matrix = torch.mm(feature, feature.t())
        sim_matrix = sim_matrix * adj_matrix
        
        sim_matrix[torch.isinf(sim_matrix)] = 0
        sim_matrix[torch.isnan(sim_matrix)] = 0
        
        row_sum = torch.sum(adj_matrix, 0)
        r_inv = torch.pow(row_sum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        
        message = torch.sum(sim_matrix, 1)
        message = message * r_inv
        
        return -torch.sum(message), message
    
    def inference(self, feature, adj_matrix):
        feature = F.normalize(feature, p=2, dim=-1)
        sim_matrix = torch.mm(feature, feature.t())
        sim_matrix = sim_matrix * adj_matrix
        
        row_sum = torch.sum(adj_matrix, 1)
        r_inv = torch.pow(row_sum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        message = torch.sum(sim_matrix, 1)
        message = message * r_inv
        return message
    
    def inference_new(self, feature, adj_matrix):
        feature = F.normalize(feature, p=2, dim=-1)
        sim_matrix = torch.mm(feature, feature.t())
        message = torch.sum(sim_matrix, 1)
        return message
    
    @staticmethod
    def normalize_score(scores):
        return (scores - scores.min()) / (scores.max() - scores.min()+1e-2)