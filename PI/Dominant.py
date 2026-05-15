import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_scatter import scatter_mean
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
from datetime import datetime

class GCNModelAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, dropout=0.):
        super(GCNModelAE, self).__init__()
        self.dropout = dropout
        self.encoder_conv1 = GCNConv(in_channels, hidden_channels)
        self.encoder_conv2 = GCNConv(hidden_channels, latent_channels)
        self.attr_decoder_conv1 = GCNConv(latent_channels, hidden_channels)
        self.attr_decoder_conv2 = GCNConv(hidden_channels, in_channels)
        
        self.struct_decoder_conv = GCNConv(latent_channels, hidden_channels)
        
    def encode(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.encoder_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.encoder_conv2(x, edge_index)
        x = F.relu(x)
        return x
    
    def decode_attributes(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attr_decoder_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attr_decoder_conv2(x, edge_index)
        x = F.relu(x)
        return x
    
    def decode_structure(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.struct_decoder_conv(x, edge_index)
        x = F.relu(x)
        adj_recon = torch.sigmoid(torch.mm(x, x.t()))
        return adj_recon
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_recon = self.decode_attributes(z, edge_index)
        adj_recon = self.decode_structure(z, edge_index)
        
        return x_recon, adj_recon, z

class AnomalyDetector:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train(self, data, optimizer, epochs=200):
        self.model.train()
        best_loss = float('inf')
        
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        adj = to_dense_adj(edge_index)[0]
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            x_recon, adj_recon, _ = self.model(x, edge_index)
           
            attr_loss = F.mse_loss(x_recon, x)
            struct_loss = F.binary_cross_entropy(adj_recon, adj)
            
            loss = 0.8 * attr_loss + 0.2 * struct_loss
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}, '
                      f'Attr Loss: {attr_loss.item():.4f}, Struct Loss: {struct_loss.item():.4f}')
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.model.state_dict(), 'best_model.pth')
    
    def detect_anomalies(self, data, threshold=0.5):
        
        self.model.eval()
        
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            adj = to_dense_adj(edge_index)[0]
            
            x_recon, adj_recon, z = self.model(x, edge_index)
            
            attr_errors = torch.mean((x - x_recon) ** 2, dim=1)
            struct_errors = torch.mean((adj - adj_recon) ** 2, dim=1)
            
            anomaly_scores = (attr_errors + struct_errors) / 2
            
            anomaly_labels = (anomaly_scores > threshold).float()
            
            scores = anomaly_scores.cpu().numpy()
            labels = anomaly_labels.cpu().numpy()
            
            node_details = []
            for i in range(len(scores)):
                node_details.append({
                    'node_id': i,
                    'anomaly_score': float(scores[i]),
                    'is_anomaly': bool(labels[i]),
                    'attr_error': float(attr_errors[i].cpu()),
                    'struct_error': float(struct_errors[i].cpu())
                })
            
            sorted_indices = np.argsort(-scores)
            
            return {
                'anomaly_scores': scores,
                'anomaly_labels': labels,
                'node_details': node_details,
                'sorted_indices': sorted_indices,
                'num_anomalies': int(labels.sum()),
                'threshold_used': threshold
            }