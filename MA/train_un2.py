import argparse
import os
from tqdm import tqdm
from data import AgentGraphDataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GATConv, global_mean_pool
from datetime import datetime
from Dominant import GCNModelAE
from TAM import TAMModel
import torch.nn.functional as F
import numpy as np
from Prem_gad import PREMModel


def rescale(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2


class ContrastiveGAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, heads=8, num_layers=2):
        super(ContrastiveGAE, self).__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.encoder_convs = nn.ModuleList()
        self.encoder_convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        
        for _ in range(num_layers - 2):
            self.encoder_convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads)
            )
        
        self.encoder_convs.append(
            GATConv(hidden_channels * heads, latent_channels, heads=1)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, latent_channels)
        )
        
    def encode(self, x, edge_index):
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x, edge_index)
            if i != len(self.encoder_convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        return x
    
    def get_graph_repr(self, x, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch)
    
    def forward(self, x, edge_index, batch=None):
        z = self.encode(x, edge_index)
        graph_repr = self.get_graph_repr(z, batch)
        
        z_proj = self.projection(z)
        graph_proj = self.projection(graph_repr)
        
        return z_proj, graph_proj

def info_nce_loss(features, batch, temperature=0.1):
    if len(features) == 0:
        return torch.tensor(0.0, device=features.device)
        
    batch_size = batch.max().item() + 1
    if batch_size < 2:
        return torch.tensor(0.0, device=features.device)
    graph_features = global_mean_pool(features, batch)
    features = F.normalize(features, dim=1)
    graph_features = F.normalize(graph_features, dim=1)
    
    sim_matrix = torch.mm(features, graph_features.t())  # [N, B]
    sim_matrix = sim_matrix / temperature
    
    labels = batch
    
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss

def train(model, train_loader, optimizer, defend_type, device):
    model.train()
    total_loss = 0
    num_batches = 0
    anamaly_ori_idx = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0]).to(device)
    
    for data in train_loader:
        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        x = edge_attr[:, 0, :]
        x = scatter_mean(x, edge_index[0], dim=0, dim_size=len(data.x))
        optimizer.zero_grad()
        # TAM
        if defend_type == "TAM":
            node_emb, feat1, feat2 = model(x, edge_index)
            num_nodes = x.size(0)
            adj = torch.eye(num_nodes, device=device)
            adj[edge_index[0], edge_index[1]] = 1.0
            loss_1, _ = model.max_message(node_emb, adj)
            reg_loss = model.reg_edge(node_emb, adj)
            loss = loss_1 + 1 * reg_loss

        # myself
        elif defend_type == "SCL":
            anamaly_idx = anamaly_ori_idx[torch.randperm(len(anamaly_ori_idx))]
            noise = torch.randn_like(x)
            noise = F.normalize(noise, dim=1)
            noise_magnitude = 0.8 * torch.norm(x, dim=1, keepdim=True)
            # print(x.shape, noise.shape, noise_magnitude.shape, anamaly_idx.shape)
            noised_embeddings = x + noise * noise_magnitude * anamaly_idx.unsqueeze(1)
            x = noised_embeddings
            node_emb = model.encode(x, edge_index)
            loss = model.neg_all(node_emb, anamaly_idx)
        
        # Dominant
        elif defend_type == "Dominant":
            x_recon, adj_recon, z = model(x, edge_index=edge_index)
            attr_loss = F.mse_loss(x_recon, x)
            num_nodes = x.size(0)
            adj = torch.eye(num_nodes, device=device)
            adj[edge_index[0], edge_index[1]] = 1.0
            struct_loss = F.binary_cross_entropy(adj_recon, adj)
            loss = 0.8 * attr_loss + 0.2 * struct_loss

        # PREM-GAD
        elif defend_type == "PREM":
            en_p, en_n, eg_p, eg_aug = model._get_prem_data(x, edge_index, device)
            score_pos = rescale(model(en_p, edge_index))
            score_aug = rescale(model(en_p, edge_index))
            score_nod = rescale(model(en_p, edge_index))
            
            num_nodes = x.size(0)
            label_zeros = torch.zeros(1, num_nodes).to(device)
            label_ones = torch.ones(1, num_nodes).to(device)
            
            loss_function = nn.BCELoss()
            loss_pos = loss_function(score_pos, label_zeros)
            loss_aug = loss_function(score_aug, label_ones)
            loss_nod = loss_function(score_nod, label_ones)
            
            alpha = 0.3
            gamma = 0.4
            loss = loss_pos + alpha * loss_aug + gamma * loss_nod
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

def test(model, test_loader, defend_type, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for data in test_loader:
            x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
            x = edge_attr[:, 0, :]
            x = scatter_mean(x, edge_index[0], dim=0, dim_size=len(data.x))
            
            # TAM
            if defend_type in ["TAM", "SCL"]:
                node_emb = model.encode(x, edge_index)
                num_nodes = x.size(0)
                adj = torch.eye(num_nodes, device=device)
                adj[edge_index[0], edge_index[1]] = 1.0
                loss1, _ = model.max_message(node_emb, adj)
                reg_loss = model.reg_edge(node_emb, adj)
                loss = loss1 + 1 * reg_loss
            
            elif defend_type == "Dominant":
                x_recon, adj_recon, z = model(x, edge_index=edge_index)
                attr_loss = F.mse_loss(x_recon, x)
                num_nodes = x.size(0)
                adj = torch.zeros((num_nodes, num_nodes), device=device)
                adj[edge_index[0], edge_index[1]] = 1.0
                struct_loss = F.binary_cross_entropy(adj_recon, adj)
                loss = 0.8 * attr_loss + 0.2 * struct_loss
            
            elif defend_type == "PREM":
                en_p, en_n, eg_p, eg_aug = model._get_prem_data(x, edge_index, device)
                
                score_pos = rescale(model(en_p, edge_index))
                score_aug = rescale(model(en_p, edge_index))
                score_nod = rescale(model(en_p, edge_index))
                
                num_nodes = x.size(0)
                label_zeros = torch.zeros(1, num_nodes).to(device)
                label_ones = torch.ones(1, num_nodes).to(device)
                
                loss_function = nn.BCELoss()
                loss_pos = loss_function(score_pos, label_zeros)
                loss_aug = loss_function(score_aug, label_ones)
                loss_nod = loss_function(score_nod, label_ones)
                
                alpha = 0.3
                gamma = 0.4
                loss = loss_pos + alpha * loss_aug + gamma * loss_nod

            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiments to train Contrastive Graph Autoencoder")
    
    parser.add_argument("--dataset_path", type=str, default="./ModelTrainingSet/memory_attack/dataset.pkl", help="Save path of the dataset")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0002)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    
    parser.add_argument("--save_dir", type=str, default="./checkpoint_un2")
    parser.add_argument("--defend_type", type=str, default="SCL", choices=["SCL", "TAM", "Dominant", "PREM"])
    
    # PREM-GAD specific parameters
    parser.add_argument("--prem_k", type=int, default=2, help="PREM aggregation steps")
    parser.add_argument("--prem_alpha", type=float, default=0.3, help="PREM augmentation loss weight")
    parser.add_argument("--prem_gamma", type=float, default=0.4, help="PREM node loss weight")
    
    args = parser.parse_args()
    
    normalized_path = os.path.normpath(args.dataset_path)
    parts = normalized_path.split(os.sep)
    dataset = parts[-2]
    args.save_dir = os.path.join(args.save_dir, dataset)

    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)

    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_time_str}-defend_type_{args.defend_type}-hiddim_{args.hidden_dim}-heads_{args.num_heads}-layers_{args.num_layers}-epochs_{args.epochs}-lr_{args.lr}-dropout_{args.dropout}-wd_{args.weight_decay}.pth"
    args.save_path = os.path.join(args.save_dir, filename)
    
    return args

def main():
    args = parse_arguments()
    
    train_dataset = AgentGraphDataset(args.dataset_path, phase="train")
    val_dataset = AgentGraphDataset(args.dataset_path, phase="val")
    
    
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_batch_size = min(args.batch_size, max(2, len(val_dataset) // 10))
    testloader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    
    example = train_dataset[0]
    edge_attr = example.edge_attr
    in_channels = edge_attr.size(-1)
    
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # TAM + myself
    if args.defend_type in ["SCL", "TAM"]:
        model = TAMModel(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            out_channels=args.latent_dim,
            dropout=0,
            readout='avg'
        )
    elif args.defend_type == "Dominant":
        model = GCNModelAE(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            latent_channels=args.latent_dim,
            dropout=0.
        )
    elif args.defend_type == "PREM":
        model = PREMModel(
            n_in=in_channels,
            n_hidden=args.hidden_dim,
            k=args.prem_k
        )

    model.to(device)
    
    for param in model.parameters():
        param.data = param.data.float()
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train(model, trainloader, optimizer, args.defend_type, device=device)
        test_loss = test(model, testloader, args.defend_type, device=device)
        scheduler.step()
        
        print(f"Epoch {epoch}/{args.epochs} || Training Loss: {train_loss:.4f} || Test Loss: {test_loss:.4f} || Samples in batch: {test_batch_size}")
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), args.save_path)
            print("Model saved!")

if __name__ == "__main__":
    main()