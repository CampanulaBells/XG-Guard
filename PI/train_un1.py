import argparse
import os
from tqdm import tqdm
from data import AgentGraphDataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW 
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GATConv
from einops import rearrange
from datetime import datetime
from Dominant import GCNModelAE
from TAM import TAMModel, GATSCL
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cal_AUROC(labels, probs, graph_size=8):
    pred_y = torch.zeros(graph_size, dtype=torch.long)
    true_y = torch.zeros(graph_size, dtype=torch.long)
    true_y[labels] = 1
    pred_y[probs] = 1
    labels = true_y.numpy()
    probs = pred_y.numpy()
    score_AUROC = roc_auc_score(labels, probs)
    # score['AUPRC'] = average_precision_score(labels, probs)
    return score_AUROC

class MyGAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, out_channels, heads=8, num_layers=2, edge_dim=None):
        super(MyGAE, self).__init__()
        self.num_layers = num_layers
        self.heads = heads
        
        self.time_steps, self.edge_feat_dim = edge_dim
        self.total_edge_dim = self.time_steps * self.edge_feat_dim
        
        self.encoder_convs = nn.ModuleList()
        self.encoder_convs.append(GATConv(in_channels, hidden_channels, heads=heads, edge_dim=self.total_edge_dim))
        
        for _ in range(num_layers - 2):
            self.encoder_convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=self.total_edge_dim)
            )
        
        self.encoder_convs.append(
            GATConv(hidden_channels * heads, latent_channels, heads=1, edge_dim=self.total_edge_dim)
        )
        
        self.decoder_convs = nn.ModuleList()
        self.decoder_convs.append(GATConv(latent_channels, hidden_channels, heads=heads, edge_dim=self.total_edge_dim))
        
        for _ in range(num_layers - 2):
            self.decoder_convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=self.total_edge_dim)
            )
        
        self.decoder_convs.append(
            GATConv(hidden_channels * heads, out_channels, heads=1, edge_dim=self.total_edge_dim)
        )
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(latent_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, self.time_steps * self.edge_feat_dim)
        )
        
    def encode(self, x, edge_index, edge_attr):
        edge_attr = edge_attr.view(edge_attr.size(0), -1)
        
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x, edge_index, edge_attr)
            if i != len(self.encoder_convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        return x
        
    def decode(self, z, edge_index, edge_attr):
        edge_attr = edge_attr.view(edge_attr.size(0), -1)
        
        for i, conv in enumerate(self.decoder_convs):
            z = conv(z, edge_index, edge_attr)
            if i != len(self.decoder_convs) - 1:
                z = F.relu(z)
                z = F.dropout(z, p=0.2, training=self.training)
        
        return z
        
    def reconstruct_edge_attr(self, z, edge_index):
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]
        edge_input = torch.cat([z_src, z_dst], dim=1)
        pred_edge_attr = self.edge_predictor(edge_input)
        pred_edge_attr = pred_edge_attr.view(-1, self.time_steps, self.edge_feat_dim)
        
        return pred_edge_attr
        
    def forward(self, x, edge_index, edge_attr):
        z = self.encode(x, edge_index, edge_attr)
        reconstructed_x = self.decode(z, edge_index, edge_attr)
        reconstructed_edge_attr = self.reconstruct_edge_attr(z, edge_index)
        return reconstructed_x, reconstructed_edge_attr

def reconstruction_loss(original, reconstructed, edge_index):
    node_loss = nn.MSELoss()(original, reconstructed)
    return node_loss

def train(model, train_loader, optimizer, defend_type, device):
    model.train()
    total_loss = 0
    num_batches = 0
    topk = 3
    num_auc = 0
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
            noised_embeddings = x + noise * noise_magnitude * anamaly_idx.unsqueeze(1)
            x = noised_embeddings
            node_emb = model.encode(x, edge_index)
            loss = model.neg_all(node_emb, anamaly_idx)

            # import ipdb;ipdb.set_trace()
            num_nodes = node_emb.size(0)
            adj = torch.eye(num_nodes)
            adj[edge_index[0], edge_index[1]] = 1.0
            message = model.inference_new(node_emb, adj)
            _, predicts = torch.topk(-message, topk)
            labels = torch.where(anamaly_idx == 1)[0].tolist()
            num_auc += cal_AUROC(labels, predicts.tolist())
        
        # Dominant
        elif defend_type == "Dominant":
            x_recon, adj_recon, z = model(x, edge_index=edge_index)
            attr_loss = F.mse_loss(x_recon, x)
            num_nodes = x.size(0)
            adj = torch.eye(num_nodes, device=device)
            adj[edge_index[0], edge_index[1]] = 1.0
            struct_loss = F.binary_cross_entropy(adj_recon, adj)
            loss = 0.8 * attr_loss + 0.2 * struct_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1), num_auc / max(len(train_loader), 1)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiments to train Graph Autoencoder")
    
    parser.add_argument("--dataset", type=str, default="mmlu")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0002)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    
    parser.add_argument("--save_dir", type=str, default="./checkpoint_un2")
    parser.add_argument("--defend_type", type=str, default="SCL")
    parser.add_argument("--rep_type", type=int, default=0)
    
    args = parser.parse_args()
    
    if args.dataset == "mmlu": 
        args.dataset_path = "./ModelTrainingSet/mmlu/dataset.pkl"
    elif args.dataset == "csqa": 
        args.dataset_path = "./ModelTrainingSet/csqa/dataset.pkl"
    elif args.dataset == "gsm8k": 
        args.dataset_path = "./ModelTrainingSet/gsm8k/dataset.pkl"
    else: 
        raise Exception(f"Unknown dataset {args.dataset}")
    
    args.save_dir = os.path.join(args.save_dir, args.dataset)
    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)
        
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_time_str}-defend_type_{args.defend_type}-hiddim_{args.hidden_dim}-latent_{args.latent_dim}-heads_{args.num_heads}-layers_{args.num_layers}-epochs_{args.epochs}-lr_{args.lr}-rep_type_{args.rep_type}.pth"
    args.save_path = os.path.join(args.save_dir, filename)
    
    return args

def main():
    args = parse_arguments()
    set_seed(42)
    train_dataset = AgentGraphDataset(args.dataset_path, phase="train")
    import pickle
    
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    example = train_dataset[0]
    edge_attr = example.edge_attr
    in_channels = edge_attr.size(-1)
    
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    # TAM + myself
    if args.defend_type in ["TAM"]:
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
    elif args.defend_type == "SCL":
        model = GATSCL(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            out_channels=args.latent_dim,
            type=args.rep_type
        )

    model.to(device)
    
    for param in model.parameters():
        param.data = param.data.float()
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    
    best_loss = float('inf')
    best_auc = 0
    
    for epoch in range(args.epochs):
        train_loss, train_auc = train(model, trainloader, optimizer, args.defend_type, device=device)
        scheduler.step()
        
        print(f"Epoch {epoch}/{args.epochs} || Training Loss: {train_loss:.4f} || Training AUC: {train_auc:.4f} || Best auc: {best_auc:.4f} || Samples in batch: 1")
        
        if train_auc > best_auc:
            best_auc = train_auc
            torch.save(model.state_dict(), args.save_path)
            print("Model saved!")

if __name__ == "__main__":
    main()