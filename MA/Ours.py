
import warnings
from requests.exceptions import RequestsDependencyWarning
warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
import pickle
import argparse
import os
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from typing import Literal

from torch_geometric.loader import DataLoader
from torch.optim import Adam, AdamW
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import random
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

# %%

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser(description="Experiment configuration")

    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0002, help="Weight decay for optimizer")
    parser.add_argument("--alpha", type=float, default=0.0001, help="alpha parameter")
    parser.add_argument("--seed", type=int, default=3701, help="Random seed")
    parser.add_argument("--experiment", type=str, default="PI-CSQA", help="Experiment name")
    parser.add_argument("--save_ckpt", type=int, default=0, help="")
    parser.add_argument("--save_results", type=int, default=1, help="")

    args = parser.parse_args()

    device = args.device
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    alpha = args.alpha
    seed = args.seed
    EXPERIMENT = args.experiment

    # 生成参数字典
    config = {
        "device": device,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "alpha": alpha,
        "seed": seed,
        "experiment": EXPERIMENT,
    }
    return args, config



args, config = get_args()
save_ckpt = args.save_ckpt != 0
save_results = args.save_results != 0


device = args.device
epochs = args.epochs
lr = args.lr
weight_decay = args.weight_decay
alpha = args.alpha
seed = args.seed
EXPERIMENT = args.experiment

# device = "cuda"
# epochs = 20
# lr = 0.0001
# weight_decay = 0.0002
# alpha = 0.0001
# seed = 3701
# EXPERIMENT = "PI-CSQA"


set_seed(seed)
EXPERIMENT_AVAILABLE = [
    "PI-CSQA",
    "PI-MMLU",
    "PI-GSM8K",
    "TA-InjecAgent",
    "MA-PoisonRAG",
    "MA-CSQA",
]
assert EXPERIMENT in EXPERIMENT_AVAILABLE
PATH_CONFIG = {
    "MA-PoisonRAG": {
        # "train": "./agent_graph_dataset/memory_attack/train1/edit_dataset.json",
        "train": "G:/AgentGAD/MA_PoisonRAG_dummy_train_dataset.json",
        "test": "G:/AgentGAD/MA_PoisonRAG_dummy_test_dataset.json",
        "emb_cache": "cahced_data_MA_PoisonRAG.pkl",
        "emb_cache_test": "cahced_data_MA_PoisonRAG_test.pkl"
    },
    "MA-CSQA": {
        # "train": "G:/AgentGAD/BlindGuard/datasets/MA-CSQA/agent_graph_dataset/memory_attack/train/dataset.json",
        "train": "G:/AgentGAD/MA_CSQA_dummy_train_dataset.json",
        "test": "G:/AgentGAD/MA_CSQA_dummy_test_dataset.json",
        "emb_cache": "cahced_data_MA_CSQA.pkl",
        "emb_cache_test": "cahced_data_MA_CSQA_test.pkl",
    },
    "TA-InjecAgent": {
        # "train": "G:/AgentGAD/BlindGuard/datasets/TA/agent_graph_dataset/tool_attack/train1/dataset.json",
        "train": "G:/AgentGAD/TA_InjecAgent_dummy_train_dataset.json",
        "test": "G:/AgentGAD/TA_InjecAgent_dummy_test_dataset.json",
        "emb_cache": "cahced_data_TA_InjecAgent.pkl",
        "emb_cache_test": "cahced_data_TA_InjecAgent_test.pkl",
    },
    "PI-CSQA": {
        # "train": "G:/AgentGAD/BlindGuard/datasets/PI/agent_grapeh_dataset/csqa/train1/dataset.json",
        "train": "G:/AgentGAD/PI_CSQA_dummy_train_dataset.json",
        "test": "G:/AgentGAD/PI_CSQA_dummy_test_dataset.json",
        "emb_cache": "cahced_data_PI_CSQA.pkl",
        "emb_cache_test": "cahced_data_PI_CSQA_test.pkl",
    },
    "PI-GSM8K": {
        # "train": "G:/AgentGAD/BlindGuard/datasets/PI/agent_graph_dataset/gsm8k/train1/dataset.json",
        "train": "G:/AgentGAD/PI_GSM8K_dummy_train_dataset.json",
        "test": "G:/AgentGAD/PI_GSM8K_dummy_test_dataset.json",
        "emb_cache": "cahced_data_PI_GSM8K.pkl",
        "emb_cache_test": "cahced_data_PI_GSM8K_test.pkl",
    },
    "PI-MMLU": {
        # "train": "G:/AgentGAD/BlindGuard/datasets/PI/agent_graph_dataset/mmlu/train1/dataset.json",
        "train": "G:/AgentGAD/PI_MMLU_dummy_train_dataset.json",
        "test": "G:/AgentGAD/PI_MMLU_dummy_test_dataset.json",
        "emb_cache": "cahced_data_PI_MMLU.pkl",
        "emb_cache_test": "cahced_data_PI_MMLU_test.pkl",
    },
}



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



def _mlp(in_dim, hidden_dim, out_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.PReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        if num_layers == 1:

            # torch.nn.init.normal_(self.x_proj.weight
            self.convs = nn.ModuleList([GCNConv(in_channels, out_channels)])
            self.norms = nn.ModuleList([])
            torch.nn.init.normal_(self.convs[0].lin.weight, mean=0.0, std=0.0005)
        else:
            layers = []
            norms = []
            layers.append(GCNConv(in_channels, hidden_channels))
            norms.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                layers.append(GCNConv(hidden_channels, hidden_channels))
                torch.nn.init.normal_(layers[-1].lin.weight, mean=0.0, std=0.0005)
                norms.append(nn.BatchNorm1d(hidden_channels))
            layers.append(GCNConv(hidden_channels, out_channels))
            self.convs = nn.ModuleList(layers)
            self.norms = nn.ModuleList(norms)

    def forward(self, x, edge_index):
        if self.num_layers == 1:
            x = self.convs[0](x, edge_index)
            return x
        x = self.convs[0](x, edge_index)
        x = self.norms[0](x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(1, self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x




class AgentGraphDatasetTrain(Dataset):
    def __init__(self, root, transform=None, phase: Literal["train", "val"] = "train"):
        super().__init__()
        with open(root, "r") as file:
            raw_dataset = json.load(file)
        origin_dataset_len = len(raw_dataset)
        if phase == "train":
            self.dataset = raw_dataset[:int(origin_dataset_len * 0.8)]
        elif phase == "val":
            self.dataset = raw_dataset[int(origin_dataset_len * 0.8):]
        else:
            raise Exception(f"Unknown phase {phase}")
        # cahced_data_TA_InjecAgent
        self.cacheflag = os.path.exists(PATH_CONFIG[EXPERIMENT]["emb_cache"])
        # self.cacheflag = False
        if self.cacheflag:
            with open(PATH_CONFIG[EXPERIMENT]["emb_cache"], "rb") as f:
                self.cache_dir = pickle.load(f)
                self.embedding_model = None
        else:
            embedding_model_dir = "sentence-transformers/all-MiniLM-L6-v2"
            self.embedding_model = SentenceTransformer(embedding_model_dir)
            self.cache_dir = {}

    def len(self):
        return len(self.dataset)

    def get_keywords(self, idx):
        return [self.cache_dir[i].x_keywords for i in idx]

    def get_tokens(self, idx):
        return [self.cache_dir[i].x_token for i in idx]

    def get(self, idx):
        idx = int(idx)
        if idx in self.cache_dir:
            return idx, self.cache_dir[idx]
        origin_data = self.dataset[idx]
        adj_matrix = origin_data["adj_matrix"]
        adj_matrix_np = np.array(adj_matrix)
        edge_index = adj_matrix_np.nonzero()
        edge_index = np.array(edge_index)
        communication_data = origin_data["communication_data"][0]

        # communication_data = origin_data["communication_data_initial"]
        attacker_idxes = origin_data["attacker_idxes"]
        labels = np.array([1 if i in attacker_idxes else 0 for i in range(len(adj_matrix))])

        summary = origin_data["summary"][0]
        summary_embeddings = self.embedding_model.encode(summary)
        x_summary = torch.tensor(summary_embeddings)

        keywords = origin_data["keywords"]
        keywords_embeddings = torch.Tensor([self.embedding_model.encode(x) for x in keywords])

        embeddings_tokenlevel = self.embedding_model.encode(
            [x[1] for x in communication_data],
            output_value='token_embeddings',
            convert_to_tensor=True
        )

        communication_embeddings = [[] for _ in range(len(adj_matrix))]
        for agent_idx, c_data in communication_data:
            i_turns_agent_idx_embedding = self.embedding_model.encode(c_data)
            communication_embeddings[agent_idx] = i_turns_agent_idx_embedding
        communication_embeddings = np.array(communication_embeddings)
        edge_attr = np.array(communication_embeddings[edge_index[1]], copy=True)

        y = torch.tensor(labels, dtype=torch.long)
        edge_index = torch.tensor(edge_index)
        edge_attr = torch.tensor(edge_attr)
        data = Data(x_summary=x_summary.unsqueeze(0), x_node=torch.tensor(communication_embeddings),
                    x_token=embeddings_tokenlevel,
                    x_keywords=keywords_embeddings, y=y, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = len(adj_matrix_np)
        self.cache_dir[idx] = data
        if not self.cacheflag and len(self.cache_dir) == len(self.dataset):
            with open(PATH_CONFIG[EXPERIMENT]["emb_cache"], "wb") as f:
                pickle.dump(self.cache_dir, f)
                self.cacheflag = True
        return int(idx), data
        # return  data


class AgentGraphDatasetTest(Dataset):
    def __init__(self, root, transform=None, phase: Literal["train", "val"] = "train", adj_key="random"):
        super().__init__()
        with open(root, "r") as file:
            raw_dataset = json.load(file)
        self.dataset = raw_dataset
        self.adj_key = adj_key
        self.cacheflag = os.path.exists(PATH_CONFIG[EXPERIMENT]["emb_cache_test"])
        self.cacheflag = False
        if self.cacheflag:
            with open(PATH_CONFIG[EXPERIMENT]["emb_cache_test"], "rb") as f:
                self.cache_dir = pickle.load(f)
                self.embedding_model = None
        else:
            embedding_model_dir = "sentence-transformers/all-MiniLM-L6-v2"
            self.embedding_model = SentenceTransformer(embedding_model_dir).to("cuda")
            self.cache_dir = {}

    def len(self):
        return len(self.dataset)

    def get_keywords(self, idx):
        return [self.cache_dir[i].x_keywords for i in idx]

    def get_tokens(self, idx):
        return [self.cache_dir[i].x_token for i in idx]

    def get(self, idx):
        if idx in self.cache_dir:
            data = self.cache_dir[idx]
        else:
            origin_data = self.dataset[idx]

            communication_data = origin_data["communication_data_initial"]
            attacker_idxes = origin_data["attacker_idxes"]
            labels = np.array([1 if i in attacker_idxes else 0 for i in range(len(origin_data["A"][self.adj_key]))])

            summary = origin_data["summary"][0]
            summary_embeddings = self.embedding_model.encode(summary)
            x_summary = torch.tensor(summary_embeddings)

            keywords = origin_data["keywords"]
            keywords_embeddings = torch.tensor([self.embedding_model.encode(x) for x in keywords])

            embeddings_tokenlevel = self.embedding_model.encode(
                [x[1] for x in communication_data],
                output_value="token_embeddings",
                convert_to_tensor=True
            )

            communication_embeddings = [[] for _ in range(len(labels))]
            for agent_idx, c_data in communication_data:
                i_turns_agent_idx_embedding = self.embedding_model.encode(c_data)
                communication_embeddings[agent_idx] = i_turns_agent_idx_embedding
            communication_embeddings = np.array(communication_embeddings)

            y = torch.tensor(labels, dtype=torch.long)

            data = Data(
                x_summary=x_summary.unsqueeze(0),
                x_node=torch.tensor(communication_embeddings),
                x_token=embeddings_tokenlevel,
                x_keywords=keywords_embeddings,
                y=y,
            )
            data.num_nodes = len(labels)

            self.cache_dir[idx] = data
            if not self.cacheflag and len(self.cache_dir) == len(self.dataset):
                with open(PATH_CONFIG[EXPERIMENT]["emb_cache_test"], "wb") as f:
                    pickle.dump(self.cache_dir, f)
                self.cacheflag = True

        # 不管缓存与否，这里始终重新生成 edge_index 和 edge_attr
        origin_data = self.dataset[idx]
        adj_matrix = origin_data["A"][self.adj_key]
        adj_matrix_np = np.array(adj_matrix)

        edge_index = np.array(adj_matrix_np.nonzero())
        edge_index = torch.tensor(edge_index)

        communication_embeddings = data.x_node.numpy()
        edge_attr = np.array(communication_embeddings[edge_index[1]], copy=True)
        edge_attr = torch.tensor(edge_attr)

        data.edge_index = edge_index
        data.edge_attr = edge_attr

        return data




def get_score_overall(s1, s2):
    s1 = (s1 - s1.mean()) / torch.std(s1)
    s2 = (s2 - s2.mean()) / torch.std(s2)
    score = s1 + torch.mean(s1 * s2) * s2
    return score



dataset_train = AgentGraphDatasetTrain(root=PATH_CONFIG[EXPERIMENT]["train"])
trainloader = DataLoader(dataset_train, batch_size=8, shuffle=True)



class OursMethod(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.x_proj = GCNEncoder(feat_dim, feat_dim, feat_dim)
        self.gnn = GCNEncoder(feat_dim, feat_dim, feat_dim)
        self.feat_dim = feat_dim

    def encode(self, x_sentance, x_token, edge_index):
        # emb_sentance = self.x_proj(x_sentance) + x_sentance
        emb_sentance = self.x_proj(x_sentance, edge_index) + x_sentance
        # emb_sentance =  x_sentance
        if type(x_token) is list:
            x_token = torch.concatenate(x_token, dim=0)
        emb_token = x_sentance + x_token
        # emb_token = self.gnn(emb_token, edge_index) + emb_token

        # emb_token_nei = self.gnn(emb_token, edge_index) # 这里不加上ego info, 等下用token-level info
        emb_token_nei = self.gnn(emb_token, edge_index) + x_sentance
        return emb_sentance, emb_token_nei

    def forward(self, x_sentance, x_token, x_token_ori, edge_index, batch=None):
        emb_sentance, emb_token_nei = self.encode(x_sentance, x_token, edge_index)
        if batch is None:
            context_sentance = emb_sentance.mean(dim=0)
            emb_token = [x_token_ori[i] + emb_token_nei[i] for i in range(len(emb_token_nei))]
            context_token = torch.stack([t.mean(dim=0) for t in emb_token]).mean(dim=0)
            return emb_sentance, emb_token, context_sentance, context_token
        else:
            num_batches = batch.max().item() + 1
            context_sentance = []
            context_token = []
            emb_token = []
            for i in range(num_batches):
                mask_nodes = (batch == i)
                # idx_mask_nodes = torch.nonzero(mask_nodes, as_tuple=True)[0]
                emb_sentance_i = emb_sentance[mask_nodes]
                emb_token_nei_i = emb_token_nei[mask_nodes]
                # print(idx_mask_nodes)
                # print(len(x_token_ori))
                x_token_ori_i = x_token_ori[i]
                # emb_token_i = [[x_token_ori_i[t][t2] + emb_token_nei_i[t] for t2 in range(len(x_token_ori_i[t]))]
                # for t in range(len(emb_token_nei_i))]
                emb_token_i = [x_token_ori_i[t] + emb_token_nei_i[t] for t in range(len(emb_token_nei_i))]
                context_sentance_i = emb_sentance_i.mean(dim=0)
                # context_token_i =  torch.stack([torch.stack([tt.mean(dim=0) for tt in t]).mean(dim=0) for t in emb_token_i])
                context_token_i = torch.stack([t.mean(dim=0) for t in emb_token_i]).mean(dim=0)
                context_sentance.append(context_sentance_i)
                context_token.append(context_token_i)
                emb_token += emb_token_i
            context_sentance = torch.stack(context_sentance, dim=0)
            context_token = torch.stack(context_token, dim=0)
            return emb_sentance, emb_token, context_sentance, context_token

    def inference_token(self, token_feature, context_token, batch=None):
        if batch is None:
            score_finegrain = [-torch.mm(feature, context_token.unsqueeze(1)) for feature in token_feature]

            # score = torch.stack([-torch.mm(feature, context_token.unsqueeze(1)).mean() for feature in token_feature])
            score = torch.stack([t.mean() for t in score_finegrain])

            return score, score_finegrain
        else:
            num_batches = batch.max().item() + 1
            outputs = []
            outputs_finegrains = []
            for i in range(num_batches):
                mask_nodes = (batch == i)
                idx_mask_nodes = torch.nonzero(mask_nodes, as_tuple=True)[0]
                emb_token_nei_i = [token_feature[t] for t in idx_mask_nodes]
                context_token_i = context_token[i]
                # print(len(emb_token_nei_i[0]))

                score_finegrain_i = [-torch.mm(feature, context_token_i.unsqueeze(1)) for feature in emb_token_nei_i]
                score_i = torch.stack([t.mean() for t in score_finegrain_i])

                outputs.append(score_i)
                outputs_finegrains.append(score_finegrain_i)
            # return torch.stack(outputs, dim=0)   # [num_batches]
            score = torch.stack(outputs, dim=0)
            score_finegrain = outputs_finegrains
            return score, score_finegrain

    def inference(self, feature, context, batch=None):
        if batch is None:
            sim_matrix = torch.mm(feature, context.unsqueeze(1))
            message = -torch.sum(sim_matrix, 1).squeeze()
            return message
        else:
            num_batches = batch.max().item() + 1
            outputs = []
            for i in range(num_batches):
                mask = (batch == i)
                sim = torch.matmul(feature[mask], context[i])  # [Ni]
                outputs.append(-sim)
            return torch.stack(outputs, dim=0)  # [num_batches]


feat_dim = dataset_train[0][1].x_summary.shape[-1]
model_ours = OursMethod(feat_dim).to(device)
optimizer = Adam(list(model_ours.parameters()), lr=lr, weight_decay=weight_decay)
best_loss = float('inf')
best_auc = 0
for epoch in range(epochs):
    total_loss = 0
    num_batches = 0
    model_ours.train()
    for item_idx, data in trainloader:
        optimizer.zero_grad()
        edge_index, edge_attr = data.edge_index.to(device), data.edge_attr.to(device)
        batch = data.batch.to(device)
        x_sentance = data.x_node.to(device)
        x_token = [torch.stack([tt.mean(dim=0).to(device) for tt in t]) for t in
                   trainloader.dataset.get_tokens(item_idx.tolist())]
        x_token_ori = [[tt.to(device) for tt in t] for t in trainloader.dataset.get_tokens(item_idx.tolist())]

        emb_sentance, emb_token, context_sentance, context_token = model_ours.forward(x_sentance, x_token, x_token_ori,
                                                                                      edge_index, batch)

        perm = torch.randperm(context_sentance.size(0))
        context_sentance_neg = context_sentance[perm]
        context_token_neg = context_token[perm]

        score_sentance_pos = model_ours.inference(emb_sentance, context_sentance, batch)
        score_sentance_neg = model_ours.inference(emb_sentance, context_sentance_neg, batch)
        score_token_pos, _ = model_ours.inference_token(emb_token, context_token, batch)
        score_token_neg, _ = model_ours.inference_token(emb_token, context_token_neg, batch)
        label_pos = torch.zeros_like(score_token_pos.view(-1)).float()
        label_neg = torch.ones_like(score_token_neg.view(-1)).float()

        loss_pos = F.binary_cross_entropy_with_logits(
            get_score_overall(score_token_pos, score_sentance_pos).view(-1) / 2, label_pos)
        loss_neg = F.binary_cross_entropy_with_logits(
            get_score_overall(score_token_neg, score_sentance_neg).view(-1) / 2, label_neg)
        loss = loss_pos + alpha * loss_neg
        loss.backward()
        total_loss += loss.item()
        num_batches += 1
        optimizer.step()
    training_loss = total_loss / max(num_batches, 1)
    print(f"Epoch {epoch}/{epochs} || Training Loss: {training_loss:.4f} ")
if save_ckpt:
    filename_pkl = f'{config["experiment"]}_seed{config["seed"]}_alpha{config["alpha"]}_lr{config["lr"]}.pkl'
    torch.save(model_ours.state_dict(), f"./ckpt/{filename_pkl}")

result_gs = {

}
for adj_key in ["random", "chain", "tree", "star"]:
    dataset_test = AgentGraphDatasetTest(root=PATH_CONFIG[EXPERIMENT]["test"], adj_key=adj_key)
    testloader = DataLoader(dataset_test, batch_size=1, shuffle=False)
    cum_auc = 0
    cum_prc = 0
    model_ours.eval()
    for data in testloader:
        edge_index, edge_attr = data.edge_index.to(device), data.edge_attr.to(device)
        y = data.y
        x_sentance = data.x_node.to(device)
        x_token = torch.stack([t.mean(dim=0).to(device) for t in data.x_token])
        x_token_ori = [t.to(device) for t in data.x_token]

        emb_sentance, emb_token, context_sentance, context_token = model_ours.forward(x_sentance, x_token, x_token_ori,
                                                                                      edge_index)
        score_sentance = model_ours.inference(emb_sentance, context_sentance)
        score_token, score_finegrain = model_ours.inference_token(emb_token, context_token)
        score = get_score_overall(score_sentance, score_token)
        cum_auc += roc_auc_score(y.detach().cpu().numpy(), score.detach().cpu().numpy())
        cum_prc += average_precision_score(y.detach().cpu().numpy(), score.detach().cpu().numpy())
    test_auc = cum_auc / max(len(testloader), 1)
    test_prc = cum_prc / max(len(testloader), 1)

    result_gs[adj_key] = {
        "roauc": test_auc,
        "roprc": test_prc
    }


result_dict = {
    "config": config,
    "results": result_gs
}
print(result_dict)


if save_results:
    out_dir = f'./results/gridsearch/{config["experiment"]}/'
    os.makedirs(out_dir, exist_ok=True)
    # Build the filename
    filename = f'{config["experiment"]}_seed{config["seed"]}_alpha{config["alpha"]}_lr{config["lr"]}.json'
    out_path = os.path.join(out_dir, filename)
    # Save result_dict to JSON
    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"Results saved to {out_path}")


