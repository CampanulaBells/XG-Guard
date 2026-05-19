import os
from tqdm import tqdm
import json
import random
import numpy as np
import torch
from einops import rearrange
from torch_scatter import scatter_mean
import argparse
from datetime import datetime
import asyncio
import copy
import time
from utils import get_adj_matrix
from load_data.dataset_paths import get_online_test_path

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

from modules.Dominant import GCNModelAE
import torch.nn.functional as F
from modules.TAM import TAMModel, GATSCL
from modules.Prem_gad import PREMModel


def get_agent_graph_with_defense(atk_type, expr_type):
    if atk_type == "MA":
        from modules.agents_ma import AgentGraphWithDefense
    elif atk_type == "TA":
        from modules.agents_ta import AgentGraphWithDefense
    elif atk_type == "PI":
        if expr_type == "GSM8K":
            from modules.agents_pi_gsm8k import AgentGraphWithDefense
        else:
            from modules.agents_pi import AgentGraphWithDefense
    else:
        raise Exception(f"Unknown atk_type: {atk_type}")
    return AgentGraphWithDefense


def prepare_case(d, atk_type):
    if atk_type == "MA":
        return d["query"], d["adv_texts"]
    if atk_type == "PI":
        qa_data_origin = d["question"], d["correct_answer"], d["wrong_answer"]
        wrong_answer = random.choice(qa_data_origin[2]) if qa_data_origin[2] else None
        return (qa_data_origin[0], qa_data_origin[1], wrong_answer)
    if atk_type == "TA":
        return (
            d["user_instruction"],
            d["available_tool_names"],
            d["available_tool_descriptions"],
            d["agent_scratchpad"],
        )
    raise Exception(f"Unknown atk_type: {atk_type}")


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


class OursMethod(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.x_proj = GCNEncoder(feat_dim, feat_dim, feat_dim)
        self.gnn = GCNEncoder(feat_dim, feat_dim, feat_dim)
        self.feat_dim = feat_dim

    def encode(self, x_sentance, x_token, edge_index):
        emb_sentance = self.x_proj(x_sentance, edge_index) + x_sentance
        if type(x_token) is list:
            x_token = torch.concatenate(x_token, dim=0)
        emb_token = x_sentance + x_token
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
                emb_sentance_i = emb_sentance[mask_nodes]
                emb_token_nei_i = emb_token_nei[mask_nodes]
                x_token_ori_i = x_token_ori[i]
                emb_token_i = [x_token_ori_i[t] + emb_token_nei_i[t] for t in range(len(emb_token_nei_i))]
                context_sentance_i = emb_sentance_i.mean(dim=0)
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
                score_finegrain_i = [-torch.mm(feature, context_token_i.unsqueeze(1)) for feature in emb_token_nei_i]
                score_i = torch.stack([t.mean() for t in score_finegrain_i])
                outputs.append(score_i)
                outputs_finegrains.append(score_finegrain_i)
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
                sim = torch.matmul(feature[mask], context[i])
                outputs.append(-sim)
            return torch.stack(outputs, dim=0)


def get_score_overall(s1, s2):
    s1 = (s1 - s1.mean()) / torch.std(s1)
    s2 = (s2 - s2.mean()) / torch.std(s2)
    score = s1 + torch.mean(s1 * s2) * s2
    return score


from sentence_transformers import SentenceTransformer

embedding_model_dir = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_dir)


def response2embeddings(responses):
    embeddings = [None for _ in range(len(responses))]
    embeddings_tokenlevel = [None for _ in range(len(responses))]
    for agent_idx, agent_response in responses:
        embeddings[agent_idx] = embedding_model.encode(agent_response)
        embeddings_tokenlevel[agent_idx] = embedding_model.encode(
            agent_response,
            output_value="token_embeddings",
            convert_to_tensor=True,
        ).to("cpu")
    embeddings = np.array(embeddings)
    return embeddings, embeddings_tokenlevel


def embeddings2graph(embeddings, adj_matrix):
    edge_index = torch.tensor(np.array(adj_matrix.nonzero()))
    edge_attr = torch.tensor(np.array(embeddings))[:, edge_index[1]]
    x = edge_attr[0, :]
    x = scatter_mean(x, edge_index[1], dim=0, dim_size=len(embeddings[0]))
    edge_attr = edge_attr.transpose(0, 1)
    edge_attr_expanded = edge_attr.reshape(edge_attr.size(0), -1)
    edge_attr_expanded = torch.nn.functional.pad(
        edge_attr_expanded,
        (0, 1536 - edge_attr_expanded.size(1)),
        mode="replicate",
    )
    return x, edge_index, edge_attr_expanded


def run_ours_inference(gnn, embeddings, embeddings_tokenlevel, edge_index):
    x_sentance = torch.tensor(embeddings)
    x_token = torch.stack([t.mean(dim=0) for t in embeddings_tokenlevel])
    x_token_ori = embeddings_tokenlevel
    emb_sentance, emb_token, context_sentance, context_token = gnn.forward(
        x_sentance, x_token, x_token_ori, edge_index
    )
    score_sentance = gnn.inference(emb_sentance, context_sentance)
    score_token, score_finegrain = gnn.inference_token(emb_token, context_token)
    return get_score_overall(score_sentance, score_token)


def dominant_scores(gnn, x, edge_index, atk_type):
    x_recon, adj_recon, z = gnn(x, edge_index)
    num_nodes = x.size(0)
    adj = torch.eye(num_nodes)
    adj[edge_index[0], edge_index[1]] = 1.0
    attr_errors = torch.mean((x - x_recon) ** 2, dim=1)
    struct_errors = torch.mean((adj - adj_recon) ** 2, dim=1)
    if atk_type == "TA":
        return 0.5 * attr_errors + 0.5 * struct_errors
    return 0.8 * attr_errors + 0.2 * struct_errors


async def defense_communication(
    ag,
    gnn,
    case_data,
    adj_m,
    num_dialogue_turns,
    defend_type,
    topk,
    atk_type,
):
    communication_data = []
    identified_attackers = []
    response_embeddings = []
    initial_responses = await ag.afirst_generate(case_data)
    embeddings, embeddings_tokenlevel = response2embeddings(initial_responses)
    response_embeddings.append(embeddings)
    x, edge_index, edge_attr = embeddings2graph(response_embeddings, adj_m)

    if defend_type == "TAM":
        z, feat1, feat2 = gnn(x, edge_index)
        num_nodes = x.size(0)
        adj = torch.eye(num_nodes)
        adj[edge_index[0], edge_index[1]] = 1.0
        message = gnn.inference(z, adj)
        _, predicts = torch.topk(-message, topk)
    elif defend_type == "SCL":
        z = gnn.encode(x, edge_index)
        num_nodes = x.size(0)
        adj = torch.eye(num_nodes)
        adj[edge_index[0], edge_index[1]] = 1.0
        message = gnn.inference_new(z, adj)
        _, predicts = torch.topk(-message, topk)
    elif defend_type == "Dominant":
        anomaly_scores = dominant_scores(gnn, x, edge_index, atk_type)
        _, predicts = torch.topk(anomaly_scores, topk)
    elif defend_type == "PREM":
        anomaly_scores = gnn.get_anomaly_scores(x, edge_index)
        _, predicts = torch.topk(anomaly_scores.squeeze(), topk)
    elif defend_type in ["XGGuard", "Ours"]:
        anomaly_scores = run_ours_inference(gnn, embeddings, embeddings_tokenlevel, edge_index)
        value, predicts = torch.topk(anomaly_scores, topk)
    else:
        raise Exception(f"Unknown defend_type: {defend_type}")

    original_predicts = predicts.tolist()
    for predict in predicts:
        ag.agents[predict].set_role("attacker")
    communication_data.append(initial_responses)
    identified = []
    for _ in range(num_dialogue_turns):
        responses = await ag.are_generate()
        embeddings, embeddings_tokenlevel = response2embeddings(responses)
        response_embeddings.append(embeddings)
        x, edge_index, edge_attr = embeddings2graph(response_embeddings, adj_m)

        for idx in range(len(ag.agents)):
            ag.agents[idx].set_role("normal")

        if defend_type == "TAM":
            z, feat1, feat2 = gnn(x, edge_index)
            num_nodes = x.size(0)
            adj = torch.eye(num_nodes)
            adj[edge_index[0], edge_index[1]] = 1.0
            message = gnn.inference(z, adj)
            _, predicts = torch.topk(-message, topk)
        elif defend_type == "SCL":
            predicts = original_predicts
        elif defend_type == "Dominant":
            anomaly_scores = dominant_scores(gnn, x, edge_index, atk_type)
            _, predicts = torch.topk(anomaly_scores, topk)
        elif defend_type == "PREM":
            anomaly_scores = gnn.get_anomaly_scores(x, edge_index)
            _, predicts = torch.topk(anomaly_scores.squeeze(), topk)
        elif defend_type in ["Ours", "XGGuard"]:
            predicts = original_predicts

        for predict in predicts:
            ag.agents[predict].set_role("attacker")
            if atk_type == "MA" and predict not in identified:
                identified.append(predict)
        communication_data.append(responses)
        if atk_type == "MA":
            identified_attackers.append(copy.deepcopy(identified))

    return communication_data, original_predicts


def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiments to train GAT")
    parser.add_argument("--atk_type", type=str, default="MA", choices=["MA", "PI", "TA"])
    parser.add_argument("--expr_type", type=str, default="PoisonRAG")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Save path of the dataset",
    )
    parser.add_argument("--graph_type", type=str, choices=["random", "chain", "tree", "star"], default=None)
    parser.add_argument(
        "--gnn_checkpoint_path",
        default="ckpt/MA-PoisonRAG_seed3701_alpha0.0001_lr1e-05.pkl",
        type=str,
    )
    parser.add_argument("--save_dir", type=str, default="./result")
    parser.add_argument("--model_type", type=str, default="gpt-4o-mini")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--defend_type",
        type=str,
        default="Ours",
        choices=["SCL", "TAM", "Dominant", "PREM", "Ours", "XGGuard"],
    )
    parser.add_argument("--rep_type", type=int, default=0)
    parser.add_argument("--prem_k", type=int, default=2, help="PREM aggregation steps")

    args = parser.parse_args()
    if args.dataset_path is None:
        args.dataset_path = get_online_test_path(args.atk_type, args.expr_type)

    if args.graph_type is None:
        args.graph_type = "star" if args.atk_type == "MA" else "random"

    args.save_dir = os.path.join(args.save_dir, f"{args.atk_type}-{args.expr_type}", args.graph_type)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_defense = (
        f"{current_time_str}-defense_type_{args.defend_type}-topk_{args.topk}"
        f"-model_type_{args.model_type}-rep_type_{args.rep_type}.json"
    )
    args.save_path_with_defense = os.path.join(args.save_dir, filename_defense)
    return args


async def main():
    args = parse_arguments()
    AgentGraphWithDefense = get_agent_graph_with_defense(args.atk_type, args.expr_type)
    filepath = args.dataset_path
    graph_type = args.graph_type
    with open(filepath, "r") as f:
        dataset = json.load(f)
    dataset = dataset[-args.samples :]
    num_dialogue_turns = len(dataset[0]["communication_data"]) - 1

    if args.defend_type in ["TAM"]:
        gnn = TAMModel(
            in_channels=384,
            hidden_channels=1024,
            out_channels=512,
            dropout=0,
            readout="avg",
        )
    elif args.defend_type == "Dominant":
        gnn = GCNModelAE(
            in_channels=384,
            hidden_channels=1024,
            latent_channels=512,
            dropout=0.0,
        )
    elif args.defend_type == "SCL":
        gnn = GATSCL(
            in_channels=384,
            hidden_channels=1024,
            out_channels=512,
            type=args.rep_type,
        )
    elif args.defend_type == "PREM":
        gnn = PREMModel(n_in=384, n_hidden=1024, k=args.prem_k)
    elif args.defend_type in ["Ours", "XGGuard"]:
        gnn = OursMethod(384)

    checkpoint = torch.load(args.gnn_checkpoint_path, map_location=torch.device("cpu"))
    gnn.load_state_dict(checkpoint)

    final_dataset_wd = []
    for d in tqdm(dataset):
        if graph_type == "random":
            adj_m = np.array(d["adj_matrix"])
        elif graph_type in ["chain", "tree", "star"]:
            adj_m = get_adj_matrix(graph_type, len(d["adj_matrix"]))
        else:
            raise Exception(
                f"Unknown graph type: {graph_type}! Can only be one of [random, chain, tree, star]"
            )
        attacker_idxes = d["attacker_idxes"]
        system_prompts = d["system_prompts"]
        case_data = prepare_case(d, args.atk_type)
        agwd = AgentGraphWithDefense(adj_m, system_prompts, attacker_idxes, model_type=args.model_type)
        try:
            communication_data_defense, original_predicts = await defense_communication(
                agwd,
                gnn,
                case_data,
                adj_m,
                num_dialogue_turns,
                args.defend_type,
                args.topk,
                args.atk_type,
            )
        except Exception as e:
            print(e)
            continue

        d_wd = copy.deepcopy(d)
        d_wd["communication_data"] = communication_data_defense
        d_wd["original_predicts"] = original_predicts
        final_dataset_wd.append(d_wd)

    with open(args.save_path_with_defense, "w") as file:
        json.dump(final_dataset_wd, file, indent=None)


if __name__ == "__main__":
    asyncio.run(main())
