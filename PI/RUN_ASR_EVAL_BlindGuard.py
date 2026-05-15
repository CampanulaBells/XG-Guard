import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--evaluator", type=str, default="main_defense_for_different_topology1.py")
parser.add_argument("--atk_type", type=str, default="MA")
parser.add_argument("--expr_type", type=str, default="PoisonRAG")
parser.add_argument("--model_type", type=str, default="gpt-4o-mini")
parser.add_argument("--save_dir", type=str, default="G:/AgentGAD/BlindGuard/MA/results/ASR_BlindGuard/")
args = parser.parse_args()

EXPR_NAME = f"{args.atk_type}-{args.expr_type}"

GRAPH_TYPES = ["random", "chain", "tree", "star"]
DEFEND_TYPE = "SCL"

PATH_CONFIG = {
    # "MA-PoisonRAG": {
    #     "dataset_path": "G:/AgentGAD/BlindGuard\datasets/MA/agent_graph_dataset/memory_attack/test/dataset.json",
    #     "gnn_checkpoint_path": "G:/AgentGAD/BlindGuard/MA/ckpt/MA-PoisonRAG_seed3701_alpha0.0001_lr1e-05.pkl",
    # },
    # "MA-CSQA": {
    #     "dataset_path": "G:/AgentGAD/BlindGuard\datasets/MA-CSQA/agent_graph_dataset/memory_attack/test/dataset.json",
    #     "gnn_checkpoint_path": "G:/AgentGAD/BlindGuard/MA/ckpt/MA-CSQA_seed3701_alpha1e-05_lr5e-05.pkl",
    # },
    # "TA-InjecAgent": {
    #     "dataset_path": "G:/AgentGAD/BlindGuard/datasets/TA/agent_graph_dataset/tool_attack/test/dataset.json",
    #     "gnn_checkpoint_path": "G:/AgentGAD/BlindGuard/MA/ckpt/TA-InjecAgent_seed3701_alpha0.0001_lr0.0001.pkl",
    # },
    "PI-CSQA": {
        "dataset_path": "G:/AgentGAD/BlindGuard/datasets/PI/agent_graph_dataset/csqa/test/dataset.json",
        "gnn_checkpoint_path": "G:/AgentGAD/BlindGuard/PI/checkpoint_un2/csqa/20250924_110747-defend_type_SCL-hiddim_1024-latent_512-heads_8-layers_2-epochs_20-lr_0.001-rep_type_0.pth",
    },
    "PI-GSM8K": {
        "dataset_path": "G:/AgentGAD/BlindGuard/datasets/PI/agent_graph_dataset/gsm8k/test/dataset.json",
        "gnn_checkpoint_path": "G:/AgentGAD/BlindGuard/PI/checkpoint_un2/gsm8k/20250924_110708-defend_type_SCL-hiddim_1024-latent_512-heads_8-layers_2-epochs_20-lr_0.001-rep_type_0.pth",
    },
    "PI-MMLU": {
        "dataset_path": "G:/AgentGAD/BlindGuard/datasets/PI/agent_graph_dataset/mmlu/test/dataset.json",
        "gnn_checkpoint_path": "G:/AgentGAD/BlindGuard/PI/checkpoint_un2/mmlu/20250924_085514-defend_type_SCL-hiddim_1024-latent_512-heads_8-layers_2-epochs_20-lr_0.001-rep_type_0.pth",
    },
}
assert EXPR_NAME in PATH_CONFIG
config_dict = PATH_CONFIG[EXPR_NAME]

for graph_type in GRAPH_TYPES:
    cmd = [
        "python",
        str(args.evaluator),
        "--atk_type", str(args.atk_type),
        "--expr_type", str(args.expr_type),
        "--model_type", str(args.model_type),
        "--save_dir", str(args.save_dir),
        "--defend_type", str(DEFEND_TYPE),
        "--graph_type", graph_type,
        "--dataset_path", str(config_dict["dataset_path"]),
        "--gnn_checkpoint_path", str(config_dict["gnn_checkpoint_path"]),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
