import argparse
import subprocess
from load_data.dataset_paths import get_online_test_path

parser = argparse.ArgumentParser()
parser.add_argument("--evaluator", type=str, default="main_defense_for_different_topology1.py")
parser.add_argument("--atk_type", type=str, default="MA")
parser.add_argument("--expr_type", type=str, default="PoisonRAG")
parser.add_argument("--model_type", type=str, default="gpt-4o-mini")
parser.add_argument("--save_dir", type=str, default="./result")
parser.add_argument("--defend_type", type=str, default="Ours")
args = parser.parse_args()

EXPR_NAME = f"{args.atk_type}-{args.expr_type}"
GRAPH_TYPES = ["random", "chain", "tree", "star"]

PATH_CONFIG = {
    "MA-PoisonRAG": {
        "gnn_checkpoint_path": "ckpt/MA-PoisonRAG_seed3701_alpha0.0001_lr1e-05.pkl",
    },
    "MA-CSQA": {
        "gnn_checkpoint_path": "ckpt/MA-CSQA_seed3701_alpha1e-05_lr5e-05.pkl",
    },
    "TA-InjecAgent": {
        "gnn_checkpoint_path": "ckpt/TA-InjecAgent_seed3701_alpha0.0001_lr0.0001.pkl",
    },
    "PI-CSQA": {
        "gnn_checkpoint_path": "ckpt/PI-CSQA_seed3701_alpha0.0001_lr1e-05.pkl",
    },
    "PI-GSM8K": {
        "gnn_checkpoint_path": "ckpt/PI-GSM8K_seed3701_alpha0.0001_lr5e-05.pkl",
    },
    "PI-MMLU": {
        "gnn_checkpoint_path": "ckpt/PI-MMLU_seed3701_alpha0.0001_lr0.0001.pkl",
    },
}
assert EXPR_NAME in PATH_CONFIG
config_dict = PATH_CONFIG[EXPR_NAME]
dataset_path = get_online_test_path(args.atk_type, args.expr_type)

for graph_type in GRAPH_TYPES:
    cmd = [
        "python",
        str(args.evaluator),
        "--atk_type",
        str(args.atk_type),
        "--expr_type",
        str(args.expr_type),
        "--model_type",
        str(args.model_type),
        "--save_dir",
        str(args.save_dir),
        "--graph_type",
        graph_type,
        "--dataset_path",
        dataset_path,
        "--gnn_checkpoint_path",
        str(config_dict["gnn_checkpoint_path"]),
        "--defend_type",
        args.defend_type,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
