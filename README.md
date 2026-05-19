## ACL26 XG-Guard

This is the official implementation of the following paper: Explainable and Fine-Grained Safeguarding of LLM Multi-Agent Systems via Bi-Level Graph Anomaly Detection, Accepted by ACL 2026 (Main)

Note: The skeleton of this project is inherited from GSafeguard and BlindGuard, but has been refactored to better support future researchers. If the refactored code doesn't work, please raise an issue, or you can push fixs. 



## Environment Setup

```
# Core Libraries
openai==1.58.1  
langgraph==1.0.4  
langchain==1.1.0  

# PyTorch Stack
torch==2.5.1  
torchvision==0.20.1  
torchaudio==2.5.1  
accelerate==1.12.0  
einops==0.8.1  

# Graph Learning
torch_geometric==2.6.1  
torch_scatter==2.1.2  
torch_sparse==0.6.18  
torch_cluster==1.6.3  
torch_spline_conv==1.2.2  
networkx==3.4.2  

# Language Models
transformers==4.44.2  
sentence_transformers==3.3.1  

# Utilities
numpy==1.26.4  
scipy==1.15.3  
pandas==2.2.3  
scikit_learn==1.6.1  
pydantic==2.10.4  
pydantic_settings==2.7.1  
python_dotenv==1.1.0  
requests==2.32.3  
tqdm==4.67.1  
```

## Dataset Preparation

### datasets_local

`datasets_local/` contains datasets used for pre-training. Which only contain normal agent dialogue. We also include dummy test datasets, which include GPT4oMini generated agents output, to help you evaluate your model faster during development. 

```
datasets_local/MA_CSQA_local_train_dataset.json
datasets_local/MA_CSQA_local_test_dataset.json
```

Paths are configured in `load_data/dataset_paths.py` and used by `Ours.py`.

### datasets_online

`datasets_online/` is what you should use for evaluation and reporting performance. It contains necessary information to set up LLM agents and attackers. 

- `datasets_online/MA/agent_graph_dataset/memory_attack/`
- `datasets_online/MA-CSQA/agent_graph_dataset/memory_attack/`
- `datasets_online/PI/agent_graph_dataset/{csqa,gsm8k,mmlu}/`
- `datasets_online/TA/agent_graph_dataset/tool_attack/`

Use `--atk_type` and `--expr_type` to select the experiment; default online test paths are resolved automatically.

Experiments:

| atk_type | expr_type   |
|----------|-------------|
| MA       | PoisonRAG   |
| MA       | CSQA        |
| PI       | CSQA        |
| PI       | MMLU        |
| PI       | GSM8K       |
| TA       | InjecAgent  |

## Training (XG-Guard)

Since the code encodes text using Sentence BERT, we provide an embedding cache to accelerate training.

After embeddings are computed once, you can enable cache loading in subsequent runs.  
Specifically, set `cacheflag=True` after line 311 in `Ours.py` to reuse cached embeddings.

```bash
python Ours.py --experiment MA-PoisonRAG
```

## Testing (online, attack mode)

Run unified online defense evaluation:

```bash
python main_defense_for_different_topology1.py \
  --atk_type MA \
  --expr_type PoisonRAG \
  --defend_type Ours \
  --gnn_checkpoint_path ckpt/MA-PoisonRAG_seed3701_alpha0.0001_lr1e-05.pkl \
  --model_type gpt-4o-mini
```

Configure `OPENAI_API_KEY` and `BASE_URL` for your LLM API.

Run all graph topologies via:

```bash
python RUN_ASR_EVAL.py --atk_type MA --expr_type PoisonRAG --save_dir ./result
```

Agent implementations live under `modules/` and are selected by attack type:

- `modules/agents_ma.py` for MA
- `modules/agents_pi.py` / `modules/agents_pi_gsm8k.py` for PI
- `modules/agents_ta.py` for TA

## Evaluation

```bash
python evaluate_output.py \
  --atk_type MA \
  --expr_type PoisonRAG \
  --result_dir ./result
```

For GSafeguard outputs, set `--model G-safeguard`. Metrics (AUROC / ASR) depend on `--atk_type` and `--expr_type`.

## Running Baselines

We also provide the instruction here to help you run baselines easier.  All baselines share the same online pipeline as XG-Guard (`main_defense_for_different_topology1.py` or `RUN_ASR_EVAL.py`), but use different `defend_type` values and checkpoints. Train unsupervised baselines first, then run online evaluation from the repository root.

### BlindGuard (SCL)

BlindGuard corresponds to the self-supervised contrastive model (`GATSCL`, `defend_type=SCL`).

Train (unsupervised):

```bash
python -m modules.train_un1 \
  --defend_type SCL \
  --dataset_path ./ModelTrainingSet/memory_attack/dataset.pkl \
  --epochs 50 \
  --save_dir ./checkpoint
```

Online test:

```bash
python main_defense_for_different_topology1.py \
  --atk_type MA \
  --expr_type PoisonRAG \
  --defend_type SCL \
  --gnn_checkpoint_path <path_to_SCL_checkpoint.pth> \
  --model_type gpt-4o-mini
```

Batch over topologies:

```bash
python RUN_ASR_EVAL.py --atk_type MA --expr_type PoisonRAG --defend_type SCL --save_dir ./result
```

Set matching SCL checkpoint paths in `PATH_CONFIG` inside `RUN_ASR_EVAL.py`.

Other unsupervised variants in the same trainer: `--defend_type TAM` or `--defend_type Dominant` via `modules/train_un1.py`.

### PREM-GAD

Train:

```bash
python -m modules.train_un2 \
  --defend_type PREM \
  --dataset_path ./ModelTrainingSet/memory_attack/dataset.pkl \
  --prem_k 2 \
  --epochs 50 \
  --save_dir ./checkpoint
```

Online test:

```bash
python main_defense_for_different_topology1.py \
  --atk_type MA \
  --expr_type PoisonRAG \
  --defend_type PREM \
  --prem_k 2 \
  --gnn_checkpoint_path <path_to_PREM_checkpoint.pth> \
  --model_type gpt-4o-mini
```

### GSafeguard

GSafeguard uses the supervised GAT (`MyGAT`) and a separate entry script.

Train (supervised):

```bash
python -m modules.train \
  --dataset_path ./ModelTrainingSet/memory_attack/dataset1.pkl \
  --save_dir ./checkpoint
```

Online test:

```bash
python main_defense_for_different_topology.py \
  --atk_type MA \
  --expr_type PoisonRAG \
  --gnn_checkpoint_path <path_to_Gsafeguard_GAT_checkpoint.pth> \
  --model_type gpt-4o-mini \
  --topk 3
```

Outputs are saved with `defense_type_Gsafe` in the filename. Evaluate with:

```bash
python evaluate_output.py \
  --atk_type MA \
  --expr_type PoisonRAG \
  --result_dir ./result \
  --model G-safeguard
```

### Citation

If you find our repository useful for your research, please consider citing our paper:

```
@inproceedings{pan2026explainable,
  title={Explainable and Fine-Grained Safeguarding of LLM Multi-Agent Systems via Bi-Level Graph Anomaly Detection},
  author={Pan, Junjun and Liu, Yixin and Miao, Rui and Ding, Kaize and Zheng, Yu and Nguyen, Quoc Viet Hung and Liew, Alan Wee-Chung and Pan, Shirui},
  journal={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year={2026}
}
```
