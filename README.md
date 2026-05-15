## ACL26 XG-Guard

This is the official implementation of the following paper: Explainable and Fine-Grained Safeguarding of LLM Multi-Agent Systems via Bi-Level Graph Anomaly Detection, Accepted by ACL 2026 (Main)






## TODO
The repo structure is inherited from BlindGuard and GSafeguard. Will refactor the code to make it easier to run and develop future methods. 

## Environment Setup

Please install the following dependencies with the specified versions.

### Core Libraries
openai==1.58.1  
langgraph==1.0.4  
langchain==1.1.0  

### PyTorch Stack
torch==2.5.1  
torchvision==0.20.1  
torchaudio==2.5.1  
accelerate==1.12.0  
einops==0.8.1  

### Graph Learning
torch_geometric==2.6.1  
torch_scatter==2.1.2  
torch_sparse==0.6.18  
torch_cluster==1.6.3  
torch_spline_conv==1.2.2  
networkx==3.4.2  

### Language Models
transformers==4.44.2  
sentence_transformers==3.3.1  

### Utilities
numpy==1.26.4  
scipy==1.15.3  
pandas==2.2.3  
scikit_learn==1.6.1  
pydantic==2.10.4  
pydantic_settings==2.7.1  
python_dotenv==1.1.0  
requests==2.32.3  
tqdm==4.67.1  

## Dataset Preparation

The complete dataset will be released after publication.

For ease of evaluation, we provide local training and testing datasets.

1. Local datasets  
./MA_CSQA_local_train_dataset.json  
./MA_CSQA_local_test_dataset.json  

2. Agent based evaluation  
For evaluations that involve API based LLM agents, datasets are stored at  
./agent_graph_dataset/memory_attack/test/dataset.json  

3. Baseline training data  
The following directories are used only for training baseline methods  
./agent_graph_dataset/memory_attack/train/dataset.json  
./agent_graph_dataset/memory_attack/train1/dataset.json  

## Training

Since the code encodes text using Sentence BERT, we provide an embedding cache to accelerate training.

After embeddings are computed once, you can enable cache loading in subsequent runs.  
Specifically, set cacheflag=True after line 311 to reuse cached embeddings.

To train XG Guard, run  
./MA/Ours.py  

## Testing

To evaluate the defense method, run main_defense_for_different_topology1.py  and select "Ours" as the model. 

This script uses the test set and communicates with the LLM for evaluation.  
You must configure the LLM model_type and provide the appropriate API URL and key.

Note  
main_defense_for_different_topology.py corresponds to GSafeguard.

## Evaluation

Run  
evaluate_output.py  

This script computes the corresponding AUC and ASR metrics.

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

