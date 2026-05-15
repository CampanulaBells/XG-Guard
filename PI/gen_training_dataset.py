import os 
import json
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def gen_model_training_set(language_dataset, embedding_model, save_path): 
    dataset = []
    for meta_data in tqdm(language_dataset, desc="Generate training data"): 
        try:
            adj_matrix = meta_data.get("adj_matrix")
            attacker_idxes = meta_data.get("attacker_idxes", [])
            system_prompts = meta_data.get("system_prompts", [])
            question = meta_data.get("question", "")
            correct_answer = meta_data.get("correct_answer", "")
            wrong_answer = meta_data.get("wrong_answer", "")
            communication_data = meta_data.get("communication_data", [])
            attacker_idxes = meta_data["attacker_idxes"]
            if not all([isinstance(adj_matrix, list), 
                       isinstance(system_prompts, list), 
                       isinstance(communication_data, list)]):
                print("Warning: Invalid data types in entry")
                continue
            
            if not all([adj_matrix, system_prompts, communication_data]):
                print("Warning: Empty required fields in data entry")
                continue
            
            adj_matrix_np = np.array(adj_matrix)
            if adj_matrix_np.ndim != 2 or adj_matrix_np.shape[0] != adj_matrix_np.shape[1]:
                print("Warning: Invalid adjacency matrix shape")
                continue
                
            labels = np.array([1 if i in attacker_idxes else 0 for i in range(len(adj_matrix))])
            
            system_prompts_embedding = []
            for prompt in system_prompts:
                if isinstance(prompt, str) and prompt.strip():
                    try:
                        embedding = embedding_model.encode(prompt)
                        system_prompts_embedding.append(embedding)
                    except Exception as e:
                        print(f"Warning: Failed to encode system prompt: {str(e)}")
                        system_prompts_embedding.append(np.zeros(384))
                else:
                    print("Warning: Invalid system prompt")
                    system_prompts_embedding.append(np.zeros(384))
            system_prompts_embedding = np.array(system_prompts_embedding)
            edge_index = adj_matrix_np.nonzero()
            edge_index = np.array(edge_index)
            num_nodes = len(adj_matrix)
            communication_embeddings = [[] for _ in range(num_nodes)]
            
            for turn_idx, turn_data in enumerate(communication_data):
                if not isinstance(turn_data, list):
                    print(f"Warning: Invalid turn data at turn {turn_idx}")
                    continue
                
                turn_embeddings = [None] * num_nodes
                
                for item in turn_data:
                    if not isinstance(item, (list, tuple)) or len(item) != 2:
                        print(f"Warning: Invalid communication item format at turn {turn_idx}")
                        continue
                        
                    agent_idx, message = item
                    
                    if not isinstance(agent_idx, int) or agent_idx >= num_nodes:
                        print(f"Warning: Invalid agent index {agent_idx} at turn {turn_idx}")
                        continue
                        
                    if message is not None and isinstance(message, str) and message.strip():
                        try:
                            embedding = embedding_model.encode(message)
                            turn_embeddings[agent_idx] = embedding
                        except Exception as e:
                            print(f"Warning: Failed to encode message from agent {agent_idx} at turn {turn_idx}: {str(e)}")
                            turn_embeddings[agent_idx] = np.zeros(384)
                    else:
                        turn_embeddings[agent_idx] = np.zeros(384)
                for agent_idx in range(num_nodes):
                    embedding = turn_embeddings[agent_idx]
                    if embedding is None:
                        embedding = np.zeros(384)
                    communication_embeddings[agent_idx].append(embedding)
            
            max_turns = max(len(agent_emb) for agent_emb in communication_embeddings)
            for agent_idx in range(num_nodes):
                while len(communication_embeddings[agent_idx]) < max_turns:
                    communication_embeddings[agent_idx].append(np.zeros(384))
            
            communication_embeddings = np.array(communication_embeddings)
            
            if edge_index[1].size > 0:
                edge_attr = np.array(communication_embeddings[edge_index[1]], copy=True)
            else:
                print("Warning: No edges found in the graph")
                continue
            
            data = {
                "adj_matrix": adj_matrix_np,
                "features": system_prompts_embedding,
                "labels": labels,    
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "attacker_idxes" : attacker_idxes
            }
            
            if all(arr is not None and arr.size > 0 for arr in [adj_matrix_np, system_prompts_embedding, edge_attr]):
                dataset.append(data)
            else:
                print("Warning: Invalid array shapes detected, skipping this entry")
                
        except Exception as e:
            print(f"Error processing data entry: {str(e)}")
            continue
        
    if not dataset:
        raise ValueError("No valid data entries were processed")
        
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    return len(dataset)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiments that generate dataset")
    parser.add_argument("--dataset", type=str, default="mmlu", choices=["mmlu", "csqa", "gsm8k"])
    parser.add_argument("--type", type=str, default="unsuper", choices=["super", "unsuper", "test"])
    args = parser.parse_args()

    if args.type == "super":
        data_dir = "./agent_graph_dataset/{}/train1/dataset.json".format(args.dataset)
    elif args.type == "unsuper":
        data_dir = "./agent_graph_dataset/{}/train/dataset.json".format(args.dataset)
    elif args.type == "test":
        data_dir = "./agent_graph_dataset/{}/test/dataset.json".format(args.dataset)

    save_dir = f"./ModelTrainingSet/{args.dataset}"
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    
    if args.type == "super":
        save_path = os.path.join(save_dir, "dataset1.pkl")
    elif args.type == "unsuper":
        save_path = os.path.join(save_dir, "dataset.pkl")
    elif args.type == "test":
        save_path = os.path.join(save_dir, "test_dataset.pkl")
    
    embedding_model_dir = "sentence-transformers/all-MiniLM-L6-v2"
    
    try:
        embedding_model = SentenceTransformer(embedding_model_dir)
        with open(data_dir, 'r') as file:
            language_dataset = json.load(file)
        
        if not isinstance(language_dataset, list):
            raise ValueError("Invalid dataset format: expected a list")
            
        processed_count = gen_model_training_set(language_dataset, embedding_model, save_path)
        print(f"Successfully processed {processed_count} data entries")
        
        with open(save_path, "rb") as f: 
            loaded_dataset = pickle.load(f)    
        print(f"Saved dataset contains {len(loaded_dataset)} entries")
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise