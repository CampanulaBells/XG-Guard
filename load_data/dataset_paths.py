import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_LOCAL = os.path.join(ROOT, "datasets_local")
DATASETS_ONLINE = os.path.join(ROOT, "datasets_online")

LOCAL_PATH_CONFIG = {
    "MA-PoisonRAG": {
        "train": os.path.join(DATASETS_LOCAL, "MA_PoisonRAG_local_train_dataset.json"),
        "test": os.path.join(DATASETS_LOCAL, "MA_PoisonRAG_local_test_dataset.json"),
        "emb_cache": "cahced_data_MA_PoisonRAG.pkl",
        "emb_cache_test": "cahced_data_MA_PoisonRAG_test.pkl",
    },
    "MA-CSQA": {
        "train": os.path.join(DATASETS_LOCAL, "MA_CSQA_local_train_dataset.json"),
        "test": os.path.join(DATASETS_LOCAL, "MA_CSQA_local_test_dataset.json"),
        "emb_cache": "cahced_data_MA_CSQA.pkl",
        "emb_cache_test": "cahced_data_MA_CSQA_test.pkl",
    },
    "TA-InjecAgent": {
        "train": os.path.join(DATASETS_LOCAL, "TA_InjecAgent_local_train_dataset.json"),
        "test": os.path.join(DATASETS_LOCAL, "TA_InjecAgent_local_test_dataset.json"),
        "emb_cache": "cahced_data_TA_InjecAgent.pkl",
        "emb_cache_test": "cahced_data_TA_InjecAgent_test.pkl",
    },
    "PI-CSQA": {
        "train": os.path.join(DATASETS_LOCAL, "PI_CSQA_local_train_dataset.json"),
        "test": os.path.join(DATASETS_LOCAL, "PI_CSQA_local_test_dataset.json"),
        "emb_cache": "cahced_data_PI_CSQA.pkl",
        "emb_cache_test": "cahced_data_PI_CSQA_test.pkl",
    },
    "PI-GSM8K": {
        "train": os.path.join(DATASETS_LOCAL, "PI_GSM8K_local_train_dataset.json"),
        "test": os.path.join(DATASETS_LOCAL, "PI_GSM8K_local_test_dataset.json"),
        "emb_cache": "cahced_data_PI_GSM8K.pkl",
        "emb_cache_test": "cahced_data_PI_GSM8K_test.pkl",
    },
    "PI-MMLU": {
        "train": os.path.join(DATASETS_LOCAL, "PI_MMLU_local_train_dataset.json"),
        "test": os.path.join(DATASETS_LOCAL, "PI_MMLU_local_test_dataset.json"),
        "emb_cache": "cahced_data_PI_MMLU.pkl",
        "emb_cache_test": "cahced_data_PI_MMLU_test.pkl",
    },
}

ONLINE_TEST_PATH_CONFIG = {
    "MA-PoisonRAG": os.path.join(DATASETS_ONLINE, "MA", "agent_graph_dataset", "memory_attack", "test", "dataset.json"),
    "MA-CSQA": os.path.join(DATASETS_ONLINE, "MA-CSQA", "agent_graph_dataset", "memory_attack", "test", "dataset.json"),
    "TA-InjecAgent": os.path.join(DATASETS_ONLINE, "TA", "agent_graph_dataset", "tool_attack", "test", "dataset.json"),
    "PI-CSQA": os.path.join(DATASETS_ONLINE, "PI", "agent_graph_dataset", "csqa", "test", "dataset.json"),
    "PI-GSM8K": os.path.join(DATASETS_ONLINE, "PI", "agent_graph_dataset", "gsm8k", "test", "dataset.json"),
    "PI-MMLU": os.path.join(DATASETS_ONLINE, "PI", "agent_graph_dataset", "mmlu", "test", "dataset.json"),
}

ONLINE_TRAIN_PATH_CONFIG = {
    "MA-PoisonRAG": os.path.join(DATASETS_ONLINE, "MA", "agent_graph_dataset", "memory_attack", "train", "dataset.json"),
    "MA-CSQA": os.path.join(DATASETS_ONLINE, "MA-CSQA", "agent_graph_dataset", "memory_attack", "train", "dataset.json"),
    "TA-InjecAgent": os.path.join(DATASETS_ONLINE, "TA", "agent_graph_dataset", "tool_attack", "train", "dataset.json"),
    "PI-CSQA": os.path.join(DATASETS_ONLINE, "PI", "agent_graph_dataset", "csqa", "train", "dataset.json"),
    "PI-GSM8K": os.path.join(DATASETS_ONLINE, "PI", "agent_graph_dataset", "gsm8k", "train", "dataset.json"),
    "PI-MMLU": os.path.join(DATASETS_ONLINE, "PI", "agent_graph_dataset", "mmlu", "train", "dataset.json"),
}

ONLINE_TRAIN1_PATH_CONFIG = {
    "MA-PoisonRAG": os.path.join(DATASETS_ONLINE, "MA", "agent_graph_dataset", "memory_attack", "train1", "dataset.json"),
    "MA-CSQA": os.path.join(DATASETS_ONLINE, "MA-CSQA", "agent_graph_dataset", "memory_attack", "train1", "dataset.json"),
    "TA-InjecAgent": os.path.join(DATASETS_ONLINE, "TA", "agent_graph_dataset", "tool_attack", "train1", "dataset.json"),
    "PI-CSQA": os.path.join(DATASETS_ONLINE, "PI", "agent_graph_dataset", "csqa", "train1", "dataset.json"),
    "PI-GSM8K": os.path.join(DATASETS_ONLINE, "PI", "agent_graph_dataset", "gsm8k", "train1", "dataset.json"),
    "PI-MMLU": os.path.join(DATASETS_ONLINE, "PI", "agent_graph_dataset", "mmlu", "train1", "dataset.json"),
}


def experiment_key(atk_type, expr_type):
    return f"{atk_type}-{expr_type}"


def get_online_test_path(atk_type, expr_type):
    key = experiment_key(atk_type, expr_type)
    return ONLINE_TEST_PATH_CONFIG[key]
