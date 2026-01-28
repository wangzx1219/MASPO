
import json
import random
from typing import List, Dict, Any

from config import DATASET_CONFIG, TaskType

def load_test_data(dataset: str) -> List[Dict[str, Any]]:
    config = DATASET_CONFIG.get(dataset)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    file_path = config["path"]
    task_type = config["task_type"]
    problem_key = config["problem_key"]
    answer_key = config["answer_key"]
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            
            entry = {
                'problem': item[problem_key],
                'unique_id': item.get('unique_id', item.get('task_id', str(len(data)))),
                'task_type': task_type,
            }
            
            if task_type == TaskType.CODE:
                entry['answer'] = ''
                entry['test_list'] = item.get('test_list', [])
            else:
                entry['answer'] = item.get(answer_key, '')
            
            data.append(entry)
    
    return data

def load_train_for_opt(dataset: str, k: int = 50) -> List[str]:
    data = load_test_data(dataset)
    if k > len(data):
        k = len(data)
    return [item["problem"] for item in random.sample(data, k)]

def get_task_type(dataset: str) -> TaskType:
    config = DATASET_CONFIG.get(dataset)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset}")
    return config["task_type"]

def get_default_use_judge(dataset: str) -> bool:
    config = DATASET_CONFIG.get(dataset)
    if not config:
        return False
    return config.get("default_use_judge", False)
