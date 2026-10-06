import os
import json
from datasets import load_dataset as hf_load_dataset

dataset_dir_path = os.path.dirname(os.path.realpath(__file__))

SPLITS = ['train', 'val', 'test']
HARMTYPES = ['harmless', 'harmful']

SPLIT_DATASET_FILENAME = os.path.join(dataset_dir_path, 'splits/{harmtype}_{split}.json')

PROCESSED_DATASET_NAMES = ["advbench", "tdc2023", "malicious_instruct", "harmbench", "jailbreakbench", "strongreject", "alpaca"]

def load_explicit_bias(split: str):
    """
    Loads the ToxiGen dataset from the SteeringSafety benchmark.
    Dataset contains ~9k train, 900 val, and 4.5k test rows covering multiple demographic groups.
    """
    # Map 'val' to 'validation' to match Hugging Face split naming
    hf_split = 'validation' if split == 'val' else split
    
    try:
        hf_data = hf_load_dataset("WangResearchLab/SteeringSafety", "ToxiGen", split=hf_split)
    except Exception as e:
        print(f"Error loading ToxiGen from Hugging Face: {e}")
        return []
        
    data = []
    for item in hf_data:
        # For ToxiGen, hate=1 means it IS hateful, so correct answer to "Is it hateful?" is "Yes"
        expected = "Yes" if item["hate"] == 1 else "No"
        data.append({
            "instruction": item["text"],
            "prompt": item["text"],
            "category": "explicit_bias",
            "expected_answer": expected,
            "hate_label": item["hate"],
            "group": item["group"]
        })
    return data

def load_dataset_split(harmtype: str, split: str, instructions_only: bool=False):
    
    # Route explicit bias requests to the new Hugging Face loader
    if harmtype == 'explicit_bias':
        dataset = load_explicit_bias(split)
        if instructions_only:
            dataset = [d['instruction'] for d in dataset]
        return dataset

    file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset

def load_dataset(dataset_name, instructions_only: bool=False):
    assert dataset_name in PROCESSED_DATASET_NAMES, f"Valid datasets: {PROCESSED_DATASET_NAMES}"

    file_path = os.path.join(dataset_dir_path, 'processed', f"{dataset_name}.json")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]
 
    return dataset