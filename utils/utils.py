import json
import sys
from tqdm import tqdm
import transformers
import seqeval.metrics
from joblib import Memory


memory = Memory('__pycache__', verbose=0)


@memory.cache(ignore=["tokenizer"])
def read_file(file_name, tokenizer, norm_labels, punc_labels, block_size):
    lines = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            lines.append(parts)
    blocks = []
    block = []
    for parts in tqdm(lines, f"Reading file: {file_name}", leave=True):
        parts[0] = tokenizer.encode(parts[0], add_special_tokens=False)
        if len(parts[0]) == 0:
            parts[0] = [tokenizer.unk_token_id]
        parts[1] = norm_labels.index(parts[1])
        parts[2] = punc_labels.index(parts[2])
        block.append(parts)
        if len(block) == block_size:
            blocks.append(block)
            block = []
    if len(block) > 0:
        blocks.append(block)
    return blocks
    

def read_json(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_tokenizer(tokenizer_config):
    print(f"Reading tokenizer: {tokenizer_config['name']}")
    Tokenizer = getattr(sys.modules["transformers"], tokenizer_config["type"])
    tokenizer = Tokenizer.from_pretrained(tokenizer_config["name"], do_lower_case=False, cache_dir="__pycache__")
    return tokenizer


def get_model(model_config):
    print(f"Reading model: {model_config['name']}")
    Model = getattr(sys.modules["transformers"], model_config["type"])
    model = Model.from_pretrained(model_config["name"], cache_dir="__pycache__")
    return model


def get_score_metric(score_metric_name):
    metric = getattr(sys.modules["seqeval.metrics"], score_metric_name)
    return metric

