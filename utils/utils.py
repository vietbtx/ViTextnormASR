import json
import sys
import seqeval.metrics
from joblib import Memory
from transformers import BertConfig, BertModelWithHeads, ElectraModel, AutoModelWithHeads

memory = Memory('__pycache__', verbose=1)


@memory.cache
def read_file(file_name, tokenizer, norm_labels, punc_labels, block_size):
    print(f"Reading file: {file_name}")
    blocks = []
    block = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
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
    print(f"Reading tokenizer {tokenizer_config['name']}")
    Tokenizer = getattr(sys.modules["transformers"], tokenizer_config["type"])
    tokenizer = Tokenizer.from_pretrained(tokenizer_config["name"], do_lower_case=False, cache_dir="__pycache__")
    return tokenizer


def get_model(model_config):
    print(f"Reading model {model_config['name']}")
    if "electra" in model_config["name"]:
        electra = ElectraModel.from_pretrained(model_config["name"])
        config = electra.config.to_dict()
        config["architectures"] = ["BertForPreTraining"]
        config["model_type"] = 'bert'
        config = BertConfig.from_dict(config)
        model = BertModelWithHeads(config)
        model.bert.load_state_dict(electra.state_dict(), False)
    else:
        model = AutoModelWithHeads.from_pretrained(model_config["name"], cache_dir="__pycache__")
    return model


def get_score_metric(score_metric_name):
    metric = getattr(sys.modules["seqeval.metrics"], score_metric_name)
    return metric

