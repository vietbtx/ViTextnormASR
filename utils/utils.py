from functools import cache
import json
import sys
import transformers
import seqeval.metrics
from joblib import Memory
from tqdm import tqdm


memory = Memory('__pycache__', verbose=0, compress=True)


def read_json(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@cache
def _get_tokenizer(tokenizer_name, tokenizer_type):
    print(f"Reading tokenizer {tokenizer_name}")
    Tokenizer = getattr(sys.modules["transformers"], tokenizer_type)
    tokenizer = Tokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
    return tokenizer

def get_tokenizer(tokenizer_config):
    return _get_tokenizer(tokenizer_config["name"], tokenizer_config["type"])

@cache
def _get_model(model_name, model_type):
    print(f"Reading model {model_name}")
    Model = getattr(sys.modules["transformers"], model_type)
    model = Model.from_pretrained(model_name)
    return model

def get_model(model_config):
    return _get_model(model_config["name"], model_config["type"])

@cache
def get_score_metric(score_metric_name):
    metric = getattr(sys.modules["seqeval.metrics"], score_metric_name)
    return metric

@memory.cache
def read_pad_token_id(tokenizer_config):
    tokenizer = get_tokenizer(tokenizer_config)
    return tokenizer.pad_token_id

@cache
def _tokenizer(tokenizer, word):
    word = tokenizer.encode(word, add_special_tokens=False)
    if len(word) == 0:
        word = [tokenizer.unk_token_id]
    return word

@memory.cache
def _read_file(file_name):
    lines = []
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]
    return lines

@memory.cache
def read_file(file_name, tokenizer_config, norm_labels, punc_labels, block_size):
    print(f"Reading file: {file_name}")
    data = _read_file(file_name)
    tokenizer = get_tokenizer(tokenizer_config)
    words, norm_goals, punc_golds = zip(*data)
    tokens = [_tokenizer(tokenizer, word) for word in words]
    norm_goals = [norm_labels.index(norm) for norm in norm_goals]
    punc_goals = [punc_labels.index(punc) for punc in punc_golds]
    blocks = []
    block = []
    for word, token, norm, punc in zip(words, tokens, norm_goals, punc_goals):
        block.append((word, token, norm, punc))
        if len(block) == block_size:
            blocks.append(block)
            block = []
    if len(block) > 0:
        blocks.append(block)
    return blocks

