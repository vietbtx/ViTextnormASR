import json
import sys
import transformers
import seqeval.metrics


def read_json(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_tokenizer(tokenizer_config):
    print(f"Reading tokenizer {tokenizer_config['name']}")
    Tokenizer = getattr(sys.modules["transformers"], tokenizer_config["type"])
    tokenizer = Tokenizer.from_pretrained(
        tokenizer_config["name"], do_lower_case=False)
    return tokenizer


def get_model(model_config):
    print(f"Reading model {model_config['name']}")
    Model = getattr(sys.modules["transformers"], model_config["type"])
    model = Model.from_pretrained(model_config["name"])
    return model


def get_score_metric(score_metric_name):
    metric = getattr(sys.modules["seqeval.metrics"], score_metric_name)
    return metric

