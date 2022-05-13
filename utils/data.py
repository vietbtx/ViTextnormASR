from functools import cache
import torch
from torch.utils.data import DataLoader
import numpy as np
from .utils import get_tokenizer, read_file, read_json, read_pad_token_id


class TextDataLoader(DataLoader):

    def __init__(self, data_set, tokenizer, surrounding_context=True, shuffle=True, batch_size=16, device="cuda"):
        super().__init__(data_set, batch_size, shuffle, collate_fn=self.collate_fn, drop_last=False)
        self.data_set = data_set
        self.tokenizer = tokenizer
        self.device = device
        self.surrounding_context = surrounding_context
        self.max_seq_len = 512

    @cache
    def process_single_block(self, block_id, add_label=True):
        data = []
        for word, tokens, norm_id, punc_id in self.data_set[block_id]:
            is_first = True
            for token in tokens:
                if is_first:
                    if not add_label:
                        norm_id = -100
                        punc_id = -100
                    data.append((word, token, norm_id, punc_id))
                else:
                    data.append(("", token, -100, -100))
                is_first = False
        return data
    
    def process_block(self, block):
        block_id = self.data_set.index(block)
        block_data = self.process_single_block(block_id)
        start_id = 0
        if self.surrounding_context:
            range_id = 1
            while len(block_data) < self.max_seq_len:
                if block_id - range_id >= 0:
                    block_data = self.process_single_block(block_id-range_id, False) + block_data
                if block_id + range_id < len(self.data_set):
                    block_data = block_data + self.process_single_block(block_id+range_id, False)
                range_id += 1
            start_id = max(len(block_data)//2-self.max_seq_len//2+1, 0)
        block_data = block_data[start_id:start_id+self.max_seq_len-2]
        block_data = [("", self.tokenizer.cls_token_id, -100, -100)] + block_data
        block_data = block_data + [("", self.tokenizer.sep_token_id, -100, -100)]
        return block_data

    def add_padding(self, block, max_len):
        while len(block) < max_len:
            block.append(("", self.tokenizer.pad_token_id, -100, -100))
        return block

    def collate_fn(self, data):
        blocks = []
        for block in data:
            block_data = self.process_block(block)
            blocks.append(block_data)
        words = []
        input_ids = []
        mask_ids = []
        norm_ids = []
        punc_ids = []
        max_len = max(len(block) for block in blocks)
        for block in blocks:
            mask_ids.append([1]*len(block) + [0]*(max_len - len(block)))
            block += [("", self.tokenizer.pad_token_id, -100, -100)] * (max_len - len(block))
            word, token, norm, punc = zip(*block)
            words.append(word)
            input_ids.append(token)
            norm_ids.append(norm)
            punc_ids.append(punc)
        input_ids = torch.LongTensor(input_ids).to(self.device)
        mask_ids = torch.LongTensor(mask_ids).to(self.device)
        norm_ids = torch.LongTensor(norm_ids).to(self.device)
        punc_ids = torch.LongTensor(punc_ids).to(self.device)
        return words, input_ids, mask_ids, norm_ids, punc_ids


class Data:

    def __init__(self, data_config, tokenizer_config, use_sc=True):
        self.data_config = data_config
        self.tokenizer_config = tokenizer_config
        self.block_size = data_config["block_size"]
        
        logging = data_config["logging"]
        data_name = data_config["name"]
        model_name = tokenizer_config["name"].replace("/", "_")
        self.tensorboard_dir = logging["tensorboard"] + "/" + data_name + "/" + model_name
        
        self.norm_labels = data_config["dataset"]["norm_labels"]
        self.punc_labels = data_config["dataset"]["punc_labels"]

        self.tokenizer = get_tokenizer(tokenizer_config)
        folder = f"{data_config['dataset']['folder']}"
        # train_data = read_file(f"{folder}/train.conll", tokenizer_config, self.norm_labels, self.punc_labels, self.block_size)
        dev_data = read_file(f"{folder}/dev.conll", tokenizer_config, self.norm_labels, self.punc_labels, self.block_size)
        # test_data = read_file(f"{folder}/test.conll", tokenizer_config, self.norm_labels, self.punc_labels, self.block_size)

        # pad_id = read_pad_token_id(tokenizer_config)
        batch_size = data_config["hyperparams"]["batch_size"]
        device = data_config["hyperparams"]["device"]
        self.n_epochs = data_config["hyperparams"]["n_epochs"]
        self.learning_rate = data_config["hyperparams"]["learning_rate"]
        self.hidden_dim = data_config["hyperparams"]["hidden_dim"]
        self.device = device

        self.train_loader = TextDataLoader(dev_data, self.tokenizer, use_sc, True, batch_size, device)
        self.dev_loader = TextDataLoader(dev_data, self.tokenizer, use_sc, True, batch_size*8, device)
        self.test_loader = TextDataLoader(dev_data, self.tokenizer, use_sc, False, batch_size*8, device)
        
    
    

    @classmethod
    def from_config(cls, data_config, model_config, use_sc):
        data_config = read_json(data_config)
        tokenizer_config = read_json(model_config)["tokenizer"]
        return cls(data_config, tokenizer_config, use_sc)