import torch
from torch.utils.data import DataLoader
import numpy as np
from .cookie import Cookie
from .utils import get_tokenizer, read_json


class TextDataLoader(DataLoader):

    def __init__(self, data_set, pad_token_id, n_extend_blocks=3, n_extend_tokens=10, shuffle=True, batch_size=16, device="cuda"):
        super().__init__(data_set, batch_size, shuffle, collate_fn=self.collate_fn, drop_last=False)
        self.data_set = data_set
        self.pad_token_id = pad_token_id
        self.device = device
        self.n_extend_blocks = n_extend_blocks
        self.n_extend_tokens = n_extend_tokens
        self.max_seq_len = 512

    def read_ids(self, block, read_tokens_only=True):
        input_ids = []
        norm_ids = []
        punc_ids = []
        for token, norm_id, punc_id in block:
            assert len(token) > 0
            input_ids += token
            if not read_tokens_only:
                norm_ids += [norm_id] + [-100]*(len(token)-1)
                punc_ids += [punc_id] + [-100]*(len(token)-1)
        if read_tokens_only:
            return input_ids
        else:
            return input_ids, norm_ids, punc_ids

    def extend_tokens(self, input_ids, norm_ids, punc_ids, block_id):
        next_block = self.data_set[block_id+1][:self.n_extend_tokens] if block_id+1 < len(self.data_set) else []
        next_tokens = [token for tokens, _, _ in next_block for token in tokens]
        prev_block = self.data_set[block_id-1][-self.n_extend_tokens:] if block_id-1 >= 0 else []
        prev_tokens = [token for tokens, _, _ in prev_block for token in tokens]
        input_ids = prev_tokens + input_ids + next_tokens
        norm_ids = [-100]*len(prev_tokens) + norm_ids + [-100]*len(next_tokens)
        punc_ids = [-100]*len(prev_tokens) + punc_ids + [-100]*len(next_tokens)
        return input_ids, norm_ids, punc_ids

    def extend_blocks(self, block_id):
        next_blocks = self.data_set[block_id+1:block_id+self.n_extend_blocks+1]
        prev_blocks = self.data_set[block_id-self.n_extend_blocks:block_id]
        next_blocks = [self.read_ids(block) for block in next_blocks]
        prev_blocks = [self.read_ids(block) for block in prev_blocks]
        if len(next_blocks) > 0 and self.n_extend_tokens > 0:
            next_blocks[0] = self.read_ids(self.data_set[block_id+1][-self.n_extend_tokens:])
        if len(prev_blocks) > 0 and self.n_extend_tokens > 0:
            prev_blocks[-1] = self.read_ids(self.data_set[block_id-1][:self.n_extend_tokens])
        return next_blocks, prev_blocks

    def add_padding(self, ids, max_len, pad_id):
        return ids + [pad_id] * (max_len - len(ids))

    def process_extend_blocks(self, extend_blocks, pad_id):
        if len(extend_blocks) == 0:
            return None, None
        next_lens = []
        prev_lens = []
        for next_blocks, prev_blocks in extend_blocks:
            next_lens.append([len(block) for block in next_blocks])
            prev_lens.append([len(block) for block in prev_blocks])
            while len(next_lens[-1]) < self.n_extend_blocks:
                next_lens[-1] = next_lens[-1] + [0]
            while len(prev_lens[-1]) < self.n_extend_blocks:
                prev_lens[-1] = [0] + prev_lens[-1]
        next_lens = np.array(next_lens)
        prev_lens = np.array(prev_lens)
        max_next_lens = np.max(next_lens, 0)
        max_prev_lens = np.max(prev_lens, 0)
        all_next_blocks = [[] for _ in range(self.n_extend_blocks)]
        all_prev_blocks = [[] for _ in range(self.n_extend_blocks)]
        for next_blocks, prev_blocks in extend_blocks:
            for i in range(self.n_extend_blocks):
                max_len = max_next_lens[i]
                next_block = next_blocks[i] if i < len(next_blocks) else []
                next_block = self.add_padding(next_block, max_len, pad_id)
                all_next_blocks[i].append(next_block)
            for i in range(self.n_extend_blocks):
                max_len = max_prev_lens[i]
                delta = self.n_extend_blocks - len(prev_blocks)
                prev_block = prev_blocks[i - delta] if i >= delta else []
                prev_block = self.add_padding(prev_block, max_len, pad_id)
                all_prev_blocks[i].append(prev_block)
        all_next_blocks = [torch.LongTensor(block).to(self.device) for block in all_next_blocks]
        all_prev_blocks = [torch.LongTensor(block).to(self.device) for block in all_prev_blocks]
        return all_next_blocks, all_prev_blocks

    def collate_fn(self, data):
        outputs = []
        extend_blocks = []
        for block in data:
            block_id = self.data_set.index(block)
            input_ids, norm_ids, punc_ids = self.read_ids(block, read_tokens_only=False)
            if self.n_extend_tokens > 0:
                input_ids, norm_ids, punc_ids = self.extend_tokens(input_ids, norm_ids, punc_ids, block_id)
            if self.n_extend_blocks > 0:
                next_blocks, prev_blocks = self.extend_blocks(block_id)
                extend_blocks.append((next_blocks, prev_blocks))
            input_ids = input_ids[:self.max_seq_len]
            norm_ids = norm_ids[:self.max_seq_len]
            punc_ids = punc_ids[:self.max_seq_len]
            outputs.append((input_ids, norm_ids, punc_ids))
        all_input_ids = []
        all_mask_ids = []
        all_norm_ids = []
        all_punc_ids = []
        pad_id = self.pad_token_id
        max_input_len = max(len(items[0]) for items in outputs)
        for input_ids, norm_ids, punc_ids in outputs:
            all_mask_ids.append([1]*len(input_ids) + [0]*(max_input_len - len(input_ids)))
            all_input_ids.append(self.add_padding(input_ids, max_input_len, pad_id))
            all_norm_ids.append(self.add_padding(norm_ids, max_input_len, -100))
            all_punc_ids.append(self.add_padding(punc_ids, max_input_len, -100))
        input_ids = torch.LongTensor(all_input_ids).to(self.device)
        mask_ids = torch.LongTensor(all_mask_ids).to(self.device)
        norm_ids = torch.LongTensor(all_norm_ids).to(self.device)
        punc_ids = torch.LongTensor(all_punc_ids).to(self.device)
        next_blocks, prev_blocks = self.process_extend_blocks(extend_blocks, pad_id)
        return input_ids, mask_ids, norm_ids, punc_ids, next_blocks, prev_blocks


class Data:

    def __init__(self, data_config, tokenizer_config, n_blocks, n_tokens):
        self.data_config = data_config
        self.tokenizer_config = tokenizer_config
        self.block_size = data_config["block_size"]
        
        logging = data_config["logging"]
        data_name = data_config["name"]
        model_name = tokenizer_config["name"].replace("/", "_")
        self.cookie_folder = logging["cookie"]
        self.tensorboard_dir = logging["tensorboard"] + "/" + data_name + "/" + model_name
        
        self.norm_labels = data_config["dataset"]["norm_labels"]
        self.punc_labels = data_config["dataset"]["punc_labels"]

        self.tokenizer = None
        folder = f"{data_config['dataset']['folder']}"
        train_data = self.read_file(f"{folder}/train.conll")
        dev_data = self.read_file(f"{folder}/dev.conll")
        test_data = self.read_file(f"{folder}/test.conll")

        pad_id = self.read_pad_token_id()
        batch_size = data_config["hyperparams"]["batch_size"]
        device = data_config["hyperparams"]["device"]
        self.n_epochs = data_config["hyperparams"]["n_epochs"]
        self.learning_rate = data_config["hyperparams"]["learning_rate"]
        self.hidden_dim = data_config["hyperparams"]["hidden_dim"]
        self.device = device

        self.train_loader = TextDataLoader(train_data, pad_id, n_blocks, n_tokens, True, batch_size, device)
        self.dev_loader = TextDataLoader(dev_data, pad_id, n_blocks, n_tokens, True, batch_size, device)
        self.test_loader = TextDataLoader(test_data, pad_id, n_blocks, n_tokens, False, batch_size, device)
    
    def read_pad_token_id(self):
        cookie = Cookie(["read_pad_token_id", self.tokenizer_config], self.cookie_folder)
        result = cookie.read_cookie()
        if result is not None:
            return result
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer(self.tokenizer_config)
        return cookie.save_cookie(self.tokenizer.pad_token_id)

    def read_file(self, file_name):
        print(f"Reading file: {file_name}")
        cookie = Cookie(file_name, self.cookie_folder)
        result = cookie.read_cookie()
        if result is not None:
            return result
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer(self.tokenizer_config)
        blocks = []
        block = []
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                parts[0] = self.tokenizer.encode(parts[0], add_special_tokens=False)
                if len(parts[0]) == 0:
                    parts[0] = [self.tokenizer.unk_token_id]
                parts[1] = self.norm_labels.index(parts[1])
                parts[2] = self.punc_labels.index(parts[2])
                block.append(parts)
                if len(block) == self.block_size:
                    blocks.append(block)
                    block = []
        if len(block) > 0:
            blocks.append(block)
        return cookie.save_cookie(blocks)

    @classmethod
    def from_config(cls, data_config, model_config, n_blocks=3, n_tokens=50):
        data_config = read_json(data_config)
        tokenizer_config = read_json(model_config)["tokenizer"]
        return cls(data_config, tokenizer_config, n_blocks, n_tokens)