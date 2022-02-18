import torch
from torch.utils.data import DataLoader
from .utils import get_tokenizer, read_file, read_json


class DataLoader(DataLoader):

    def __init__(self, data_set, pad_token_id, extend_tokens, shuffle=True, batch_size=16, device="cuda"):
        super().__init__(data_set, batch_size, shuffle, collate_fn=self.collate_fn)
        self.data_set = data_set
        self.pad_token_id = pad_token_id
        self.device = device
        self.extend_tokens = extend_tokens
        self.max_seq_len = 256

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

    def get_next_tokens(self, block_id):
        next_block = self.data_set[block_id+1] if block_id+1 < len(self.data_set) else []
        next_tokens = [token for tokens, _, _ in next_block for token in tokens]
        return next_tokens
    
    def get_prev_tokens(self, block_id):
        prev_block = self.data_set[block_id-1] if block_id-1 >= 0 else []
        prev_tokens = [token for tokens, _, _ in prev_block for token in tokens]
        return prev_tokens

    def get_extend_tokens(self, input_ids, norm_ids, punc_ids, block_id):
        k = 0
        while True:
            next_tokens = self.get_next_tokens(block_id + k)
            prev_tokens = self.get_prev_tokens(block_id - k)
            input_ids = prev_tokens + input_ids + next_tokens
            norm_ids = [-100]*len(prev_tokens) + norm_ids + [-100]*len(next_tokens)
            punc_ids = [-100]*len(prev_tokens) + punc_ids + [-100]*len(next_tokens)
            k += 1
            if len(input_ids) > self.max_seq_len - 2:
                while len(input_ids) > self.max_seq_len - 2:
                    if norm_ids[0] == -100:
                        norm_ids = norm_ids[1:]
                        punc_ids = punc_ids[1:]
                        input_ids = input_ids[1:]
                    if norm_ids[-1] == -100:
                        norm_ids = norm_ids[:-1]
                        punc_ids = punc_ids[:-1]
                        input_ids = input_ids[:-1]
                break
        
        return input_ids, norm_ids, punc_ids

    def add_padding(self, ids, max_len, pad_id):
        return ids + [pad_id] * (max_len - len(ids))

    def collate_fn(self, data):
        outputs = []
        for block in data:
            block_id = self.data_set.index(block)
            input_ids, norm_ids, punc_ids = self.read_ids(block, read_tokens_only=False)
            if self.extend_tokens:
                input_ids, norm_ids, punc_ids = self.get_extend_tokens(input_ids, norm_ids, punc_ids, block_id)
            outputs.append((input_ids, norm_ids, punc_ids))
        all_input_ids, all_mask_ids, all_norm_ids, all_punc_ids = [], [], [], []
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
        return input_ids, mask_ids, norm_ids, punc_ids


class Data:

    def __init__(self, data_config, tokenizer_config, extend_tokens):
        self.data_config = data_config
        block_size = data_config["block_size"]
        
        logging = data_config["logging"]
        data_name = data_config["name"]
        model_name = tokenizer_config["name"].replace("/", "_")
        self.cookie_folder = logging["cookie"]
        self.tensorboard_dir = logging["tensorboard"] + "/" + data_name + "/" + model_name
        self.model_name = model_name

        norm_labels = data_config["dataset"]["norm_labels"]
        punc_labels = data_config["dataset"]["punc_labels"]

        tokenizer = get_tokenizer(tokenizer_config)
        folder = f"{data_config['dataset']['folder']}"
        train_data = read_file(f"{folder}/train.conll", tokenizer, norm_labels, punc_labels, block_size)
        dev_data = read_file(f"{folder}/dev.conll", tokenizer, norm_labels, punc_labels, block_size)
        test_data = read_file(f"{folder}/test.conll", tokenizer, norm_labels, punc_labels, block_size)

        pad_id = tokenizer.pad_token_id
        batch_size = data_config["hyperparams"]["batch_size"]
        device = data_config["hyperparams"]["device"]
        self.n_epochs = data_config["hyperparams"]["n_epochs"]
        self.learning_rate = data_config["hyperparams"]["learning_rate"]
        self.hidden_dim = data_config["hyperparams"]["hidden_dim"]
        self.device = device
        self.norm_labels = norm_labels
        self.punc_labels = punc_labels

        self.train_loader = DataLoader(train_data, pad_id, extend_tokens, True, batch_size, device)
        self.dev_loader = DataLoader(dev_data, pad_id, extend_tokens, False, batch_size*8, device)
        self.test_loader = DataLoader(test_data, pad_id, extend_tokens, False, batch_size*8, device)

    @classmethod
    def from_config(cls, data_config, model_config, extend_tokens):
        data_config = read_json(data_config)
        tokenizer_config = read_json(model_config)["tokenizer"]
        return cls(data_config, tokenizer_config, extend_tokens)