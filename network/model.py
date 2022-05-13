import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import _tokenizer, get_model, read_json
from .layers import AttentionLayer, BiaffineAttention

class BERTModel(nn.Module):

    def __init__(self, model_config, tokenizer, norm_labels, punc_labels, hidden_dim, model_mode, use_biaffine=True):
        super().__init__()
        self.bert = get_model(model_config)
        # self.attn = AttentionLayer(self.bert.config.hidden_size, hidden_dim)
        hidden_dim = self.bert.config.hidden_size
        self.norm_labels = norm_labels
        self.punc_labels = punc_labels

        self.mode = model_mode
        self.tokenizer = tokenizer
        
        if self.mode == "nojoint":
            self.norm_decoder = nn.Linear(hidden_dim, len(norm_labels))
            self.punc_decoder = nn.Linear(hidden_dim, len(punc_labels))
        elif self.mode == "norm_to_punc":
            self.norm_decoder = nn.Linear(hidden_dim, len(norm_labels))
            self.punc_decoder = nn.Linear(hidden_dim*2+len(norm_labels), len(punc_labels))
        elif self.mode == "punc_to_norm":
            self.norm_decoder = nn.Linear(hidden_dim*2+len(punc_labels), len(norm_labels))
            self.punc_decoder = nn.Linear(hidden_dim, len(punc_labels))

        self.norm_criterion = nn.CrossEntropyLoss()
        self.punc_criterion = nn.CrossEntropyLoss()
    
    @classmethod
    def from_config(cls, model_config, tokenizer, norm_labels, punc_labels, hidden_dim, model_mode, use_biaffine=True):
        model_config = read_json(model_config)["pretrained_model"]
        return cls(model_config, tokenizer, norm_labels, punc_labels, hidden_dim, model_mode, use_biaffine)

    def forward_bert(self, input_ids, mask_ids=None):
        bert_output = self.bert(input_ids, mask_ids)[0]
        if self.training and self.mode == "nojoint":
            bert_output.register_hook(lambda grad: torch.zeros_like(grad))
        return bert_output

    def forward_norm(self, bert_output, norm_logits, words, input_ids, mask_ids):
        norm_preds = torch.argmax(norm_logits, -1).detach().cpu().numpy()
        new_token_ids = []
        for tokens, norms in zip(words, norm_preds):
            token_ids = []
            for token, norm in zip(tokens, norms):
                if len(token) == 0: continue
                if self.norm_labels[norm] == "B-CAP":
                    token = token.capitalize()
                token_ids += _tokenizer(self.tokenizer, token)
            token_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]
            new_token_ids.append(token_ids)
        max_len = max(len(ids) for ids in new_token_ids)
        new_mask_ids = [[1]*len(ids)+[0]*(max_len-len(ids)) for ids in new_token_ids]
        new_token_ids = [ids+[self.tokenizer.pad_token_id]*(max_len-len(ids)) for ids in new_token_ids]
        with torch.no_grad():
            new_mask_ids = torch.LongTensor(new_mask_ids).to(mask_ids.device)
            new_token_ids = torch.LongTensor(new_token_ids).to(input_ids.device)
            norm_features = self.forward_bert(new_token_ids, new_mask_ids)[:, :1, :]
            norm_features = torch.tile(norm_features, [1, norm_logits.shape[1], 1])
        norm_features = torch.cat((bert_output, norm_features, norm_logits), -1)
        return norm_features
    
    def forward_punc(self, bert_output, punc_logits, words, input_ids, mask_ids):
        punc_preds = torch.argmax(punc_logits, -1).detach().cpu().numpy()
        new_token_ids = []
        punc_symbols = [".", ",", "?", "!", ":", ";", "-", ""]
        for tokens, puncs in zip(words, punc_preds):
            token_ids = []
            new_tokens = []
            for token, punc in zip(tokens, puncs):
                if len(token) == 0: continue
                new_tokens.append(token)
                if len(punc_symbols[punc]) > 0:
                    new_tokens.append(punc_symbols[punc])
            for i in range(len(new_tokens)-1):
                if new_tokens[i] in [".", "?", "!"]:
                    new_tokens[i+1] = new_tokens[i+1].capitalize()
            for token in new_tokens:
                token_ids += _tokenizer(self.tokenizer, token)
            token_ids = [self.tokenizer.cls_token_id] + token_ids[:510] + [self.tokenizer.sep_token_id]
            new_token_ids.append(token_ids)
        max_len = max(len(ids) for ids in new_token_ids)
        new_mask_ids = [[1]*len(ids)+[0]*(max_len-len(ids)) for ids in new_token_ids]
        new_token_ids = [ids+[self.tokenizer.pad_token_id]*(max_len-len(ids)) for ids in new_token_ids]
        with torch.no_grad():
            new_mask_ids = torch.LongTensor(new_mask_ids).to(mask_ids.device)
            new_token_ids = torch.LongTensor(new_token_ids).to(input_ids.device)
            punc_features = self.forward_bert(new_token_ids, new_mask_ids)[:, :1, :]
            punc_features = torch.tile(punc_features, [1, punc_logits.shape[1], 1])
        punc_features = torch.cat((bert_output, punc_features, punc_logits), -1)
        return punc_features

    def forward(self, words, input_ids, mask_ids, norm_ids=None, punc_ids=None):
        bert_output = self.forward_bert(input_ids, mask_ids)

        if self.mode == "norm_to_punc":
            norm_logits = self.norm_decoder(bert_output)
            norm_features = self.forward_norm(bert_output, norm_logits, words, input_ids, mask_ids)
            punc_logits = self.punc_decoder(norm_features)
        elif self.mode == "punc_to_norm":
            punc_logits = self.punc_decoder(bert_output)
            punc_features = self.forward_punc(bert_output, punc_logits, words, input_ids, mask_ids)
            norm_logits = self.norm_decoder(punc_features)
        else:
            norm_logits = self.norm_decoder(bert_output)
            punc_logits = self.punc_decoder(bert_output)

        if norm_ids is None and punc_ids is None:
            return norm_logits, punc_logits
        else:
            norm_ids = norm_ids.view(-1)
            punc_ids = punc_ids.view(-1)
            norm_loss = self.norm_criterion(norm_logits.view(norm_ids.shape[0], -1), norm_ids)
            punc_loss = self.punc_criterion(punc_logits.view(punc_ids.shape[0], -1), punc_ids)
            return norm_loss, punc_loss
