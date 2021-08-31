import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_model, read_json

class BERTModel(nn.Module):

    def __init__(self, model_config, norm_labels, punc_labels, lstm_dim, model_mode):
        super().__init__()
        self.bert = get_model(model_config)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, lstm_dim, bidirectional=True)
        self.n_norm_labels = len(norm_labels)
        self.n_punc_labels = len(punc_labels)

        self.mode = model_mode
        if self.mode == "nojoint":
            self.norm_decoder = nn.Linear(lstm_dim*2, self.n_norm_labels)
            self.punc_decoder = nn.Linear(lstm_dim*2, self.n_punc_labels)
        elif self.mode == "norm_to_punc":
            self.norm_decoder = nn.Linear(lstm_dim*2, self.n_norm_labels)
            self.punc_decoder = nn.Linear(lstm_dim*2 + self.n_norm_labels, self.n_punc_labels)
        elif self.mode == "punc_to_norm":
            self.norm_decoder = nn.Linear(lstm_dim*2 + self.n_punc_labels, self.n_norm_labels)
            self.punc_decoder = nn.Linear(lstm_dim*2, self.n_punc_labels)
        
        self.norm_loss_fct = nn.CrossEntropyLoss()
        self.punc_loss_fct = nn.CrossEntropyLoss()

    
    @classmethod
    def from_config(cls, model_config, norm_labels, punc_labels, lstm_dim, model_mode):
        model_config = read_json(model_config)["pretrained_model"]
        return cls(model_config, norm_labels, punc_labels, lstm_dim, model_mode)

    def make_onehot(self, tensor, num_classes):
        result = tensor.clone()
        result[result==-100] = num_classes
        result = F.one_hot(result, num_classes+1)
        result = result[...,:-1]
        return result
    
    def forward_bert(self, input_ids, mask_ids, next_blocks=None, prev_blocks=None):
        bert_output = self.bert(input_ids, mask_ids)[0]
        if next_blocks is not None and prev_blocks is not None:
            with torch.no_grad():
                next_blocks = [self.bert(block)[0] for block in next_blocks]
                prev_blocks = [self.bert(block)[0] for block in prev_blocks]
                next_blocks = torch.cat(next_blocks, 1)
                prev_blocks = torch.cat(prev_blocks, 1)
            bert_output = torch.cat((prev_blocks, bert_output, next_blocks), 1)
        return bert_output, next_blocks, prev_blocks
    
    def forward_lstm(self, bert_output, next_blocks, prev_blocks):
        lstm_output, _ = self.lstm(bert_output)
        if next_blocks is not None and prev_blocks is not None:
            next_dim = next_blocks.shape[1]
            prev_dim = prev_blocks.shape[1]
            lstm_output = lstm_output[:,prev_dim:-next_dim,:].contiguous()
        return lstm_output
    
    def forward_decoders(self, lstm_output, norm_ids=None, punc_ids=None):
        if self.mode == "nojoint":
            norm_logits = self.norm_decoder(lstm_output)
            punc_logits = self.punc_decoder(lstm_output)
        elif self.mode == "norm_to_punc":
            norm_logits = self.norm_decoder(lstm_output)
            if norm_ids is None:
                norm_ids = torch.argmax(norm_logits, -1)
            norm_onehot = self.make_onehot(norm_ids, self.n_norm_labels)
            punc_logits = self.punc_decoder(torch.cat((lstm_output, norm_onehot), -1))
        elif self.mode == "punc_to_norm":
            punc_logits = self.punc_decoder(lstm_output)
            if punc_ids is None:
                punc_ids = torch.argmax(punc_logits, -1)
            punc_onehot = self.make_onehot(punc_ids, self.n_punc_labels)
            norm_logits = self.norm_decoder(torch.cat((lstm_output, punc_onehot), -1))
        return norm_logits, punc_logits

    def forward(self, input_ids, mask_ids, norm_ids=None, punc_ids=None, next_blocks=None, prev_blocks=None):
        bert_output, next_blocks, prev_blocks = self.forward_bert(input_ids, mask_ids, next_blocks, prev_blocks)
        lstm_output = self.forward_lstm(bert_output, next_blocks, prev_blocks)
        norm_logits, punc_logits = self.forward_decoders(lstm_output, norm_ids, punc_ids)

        if norm_ids is not None and punc_ids is not None:
            norm_ids = norm_ids.view(-1)
            punc_ids = punc_ids.view(-1)
            norm_loss = self.norm_loss_fct(norm_logits.view(norm_ids.shape[0], -1), norm_ids)
            punc_loss = self.punc_loss_fct(punc_logits.view(punc_ids.shape[0], -1), punc_ids)
            return norm_loss, punc_loss
        else:
            return norm_logits, punc_logits