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
        mlp_dim = lstm_dim // 2

        if self.mode == "nojoint":
            self.norm_mlp = nn.Linear(lstm_dim*2, mlp_dim)
            self.punc_mlp = nn.Linear(lstm_dim*2, mlp_dim)
            self.norm_decoder = nn.Linear(mlp_dim, self.n_norm_labels)
            self.punc_decoder = nn.Linear(mlp_dim, self.n_punc_labels)
        elif self.mode == "norm_to_punc":
            self.norm_mlp = nn.Linear(lstm_dim*2, mlp_dim)
            self.punc_mlp = nn.Linear(lstm_dim*2+mlp_dim+self.n_norm_labels, mlp_dim)
            self.norm_decoder = nn.Linear(mlp_dim, self.n_norm_labels)
            self.punc_decoder = nn.Linear(mlp_dim, self.n_punc_labels)
        elif self.mode == "punc_to_norm":
            self.norm_mlp = nn.Linear(lstm_dim*2+mlp_dim+self.n_punc_labels, mlp_dim)
            self.punc_mlp = nn.Linear(lstm_dim*2, mlp_dim)
            self.norm_decoder = nn.Linear(mlp_dim, self.n_norm_labels)
            self.punc_decoder = nn.Linear(mlp_dim, self.n_punc_labels)
        
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
        lstm_output = self.forward_lstm(bert_output, next_blocks, prev_blocks)
        return lstm_output
    
    def forward_lstm(self, bert_output, next_blocks, prev_blocks):
        lstm_output, _ = self.lstm(bert_output)
        if next_blocks is not None and prev_blocks is not None:
            next_dim = next_blocks.shape[1]
            prev_dim = prev_blocks.shape[1]
            lstm_output = lstm_output[:,prev_dim:-next_dim,:].contiguous()
        lstm_output = torch.tanh(lstm_output)
        return lstm_output
    
    def forward_decoders(self, lstm_output, norm_ids=None, punc_ids=None, phase="nojoint"):
        norm_logits = None
        punc_logits = None
        
        if self.mode == "nojoint":
            norm_mlp_output = torch.tanh(self.norm_mlp(lstm_output))
            punc_mlp_output = torch.tanh(self.punc_mlp(lstm_output))
            norm_logits = self.norm_decoder(norm_mlp_output)
            punc_logits = self.punc_decoder(punc_mlp_output)
        elif self.mode == "norm_to_punc":
            norm_mlp_output = torch.tanh(self.norm_mlp(lstm_output))
            norm_logits = self.norm_decoder(norm_mlp_output)
            if phase in ["nojoint", "punc"]:
                if norm_ids is None:
                    norm_ids = torch.argmax(norm_logits, -1)
                norm_onehot = self.make_onehot(norm_ids, self.n_norm_labels)
                mlp_input = torch.cat((lstm_output, norm_mlp_output, norm_onehot), -1)
                punc_mlp_output = torch.tanh(self.punc_mlp(mlp_input))
                punc_logits = self.punc_decoder(punc_mlp_output)
        elif self.mode == "punc_to_norm":
            punc_mlp_output = torch.tanh(self.punc_mlp(lstm_output))
            punc_logits = self.punc_decoder(punc_mlp_output)
            if phase in ["nojoint", "norm"]:
                if punc_ids is None:
                    punc_ids = torch.argmax(punc_logits, -1)
                punc_onehot = self.make_onehot(punc_ids, self.n_punc_labels)
                mlp_input = torch.cat((lstm_output, punc_mlp_output, punc_onehot), -1)
                norm_mlp_output = torch.tanh(self.norm_mlp(mlp_input))
                norm_logits = self.norm_decoder(norm_mlp_output)
            
        return norm_logits, punc_logits

    def forward(self, input_ids, mask_ids, norm_ids=None, punc_ids=None, next_blocks=None, prev_blocks=None, phase="nojoint"):
        is_grad = torch.is_grad_enabled()
        if self.mode == "norm_to_punc" and phase == "punc":
            torch.set_grad_enabled(False)
        if self.mode == "punc_to_norm" and phase == "norm":
            torch.set_grad_enabled(False)
        bert_output = self.forward_bert(input_ids, mask_ids, next_blocks, prev_blocks)
        torch.set_grad_enabled(is_grad)
        
        norm_logits, punc_logits = self.forward_decoders(bert_output, norm_ids, punc_ids, phase)

        if norm_ids is None and punc_ids is None:
            return norm_logits, punc_logits
        else:
            if norm_logits is not None:
                norm_ids = norm_ids.view(-1)
                norm_loss = self.norm_loss_fct(norm_logits.view(norm_ids.shape[0], -1), norm_ids)
            else:
                norm_loss = -1
            if punc_logits is not None:
                punc_ids = punc_ids.view(-1)
                punc_loss = self.punc_loss_fct(punc_logits.view(punc_ids.shape[0], -1), punc_ids)
            else:
                punc_loss = -1
            return norm_loss, punc_loss