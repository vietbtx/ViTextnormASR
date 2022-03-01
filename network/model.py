import torch.nn as nn
from transformers.adapters.composition import Fuse
from utils.utils import get_model, read_json


class AdapterModel(nn.Module):

    def __init__(self, model_config, norm_labels, punc_labels, model_mode, adapter_path):
        super().__init__()
        self.bert = get_model(model_config)
        self.model_mode = model_mode
        norm_id2label = {id: label for id, label in enumerate(norm_labels)}
        punc_id2label = {id: label for id, label in enumerate(punc_labels)}

        if model_mode == "norm_only":
            self.bert.add_adapter("ner_norm")
            self.bert.add_tagging_head("ner_norm", len(norm_labels), id2label=norm_id2label)
            self.bert.train_adapter("ner_norm")
        elif model_mode == "punc_only":
            self.bert.add_adapter("ner_punc")
            self.bert.add_tagging_head("ner_punc", len(punc_labels), id2label=punc_id2label)
            self.bert.train_adapter("ner_punc")
        elif model_mode == "norm_to_punc":
            self.bert.load_adapter(adapter_path, with_head=False)
            self.bert.add_adapter_fusion(Fuse("ner_norm"))
            self.bert.set_active_adapters(Fuse("ner_norm"))
            self.bert.train_adapter_fusion(Fuse("ner_norm"))
            self.bert.add_tagging_head("ner_punc", len(punc_labels), id2label=punc_id2label)
        elif model_mode == "punc_to_norm":
            self.bert.load_adapter(adapter_path, with_head=False)
            self.bert.add_adapter_fusion(Fuse("ner_punc"))
            self.bert.set_active_adapters(Fuse("ner_punc"))
            self.bert.train_adapter_fusion(Fuse("ner_punc"))
            self.bert.add_tagging_head("ner_norm", len(norm_labels), id2label=norm_id2label)

        self.criterion = nn.CrossEntropyLoss()
    
    @classmethod
    def from_config(cls, model_config, norm_labels, punc_labels, model_mode, adapter_path):
        model_config = read_json(model_config)["pretrained_model"]
        return cls(model_config, norm_labels, punc_labels, model_mode, adapter_path)

    def forward(self, input_ids, mask_ids, norm_ids=None, punc_ids=None):
        logits = self.bert(input_ids, mask_ids).logits
        if norm_ids is None and punc_ids is None:
            return logits
        else:
            if norm_ids is not None:
                norm_ids = norm_ids.view(-1)
                norm_loss = self.criterion(logits.view(norm_ids.shape[0], -1), norm_ids)
            if punc_ids is not None:
                punc_ids = punc_ids.view(-1)
                punc_loss = self.criterion(logits.view(punc_ids.shape[0], -1), punc_ids)
            loss = norm_loss + punc_loss
            return loss