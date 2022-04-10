from collections import Counter, defaultdict
import torch
import torch.nn as nn
from utils.utils import get_model, read_json

class SpanDecoder(nn.Module):

    def __init__(self, input_dim, labels, addion_dim=0, use_span=False, max_span_distance=8):
        super().__init__()
        self.input_dim = input_dim
        self.labels = labels
        self.span_labels = [label for label in labels if not label.startswith("I-")]
        self.use_span = use_span and len(self.span_labels) != len(labels)
        self.max_span_distance = max_span_distance
        if self.use_span:
            self.distance_emb = nn.Embedding(max_span_distance, input_dim)
            input_dim = input_dim*3
            self.decoder_len = len(self.span_labels)
        else:
            self.decoder_len = len(labels)
        self.decoder = nn.Linear(input_dim + addion_dim, self.decoder_len)
        self.criterion = nn.CrossEntropyLoss()

    def generate_pred_label_ids(self, logits, label_ids, pair_ids=None):
        pred_ids = torch.argmax(logits, -1).view(-1).detach().cpu().numpy().tolist()
        if not self.use_span or pair_ids is None:
            return pred_ids
        all_pred_labels = {}
        for (batch_id, s_id, e_id), pred_id in zip(pair_ids, pred_ids):
            pred_label = self.span_labels[pred_id]
            if pred_label != "O":
                pred_label = pred_label[2:]
                if e_id == s_id or f"I-{pred_label}" not in self.labels:
                    pred_label = f"B-{pred_label}"
                else:
                    pred_label = f"I-{pred_label}"
            if batch_id not in all_pred_labels:
                all_pred_labels[batch_id] = defaultdict(list)
            for id in range(s_id, e_id+1):
                all_pred_labels[batch_id][id].append(pred_label)
        all_pred_ids = []
        for batch_id, gold_ids in enumerate(label_ids):
            pred_labels = all_pred_labels[batch_id]
            new_pred_ids = []
            prev_label = "O"
            for i, _ in enumerate(gold_ids):
                if i in pred_labels:
                    _pred_labels = pred_labels[i]
                    if prev_label == "O":
                        _pred_labels = [label for label in _pred_labels if not label.startswith("I-")]
                    else:
                        _pred_labels = [label for label in _pred_labels if label in ["O", f"B-{prev_label}", f"I-{prev_label}"]]
                    _pred_labels = [label for label in _pred_labels if label != "O"]
                    if len(_pred_labels) == 0:
                        _pred_labels = ["O"]
                    new_pred_label = Counter(_pred_labels).most_common(1)[0][0]
                    new_pred_ids.append(self.labels.index(new_pred_label))
                    prev_label = new_pred_label[2:] if new_pred_label != "O" else "O"
                else:
                    new_pred_ids.append(-100)
            all_pred_ids += new_pred_ids
        return all_pred_ids

    def prepare_inputs(self, features, label_ids):
        inputs = []
        all_label_ids = []
        pair_ids = []
        for batch_id, (_features, _norm_ids) in enumerate(zip(features, label_ids)):
            s_span_ids = []
            e_span_ids = []
            distance_ids = []
            _label_ids = []
            for s_id, s_norm_id in enumerate(_norm_ids):
                for e_id, e_norm_id in enumerate(_norm_ids):
                    if s_norm_id == -100 or e_norm_id == -100 or s_id > e_id:
                        continue
                    distance = len([id for id in _norm_ids[s_id:e_id+1] if id != -100]) - 1
                    if distance >= self.max_span_distance:
                        continue
                    s_norm_label = self.labels[s_norm_id]
                    e_norm_label = self.labels[e_norm_id]
                    label_id = self.span_labels.index("O")
                    if s_norm_label != "O" and e_norm_label != "O":
                        if s_norm_label[2:] == e_norm_label[2:]:
                            label_id = self.span_labels.index("B-" + s_norm_label[2:])
                    s_span_ids.append(s_id)
                    e_span_ids.append(e_id)
                    distance_ids.append(distance)
                    _label_ids.append(label_id)
                    pair_ids.append((batch_id, s_id, e_id))
            s_features = _features[s_span_ids]
            e_features = _features[e_span_ids]
            d_deatures = self.distance_emb(torch.LongTensor(distance_ids).to(_features.device))
            _inputs = torch.concat((s_features, e_features, d_deatures), -1)
            inputs.append(_inputs)
            all_label_ids.append(torch.LongTensor(_label_ids).to(_features.device))
        return inputs, all_label_ids, pair_ids

    def forward(self, features, label_ids):
        if not self.use_span:
            logits = self.decoder(features)
            if self.training:
                label_ids = label_ids.view(-1)
                norm_loss = self.criterion(logits.view(label_ids.shape[0], -1), label_ids)
                return logits, norm_loss
            else:
                pred_ids = self.generate_pred_label_ids(logits, label_ids)
                return logits, pred_ids
        inputs, all_label_ids, pair_ids = self.prepare_inputs(features, label_ids)
        logits = self.decoder(torch.concat(inputs))
        norm_loss = self.criterion(logits, torch.concat(all_label_ids))
        if not self.training:
            pred_ids = self.generate_pred_label_ids(logits, label_ids, pair_ids)
            return logits, pred_ids
        return logits, norm_loss


class Model(nn.Module):

    def __init__(self, model_config, norm_labels, punc_labels, model_mode):
        super().__init__()
        self.bert = get_model(model_config)
        self.model_mode = model_mode
        self.norm_labels = norm_labels
        self.punc_labels = punc_labels
        emb_len = self.bert.config.hidden_size
        if model_mode == "norm_only":
            self.norm_decoder = SpanDecoder(emb_len, norm_labels)
        elif model_mode == "punc_only":
            self.punc_decoder = SpanDecoder(emb_len, punc_labels)
        elif model_mode == "norm_to_punc":
            self.norm_decoder = SpanDecoder(emb_len, norm_labels)
            self.punc_decoder = SpanDecoder(emb_len, punc_labels, len(norm_labels))
        elif model_mode == "punc_to_norm":
            self.punc_decoder = SpanDecoder(emb_len, punc_labels)
            self.norm_decoder = SpanDecoder(emb_len, norm_labels, len(punc_labels))
    
    @classmethod
    def from_config(cls, model_config, norm_labels, punc_labels, model_mode):
        model_config = read_json(model_config)["pretrained_model"]
        return cls(model_config, norm_labels, punc_labels, model_mode)

    def forward(self, input_ids, mask_ids, norm_ids=None, punc_ids=None):
        features = self.bert(input_ids, mask_ids)[0]
        if self.model_mode == "norm_only":
            if self.training:
                features.register_hook(lambda grad: grad * 0.01)
            norm_logits, norm_loss = self.norm_decoder(features, norm_ids)
            punc_logits, punc_loss = None, torch.tensor(0.0)
        elif self.model_mode == "punc_only":
            if self.training:
                features.register_hook(lambda grad: grad * 0.01)
            punc_logits = self.punc_decoder(features, punc_ids)
            norm_logits, norm_loss = None, torch.tensor(0.0)
        elif self.model_mode == "norm_to_punc":
            norm_logits, norm_loss = self.norm_decoder(features, norm_ids)
            punc_logits, punc_loss = self.punc_decoder(torch.concat((features, norm_logits), -1), punc_ids)
        elif self.model_mode == "punc_to_norm":
            punc_logits, punc_loss = self.punc_decoder(features, punc_ids)
            norm_logits, norm_loss = self.norm_decoder(torch.concat((features, punc_logits), -1), norm_ids)
        return norm_loss, punc_loss