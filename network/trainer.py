from time import time
import torch
from tqdm import tqdm
from .model import Model
from utils.data import Data
from utils.utils import get_score_metric
from tensorboardX import SummaryWriter

try:
    from apex import amp
except:
    print("Skip loading apex library")
    amp = None


def init_default_optimizer(model, bert_lr, lr, bert_weight_decay=0.05, adam_epsilon=1e-8):
    optimizer_grouped_parameters = []
    for n, p in model.named_parameters():
        optimizer_params = {"params": p}
        if n.startswith("bert.bert."):
            optimizer_params["lr"] = bert_lr
            if any(x in n for x in ['bias', 'LayerNorm.weight']):
                optimizer_params["weight_decay"] = 0
            else:
                optimizer_params["weight_decay"] = bert_weight_decay
        else:
            optimizer_params["lr"] = lr
        optimizer_grouped_parameters.append(optimizer_params)
    return torch.optim.AdamW(optimizer_grouped_parameters, eps=adam_epsilon)

def prefix_label(label):
    if label != "O" and not label.startswith("B-") and not label.startswith("I-"):
        label = "B-" + label
    return label

def generate_eval(preds, labels, label_dict):
    pred_labels = []
    goal_labels = []
    for pred_id, goal_id in zip(preds, labels):
        if goal_id != -100:
            pred_label = prefix_label(label_dict[pred_id])
            goal_label = prefix_label(label_dict[goal_id])
            pred_labels.append(pred_label)
            goal_labels.append(goal_label)
    return pred_labels, goal_labels
    

def evaluate(model:Model, data, mode, writer, epoch):
    data_loader = data.dev_loader if mode == "dev" else data.test_loader
    model.eval()
    with torch.no_grad():
        norm_pred_labels, norm_goal_labels = [], []
        punc_pred_labels, punc_goal_labels = [], []
        for input_ids, mask_ids, norm_ids, punc_ids in tqdm(data_loader, f"Evaluating {mode}", leave=False):
            norm_preds, punc_preds = model(input_ids, mask_ids, norm_ids, punc_ids)
            if norm_preds is not None and isinstance(norm_preds, list):
                norm_labels = norm_ids.view(-1).detach().cpu().numpy()
                norm_preds, norm_labels = generate_eval(norm_preds, norm_labels, data.norm_labels)
                norm_pred_labels.append(norm_preds)
                norm_goal_labels.append(norm_labels)
            if punc_preds is not None  and isinstance(punc_preds, list):
                punc_labels = punc_ids.view(-1).detach().cpu().numpy()
                punc_preds, punc_labels = generate_eval(punc_preds, punc_labels, data.punc_labels)
                punc_pred_labels.append(punc_preds)
                punc_goal_labels.append(punc_labels)
    report_func = get_score_metric("classification_report")
    
    norm_scores = {}
    norm_f1 = 0
    if len(norm_pred_labels) > 0:
        norm_scores = report_func(norm_goal_labels, norm_pred_labels, output_dict=True)
        for name, score in norm_scores.items():
            writer.add_scalar(f"norm_{mode}/{name}", score["f1-score"], epoch)
        norm_f1 = norm_scores["micro avg"]["f1-score"]
    
    punc_scores = {}
    punc_f1 = 0
    if len(punc_pred_labels) > 0:
        punc_scores = report_func(punc_goal_labels, punc_pred_labels, output_dict=True)
        for name, score in punc_scores.items():
            writer.add_scalar(f"punc_{mode}/{name}", score["f1-score"], epoch)
        punc_f1 = punc_scores["micro avg"]["f1-score"]

    return norm_f1, punc_f1, norm_scores, punc_scores

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def write_log(writer, norm_loss, punc_loss, phase_name, step, total_step, epoch, global_step, scheduler):
    norm_loss = norm_loss.item()
    punc_loss = punc_loss.item()
    end = "\n" if step % (total_step//4) == 0 else "\r"
    print(f"Phase: {phase_name} - epoch: {epoch} - step: {step+1}/{total_step} - loss: {norm_loss:.5f}/{punc_loss:.5f}", end=end)
    if norm_loss > 0:
        writer.add_scalar("norm_loss", norm_loss, global_step)
    if punc_loss > 0:
        writer.add_scalar("punc_loss", punc_loss, global_step)
    writer.add_scalar('learning_rate', scheduler.optimizer.param_groups[0]["lr"], global_step)

def train(data_config, model_config, model_mode, extend_tokens):
    phase_name = f"{model_mode}+SC" if extend_tokens else model_mode
    data = Data.from_config(data_config, model_config, extend_tokens)
    writer = SummaryWriter(f"{data.tensorboard_dir}/{phase_name}")
    model = Model.from_config(model_config, data.norm_labels, data.punc_labels, model_mode)
    model.to(data.device)
    optimizer = init_default_optimizer(model, data.learning_rate, 0.001)
    if amp is not None and data.device != "cpu":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    total_step = len(data.train_loader)
    global_step = 0
    best_norm_f1_score = 0
    best_punc_f1_score = 0
    n_epochs = data.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, total_step//8, n_epochs*total_step)

    for epoch in range(n_epochs):
        t0 = time()
        torch.cuda.empty_cache()
        model.train()
        for step, batch in enumerate(data.train_loader):
            global_step += 1
            norm_loss, punc_loss = model(*batch)
            loss = norm_loss + punc_loss
            write_log(writer, norm_loss, punc_loss, phase_name, step, total_step, epoch, global_step, scheduler)
            if amp is not None and data.device != "cpu":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            if step > 10:
                break
        writer.add_text(f"time/train", str(time() - t0), epoch)
        
        t0 = time()
        dev_norm_f1, dev_punc_f1, _, _ = evaluate(model, data, "dev", writer, epoch)
        print(f"\nDev score: {dev_norm_f1:.5f}/{dev_punc_f1:.5f}")
        writer.add_text(f"time/dev", str(time() - t0), epoch)
        
        t0 = time()
        test_norm_f1, test_punc_f1, norm_scores, punc_scores = evaluate(model, data, "test", writer, epoch)
        print(f"Test score: {test_norm_f1:.5f}/{test_punc_f1:.5f}")
        writer.add_text(f"time/test", str(time() - t0), epoch)

        if dev_norm_f1 > best_norm_f1_score:
            best_norm_f1_score = dev_norm_f1
            print(f"Best norm score: dev = {dev_norm_f1:.5f} & test = {test_norm_f1:.5f}")
            writer.add_text(f"test_norm", str(norm_scores), epoch)
            writer.add_scalar(f"F1_score/norm", test_norm_f1, epoch)
        
        if dev_punc_f1 > best_punc_f1_score:
            best_punc_f1_score = dev_punc_f1
            print(f"Best punc score: dev = {dev_punc_f1:.5f} & test = {test_punc_f1:.5f}")
            writer.add_text(f"test_punc", str(punc_scores), epoch)
            writer.add_scalar(f"F1_score/punc", test_punc_f1, epoch)