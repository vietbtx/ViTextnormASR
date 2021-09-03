import torch
from utils.data import Data
from .model import BERTModel
from utils.utils import get_score_metric
from transformers import AdamW, get_linear_schedule_with_warmup as linear_schedule
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
        if n.startswith("bert."):
            optimizer_params["lr"] = bert_lr
            if any(x in n for x in ['bias', 'LayerNorm.weight']):
                optimizer_params["weight_decay"] = 0
            else:
                optimizer_params["weight_decay"] = bert_weight_decay
        else:
            optimizer_params["lr"] = lr
        optimizer_grouped_parameters.append(optimizer_params)
    return AdamW(optimizer_grouped_parameters, eps=adam_epsilon)


def evaluate(model, data_loader, norm_dict, punc_dict):
    model.eval()
    with torch.no_grad():
        pred_labels = {"norm": [], "punc": []}
        goal_labels = {"norm": [], "punc": []}
        for input_ids, mask_ids, norm_ids, punc_ids, next_blocks, prev_blocks in data_loader:
            norm_logits, punc_logits = model(input_ids, mask_ids, next_blocks=next_blocks, prev_blocks=prev_blocks)
            norm_labels = norm_ids.view(-1).detach().cpu().numpy()
            punc_labels = punc_ids.view(-1).detach().cpu().numpy()
            pred_norm_labels = torch.argmax(norm_logits, -1).view(-1).detach().cpu().numpy()
            pred_punc_labels = torch.argmax(punc_logits, -1).view(-1).detach().cpu().numpy()
            pred_labels["norm"].append([])
            goal_labels["norm"].append([])
            for pred_id, goal_id in zip(pred_norm_labels, norm_labels):
                if goal_id != -100:
                    pred_label = norm_dict[pred_id]
                    goal_label = norm_dict[goal_id]
                    pred_labels["norm"][-1].append(pred_label)
                    goal_labels["norm"][-1].append(goal_label)
            pred_labels["punc"].append([])
            goal_labels["punc"].append([])
            for pred_id, goal_id in zip(pred_punc_labels, punc_labels):
                if goal_id != -100:
                    pred_label = punc_dict[pred_id]
                    goal_label = punc_dict[goal_id]
                    pred_label = "B-" + pred_label if pred_label != "O" else pred_label
                    goal_label = "B-" + goal_label if goal_label != "O" else goal_label
                    pred_labels["punc"][-1].append(pred_label)
                    goal_labels["punc"][-1].append(goal_label)
            break
    report_func = get_score_metric("classification_report")
    norm_score = report_func(goal_labels["norm"], pred_labels["norm"], output_dict=True)
    punc_score = report_func(goal_labels["punc"], pred_labels["punc"], output_dict=True)
    return norm_score, punc_score


def train(data_config, model_config, model_mode, n_blocks=0, n_tokens=0):
    data = Data.from_config(data_config, model_config, n_blocks, n_tokens)
    writer = SummaryWriter(f"{data.tensorboard_dir}/{model_mode}/{n_blocks}-{n_tokens}")
    model = BERTModel.from_config(model_config, data.norm_labels, data.punc_labels, data.lstm_dim, model_mode)
    model.to(data.device)
    optimizer = init_default_optimizer(model, data.learning_rate, 0.001)
    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    total_step = len(data.train_loader)
    global_step = 0
    best_f1_scores = {"norm": 0, "punc": 0}
    if model_mode == "norm_to_punc":
        phases = ["nojoint", "norm", "punc"]
    elif model_mode == "punc_to_norm":
        phases = ["nojoint", "punc", "norm"]
    else:
        phases = ["nojoint"]
    for phase in phases:
        n_epochs = data.n_epochs
        scheduler = linear_schedule(optimizer, num_warmup_steps=total_step//8, num_training_steps=n_epochs*total_step)
        for epoch in range(n_epochs):
            model.train()
            for step, batch in enumerate(data.train_loader):
                global_step += 1
                if phase == "nojoint":
                    norm_loss, punc_loss = model(*batch)
                    loss = norm_loss + punc_loss
                    norm_loss = norm_loss.item()
                    punc_loss = punc_loss.item()
                elif phase == "norm":
                    norm_loss, punc_loss = model(*batch, phase=phase)
                    loss = norm_loss
                    norm_loss = norm_loss.item()
                elif phase == "punc":
                    norm_loss, punc_loss = model(*batch, phase=phase)
                    loss = punc_loss
                    punc_loss = punc_loss.item()
                
                end = "\n" if step % (total_step//8) == 0 else "\r"
                phase_name = f"{phase}/{model_mode}/{n_blocks}-{n_tokens}"
                print(f"Phase: {phase_name} - epoch: {epoch} - step: {step+1}/{total_step} - loss: {norm_loss:.5f}/{punc_loss:.5f}", end=end)
                if norm_loss > 0:
                    writer.add_scalar("loss/norm", norm_loss, global_step)
                if punc_loss > 0:
                    writer.add_scalar("loss/punc", punc_loss, global_step)
                writer.add_scalar('learning_rate', scheduler.optimizer.param_groups[0]["lr"], global_step)
                if amp is not None:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if model_mode == "nojoint" or phase in ["norm", "punc"]:
                dev_norm_score, dev_punc_score = evaluate(model, data.dev_loader, data.norm_labels, data.punc_labels)
                test_norm_score, test_punc_score = evaluate(model, data.test_loader, data.norm_labels, data.punc_labels)

            if phase == "norm" or model_mode == "nojoint":
                for name, score in dev_norm_score.items():
                    writer.add_scalar(f"dev_norm/{name}", score["f1-score"], epoch)
                for name, score in test_norm_score.items():
                    writer.add_scalar(f"test_norm/{name}", score["f1-score"], epoch)
                dev_f1_norm = dev_norm_score["micro avg"]["f1-score"]
                test_f1_norm = test_norm_score["micro avg"]["f1-score"]
                if dev_f1_norm > best_f1_scores["norm"]:
                    best_f1_scores["norm"] = dev_f1_norm
                    print(f"\nBest F1 norm: {dev_f1_norm:.5f}/{test_f1_norm:.5f}")
                    writer.add_text("dev_norm", str(dev_norm_score), epoch)
                    writer.add_text("test_norm", str(test_norm_score), epoch)
                
            if phase == "punc" or model_mode == "nojoint":
                for name, score in dev_punc_score.items():
                    writer.add_scalar(f"dev_punc/{name}", score["f1-score"], epoch)
                for name, score in test_punc_score.items():
                    writer.add_scalar(f"test_punc/{name}", score["f1-score"], epoch)
                dev_f1_punc = dev_punc_score["micro avg"]["f1-score"]
                test_f1_punc = test_punc_score["micro avg"]["f1-score"]
                if dev_f1_punc > best_f1_scores["punc"]:
                    best_f1_scores["punc"] = dev_f1_punc
                    print(f"\nBest F1 punc: {dev_f1_punc:.5f}/{test_f1_punc:.5f}")
                    writer.add_text("test_punc", str(test_punc_score), epoch)
                    writer.add_text("dev_punc", str(dev_punc_score), epoch)
