import torch
from tqdm import tqdm
from utils.data import Data
from .model import AdapterModel
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
        if n.startswith("bert.bert."):
            optimizer_params["lr"] = bert_lr
            if any(x in n for x in ['bias', 'LayerNorm.weight']):
                optimizer_params["weight_decay"] = 0
            else:
                optimizer_params["weight_decay"] = bert_weight_decay
        else:
            optimizer_params["lr"] = lr
        optimizer_grouped_parameters.append(optimizer_params)
    return AdamW(optimizer_grouped_parameters, eps=adam_epsilon)


def evaluate(model:AdapterModel, data_loader, norm_dict, punc_dict):
    model.eval()
    with torch.no_grad():
        pred_labels, goal_labels = [], []
        for input_ids, mask_ids, norm_ids, punc_ids in tqdm(data_loader, "Evaluating", leave=False):
            logits = model(input_ids, mask_ids)
            mode = "norm" if model.model_mode in ["norm_only", "punc_to_norm"] else "punc"
            labels = norm_ids if mode == "norm" else punc_ids
            labels = labels.view(-1).detach().cpu().numpy()
            preds = torch.argmax(logits, -1).view(-1).detach().cpu().numpy()
            pred_labels.append([])
            goal_labels.append([])
            for pred_id, goal_id in zip(preds, labels):
                if goal_id != -100:
                    pred_label = norm_dict[pred_id] if mode == "norm" else punc_dict[pred_id]
                    goal_label = norm_dict[goal_id] if mode == "norm" else punc_dict[goal_id]
                    if mode == "punc":
                        pred_label = "B-" + pred_label if pred_label != "O" else pred_label
                        goal_label = "B-" + goal_label if goal_label != "O" else goal_label
                    pred_labels[-1].append(pred_label)
                    goal_labels[-1].append(goal_label)
    report_func = get_score_metric("classification_report")
    score = report_func(goal_labels, pred_labels, output_dict=True)
    return score, mode


def train(data_config, model_config, model_mode, extend_tokens):
    phase_name = f"{model_mode}+SC" if extend_tokens else model_mode
    data = Data.from_config(data_config, model_config, extend_tokens)
    writer = SummaryWriter(f"{data.tensorboard_dir}/{phase_name}")
    mode = "punc" if model_mode in ["norm_only", "punc_to_norm"] else "norm"
    adapter_path = f"./pretrained_models/{data.model_name}/{mode}_only/model_{mode}"
    model = AdapterModel.from_config(model_config, data.norm_labels, data.punc_labels, model_mode, adapter_path)
    model.to(data.device)
    optimizer = init_default_optimizer(model, data.learning_rate, 0.001)
    if amp is not None and data.device != "cpu":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    total_step = len(data.train_loader)
    global_step = 0
    best_f1_score = 0
    n_epochs = data.n_epochs
    scheduler = linear_schedule(optimizer, num_warmup_steps=total_step//8, num_training_steps=n_epochs*total_step)
    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        model.train()
        for step, batch in enumerate(data.train_loader):
            global_step += 1
            loss = model(*batch)
            loss_value = loss.item()
            end = "\n" if step % (total_step//4) == 0 else "\r"
            print(f"Phase: {phase_name} - epoch: {epoch} - step: {step+1}/{total_step} - loss: {loss_value:.5f}", end=end)
            writer.add_scalar("loss", loss_value, global_step)
            writer.add_scalar('learning_rate', scheduler.optimizer.param_groups[0]["lr"], global_step)
            if amp is not None and data.device != "cpu":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        scores, mode = evaluate(model, data.dev_loader, data.norm_labels, data.punc_labels)
        for name, score in scores.items():
            writer.add_scalar(f"dev_{mode}/{name}", score["f1-score"], epoch)
        dev_f1 = scores["micro avg"]["f1-score"]
        print(f"\nDev {mode} score: {dev_f1:.5f}")

        scores, mode = evaluate(model, data.test_loader, data.norm_labels, data.punc_labels)
        for name, score in scores.items():
            writer.add_scalar(f"test_{mode}/{name}", score["f1-score"], epoch)
        test_f1 = scores["micro avg"]["f1-score"]
        print(f"Test {mode} score: {test_f1:.5f}")

        if dev_f1 > best_f1_score:
            best_f1_score = dev_f1
            print(f"Best {mode} score: dev = {dev_f1:.5f} & test = {test_f1:.5f}")
            writer.add_text(f"test_{mode}", str(scores), epoch)
            writer.add_scalar(f"F1_score/{mode}", test_f1, epoch)
            model.bert.save_pretrained(f"./pretrained_models/{data.model_name}/{phase_name}/model_{mode}")
            if model_mode in ["norm_only", "punc_only"]:
                model.bert.save_adapter(f"./pretrained_models/{data.model_name}/{phase_name}/model_{mode}", f"ner_{mode}")
        