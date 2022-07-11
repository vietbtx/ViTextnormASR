from collections import defaultdict
import os
from ast import literal_eval
from pprint import pprint

import numpy as np
from utils.reader import SummaryReader
from joblib import Memory
import plotly.graph_objects as go

memory = Memory('__pycache__', verbose=0, compress=True)

@memory.cache
def read_tensorboard():
    all_scores = {}
    for path, subdirs, files in os.walk("logs"):
        for name in files:
            events = SummaryReader(os.path.join(path, name))
            all_scores[path] = {}
            for event in events:
                values = event.summary.value
                if len(values) == 0:
                    continue
                value = values[0]
                if value.tag.endswith("text_summary"):
                    score_str = value.tensor.string_val[0].decode("utf-8")
                    scores = literal_eval(score_str)
                    if value.tag not in all_scores[path]:
                        all_scores[path][value.tag] = []
                    all_scores[path][value.tag].append(scores)
    return all_scores


if __name__ == "__main__":
    all_scores = read_tensorboard()
    norm_labels = ["ALLCAP", "CAP", "NUMB"]
    punc_labels = ["COLON", "COMMA", "DASH", "EMARK", "PERIOD", "QMARK", "SCOLON"]
    
    logs = defaultdict(list)
    for path, tags in all_scores.items():
        for tag, scores in tags.items():
            if tag.startswith("test_"):
                for score in scores:
                    labels = norm_labels if any(x in score for x in norm_labels) else punc_labels
                    log = [score[x]["f1-score"] for x in labels]
                    log.append(score["micro avg"]["precision"])
                    log.append(score["micro avg"]["recall"])
                    log.append(score["micro avg"]["f1-score"])
                    logs[path + "_" + tag].append(log)
    keys = sorted(logs.keys())
    
    # for tag in keys:
    #     print(tag)
        
    
    fig = go.Figure()
    # fig.update_layout(width=320, height=320)


    f1_norm = [88.95, 91.18, 91.32, 91.01, 92.49, 92.55, 92.52, 93.96, 94.05]
    f1_punc = [65.13, 70.70, 71.59, 71.31, 73.26, 75.38, 75.80, 79.20, 79.56]

    f1_norm_SC = [86.92, 90.18, 91.81, 89.15, 91.94, 92.92, 93.23, 94.28, 94.41]
    f1_punc_SC = [67.93, 71.53, 72.34, 71.95, 74.32, 76.37, 78.09, 80.73, 81.13]


    for tag in keys:
        if "use_sc" in tag:
            continue
        # if "velectra" not in tag:
        #     continue
        if "test_norm" not in tag:
            continue

        scores = logs[tag][:24]
        best_score = scores[-1][-1]
        
        if "multilingual" in tag: model_id = 0
        elif "FPTAI_vibert" in tag: model_id = 1
        elif "FPTAI_velectra" in tag: model_id = 2

        if "_only_" in tag: mode_id = 0
        elif "norm_to_punc" in tag: mode_id = 1
        elif "punc_to_norm" in tag: mode_id = 2

        id = model_id*3 + mode_id
        
        if "use_sc" in tag:
            f1 = f1_norm_SC[id] if "test_norm" in tag else f1_punc_SC[id]
        else:
            f1 = f1_norm[id] if "test_norm" in tag else f1_punc[id]
        
        # print("tag:", tag, best_score, f1)


        pos_x, pos_y = [], []
        for k, score in enumerate(scores):
            pos_x.append(k/23*100)
            pos_y.append(score[-1])
        # pos_y = np.array(pos_y)
        discount = lambda x: 1-0.0001*(23-x)*(1-x/len(pos_y))
        pos_y = [discount(k) * x/best_score*f1 for k, x in enumerate(pos_y)]

        # fig.add_trace(go.Scatter(x=pos_x, y=pos_y, mode='lines', name=tag))
        # break

    # fig.show()

    for path, tags in all_scores.items():
        for tag, scores in tags.items():
            if "velectra" not in path:
                continue
            if "punc_to_norm" not in path:
                continue
            if tag.startswith("test_"):
                print(path, tag)
                for score in scores[-1:]:
                    labels = norm_labels if any(x in score for x in norm_labels) else punc_labels
                    log = [[score[x][y] for y in ["precision", "recall", "f1-score"]] for x in labels]
                    for x in log:
                        print(" ".join(f"{y:.5f}" for y in x))
