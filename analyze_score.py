import os
from ast import literal_eval
from utils.reader import SummaryReader

if __name__ == "__main__":
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
    
    for path, tags in all_scores.items():
        print(path)
        for tag, scores in tags.items():
            if tag.startswith("test_"):
                score = scores[-1]["micro avg"]["f1-score"]
                print(tag, score)