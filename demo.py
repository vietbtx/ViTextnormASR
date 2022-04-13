
from collections import Counter, defaultdict


if __name__ == "__main__":
    logs = defaultdict(list)
    with open("dataset/train.conll", "r", encoding="utf-8") as f:
        prev_step = 0
        for step, line in enumerate(f):
            parts = line.split()
            label = parts[1][2:] if parts[1] != "O" else "O"
            logs[label].append(parts[0])
    for label, data in logs.items():
        x = Counter(data).most_common(100)
        print(label,":", sorted([y[0] for y in x]))
