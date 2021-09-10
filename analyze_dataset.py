

from collections import Counter
import csv


def read_data(file_name):
    data = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            data.append(parts)
    return data

def read_csv(file_name):
    print("reading ...", file_name)
    data = []
    with open(file_name, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for parts in reader:
            if len(parts) > 0:
                data.append(parts)
    return data[1:]

def analyze(name):
    data = read_data(f"dataset/fold_0/{name}.txt")
    print(f"Number of tokens: {len(data)}")
    labels_1 = [item[1] for item in data]
    labels_1 = Counter(labels_1)
    print("Textnorm labels:", labels_1)
    for label in ["B-CAP", "B-ALLCAP", "B-NUMB", "O"]:
        print(f"{label:>16}: {labels_1[label]}")
    labels_2 = [item[2] for item in data]
    labels_2 = Counter(labels_2)
    print("Punctuation labels:", labels_2)
    for label in ["PERIOD", "COMMA", "QMARK", "EMARK", "COLON", "SCOLON", "DASH", "O"]:
        print(f"{label:>16}: {labels_2[label]}")
    
    data = read_csv(f"dataset/fold_0/{name}_metadata.csv")
    print("pages:", len(data))
    min_url = min(data, key=lambda x: int(x[2][1:]))[0]
    print("min_url:", min_url)
    max_url = max(data, key=lambda x: int(x[2][1:]))[0]
    print("max_url:", max_url)
    print(" "*10 + "-"*20)


if __name__ == "__main__":
    print("train data")
    analyze("train")
    print("test data")
    analyze("test")

