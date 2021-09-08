from tools.newscrawler import VNExpressCrawler
from tools.utils import write_csv
from tools.numb2text import Numb2Text
from p_tqdm import p_umap
import random
from sklearn.model_selection import KFold
import os


def is_valid_paragraph(text):
    if len(text) < 10:
        return False
    if not text[0].isdigit() and not text[0].isalpha() and not text[0] in '"\'':
        return False
    return text[-1] in ".?!"

def create_labels(line):
    if not is_valid_paragraph(line):
        return
    data = Numb2Text.read(line)
    result = []
    for text, label in data:
        if label == "O":
            for word in text.split():
                result.append([word, label, "O"])
        elif label == "PUNC":
            result[-1][2] = text
        else:
            result.append([text, label, "O"])
    return result

def write_data(file_name, data):
    with open(file_name, "w", encoding="utf-8") as f:
        for paragraph in data:
            for text, label_1, label_2 in paragraph:
                data = []
                is_start = True
                for word in text.split():
                    if label_1 != "O":
                        data.append([word, f"B-{label_1}" if is_start else f"I-{label_1}", "O"])
                    else:
                        data.append([word, label_1, "O"])
                    is_start = False
                if len(data) > 0:
                    data[-1][2] = label_2
                    for word, label_1, label_2 in data:
                        f.write(f"{word}\t{label_1}\t{label_2}\n")
            # f.write("\n")

def read_content(url):
    data = []
    Numb2Text.reset()
    crawler = VNExpressCrawler()
    try:
        text = crawler.read_content(url)
        if text is not None:
            for line in text.splitlines():
                labels = create_labels(line)
                if labels:
                    data.append(labels)
    except:
        pass
    return url, data

if __name__=="__main__":
    print("Scanning VnExpress ...")
    data = {}
    for result in p_umap(VNExpressCrawler.from_category, VNExpressCrawler.CATEGORIES):
        data.update(result)
    all_urls = list(data.keys())
    print(f"Scanned {len(all_urls)} articles")
    all_urls = all_urls[:6000]
    random.shuffle(all_urls)

    print("Reading content ...")
    content = {}
    for url, result in p_umap(read_content, all_urls):
        if len(result) > 0:
            content[url] = result

    all_urls = list(content.keys())

    print(f"Selected {len(all_urls)} articles")

    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(all_urls)

    for fold_id, (train_index, test_index) in enumerate(kf.split(all_urls)):
        print("Fold:", fold_id)
        os.makedirs("dataset", exist_ok=True)
        os.makedirs(f"dataset/fold_{fold_id}", exist_ok=True)
        train_urls = [all_urls[x] for x in train_index]
        test_urls = [all_urls[x] for x in test_index]

        print(f"    - Train:\t{len(train_urls)} articles")
        print(f"    - Test:\t{len(test_urls)} articles")

        write_csv(f"dataset/fold_{fold_id}/train_metadata.csv", {url: data[url] for url in train_urls})
        write_csv(f"dataset/fold_{fold_id}/test_metadata.csv", {url: data[url] for url in test_urls})
        
        train_data = []
        test_data = []
        for url, result in content.items():
            if url in train_urls:
                train_data += result
            elif url in test_urls:
                test_data += result
        write_data(f"dataset/fold_{fold_id}/train.txt", train_data)
        write_data(f"dataset/fold_{fold_id}/test.txt", test_data)
