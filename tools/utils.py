import csv
from typing import Any, Dict, List
import unicodedata as ud
import random
from bs4 import BeautifulSoup


def write_csv(file_name: str, data: Any):
    with open(file_name, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "category", "page_numb", "title", "description"])
        for url, (category, page_numb, title, description) in data.items():
            writer.writerow([url, category, page_numb, title, description])


def remove_soup_content(soup: BeautifulSoup, *args, **kwargs):
    if soup is not None:
        for content in soup.find_all(*args, **kwargs):
            content.replace_with("")


def normalize(text: str):
    lines = text.splitlines()
    lines = [ud.normalize("NFKC", s) for s in lines]
    lines = [" ".join(s.split()) for s in lines]
    lines = [s for s in lines if len(s) > 0]
    text = "\n".join(lines)
    return text


def is_correct_integral(parts: List, split_char: str = "."):
    if len(parts) < 3:
        return len(parts) == 1 and parts[0].isdigit()

    for i, part in enumerate(parts):
        if i % 2 == 0:
            if not part.isdigit():
                return False
            elif i == 0 and len(part) > 3:
                return False
            elif i > 0 and len(part) != 3:
                return False
        elif part != split_char:
            return False
    return True


def is_correct_fractional(parts: List):
    if len(parts) == 0:
        return True
    if parts[0] not in ".,":
        return False
    return len(parts) == 2 and parts[1].isdigit()


def index(items: List, value: Any):
    for i, part in enumerate(items):
        if part == value:
            return i
    return len(items)

def is_number(text: str, func: Any = int):
    if func != int:
        result = is_number(text, int)
        if result:
            return result
    try:
        return func(text)
    except:
        pass

def choice(options: Dict, k: int = 1, otherwise: Any = ""):
    arr = list(options.keys())
    weights = [options[k] for k in arr]
    otherwise_weight = 1 - sum(weights)
    arr.append(otherwise)
    weights.append(otherwise_weight)
    result = random.choices(arr, weights=weights, k=k)
    return result[0] if k == 1 else result

def append_or_add(arr: List, data: str, label: str):
    if len(arr) == 0 or arr[-1][1] != label:
        arr.append([data, label])
    else:
        arr[-1][0] += " " + data
        arr[-1][0] = arr[-1][0].strip()