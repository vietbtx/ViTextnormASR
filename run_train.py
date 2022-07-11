from multiprocessing import Pool
import os
from random import randint
from time import sleep

def run(cmd, use_sleep=True):
    print(cmd)
    if use_sleep:
        sleep(randint(0, 200))
    os.system(cmd)

if __name__ == "__main__":
    all_cmd = [
        "python main.py --mode norm_only --bert-config configs/config.velectra.json",
        "python main.py --mode punc_only --bert-config configs/config.velectra.json",
        "python main.py --mode norm_to_punc --bert-config configs/config.velectra.json",
        "python main.py --mode punc_to_norm --bert-config configs/config.velectra.json",

        "python main.py --mode norm_only --bert-config configs/config.vibert.json",
        "python main.py --mode punc_only --bert-config configs/config.vibert.json",
        "python main.py --mode norm_to_punc --bert-config configs/config.vibert.json",
        "python main.py --mode punc_to_norm --bert-config configs/config.vibert.json",

        "python main.py --mode norm_only --bert-config configs/config.mbert.json",
        "python main.py --mode punc_only --bert-config configs/config.mbert.json",
        "python main.py --mode norm_to_punc --bert-config configs/config.mbert.json",
        "python main.py --mode punc_to_norm --bert-config configs/config.mbert.json",
    ]

    with Pool(3) as p:
        p.map(run, all_cmd)