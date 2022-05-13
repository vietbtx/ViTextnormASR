python main.py --mode norm_only --bert-config configs/config.mbert.json
python main.py --mode punc_only --bert-config configs/config.mbert.json
python main.py --mode norm_to_punc --bert-config configs/config.mbert.json
python main.py --mode punc_to_norm --bert-config configs/config.mbert.json

python main.py --mode norm_only --surrounding-context --bert-config configs/config.mbert.json
python main.py --mode punc_only --surrounding-context --bert-config configs/config.mbert.json
python main.py --mode norm_to_punc --surrounding-context --bert-config configs/config.mbert.json
python main.py --mode punc_to_norm --surrounding-context --bert-config configs/config.mbert.json
