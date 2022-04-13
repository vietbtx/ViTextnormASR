python main.py --mode nojoint --n-blocks 0 --n-tokens 0 --bert-config configs/config.velectra.json
python main.py --mode norm_to_punc --n-blocks 0 --n-tokens 0 --bert-config configs/config.velectra.json
python main.py --mode punc_to_norm --n-blocks 0 --n-tokens 0 --bert-config configs/config.velectra.json

python main.py --mode nojoint --n-blocks 0 --n-tokens 50 --bert-config configs/config.velectra.json
python main.py --mode norm_to_punc --n-blocks 0 --n-tokens 50 --bert-config configs/config.velectra.json
python main.py --mode punc_to_norm --n-blocks 0 --n-tokens 50 --bert-config configs/config.velectra.json
