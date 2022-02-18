python main.py --mode norm_only --extend-tokens --bert-config configs/config.velectra.json
python main.py --mode punc_only --extend-tokens --bert-config configs/config.velectra.json
python main.py --mode norm_to_punc --extend-tokens --bert-config configs/config.velectra.json
python main.py --mode punc_to_norm --extend-tokens --bert-config configs/config.velectra.json

python main.py --mode norm_only --bert-config configs/config.velectra.json
python main.py --mode punc_only --bert-config configs/config.velectra.json
python main.py --mode norm_to_punc --bert-config configs/config.velectra.json
python main.py --mode punc_to_norm --bert-config configs/config.velectra.json