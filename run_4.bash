#!/bin/bash
#PBS -N ViTextnorm_4
#PBS -q gpuvolta
#PBS -P aw84
#PBS -l walltime=24:00:00
#PBS -l wd
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l jobfs=16GB
#PBS -l storage=gdata/aw84

module load pytorch/1.9.0
source /home/582/tn0796/NLP/venv/bin/activate

cd /home/582/tn0796/NLP/ViTextnormASR/

python main.py --fold-id 4 --mode nojoint --n-blocks 0 --n-tokens 0
python main.py --fold-id 4 --mode norm_to_punc --n-blocks 0 --n-tokens 0
python main.py --fold-id 4 --mode punc_to_norm --n-blocks 0 --n-tokens 0

python main.py --fold-id 4 --mode nojoint --n-blocks 0 --n-tokens 50
python main.py --fold-id 4 --mode norm_to_punc --n-blocks 0 --n-tokens 50
python main.py --fold-id 4 --mode punc_to_norm --n-blocks 0 --n-tokens 50

python main.py --fold-id 4 --mode nojoint --n-blocks 0 --n-tokens 0 --use-biaffine
python main.py --fold-id 4 --mode norm_to_punc --n-blocks 0 --n-tokens 0 --use-biaffine
python main.py --fold-id 4 --mode punc_to_norm --n-blocks 0 --n-tokens 0 --use-biaffine

python main.py --fold-id 4 --mode nojoint --n-blocks 0 --n-tokens 50 --use-biaffine
python main.py --fold-id 4 --mode norm_to_punc --n-blocks 0 --n-tokens 50 --use-biaffine
python main.py --fold-id 4 --mode punc_to_norm --n-blocks 0 --n-tokens 50 --use-biaffine