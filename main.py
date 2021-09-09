from network.trainer import train
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters for Vietnamese Text Normalization')
    parser.add_argument('--data-config', type=str, default="configs/config.norm.json")
    parser.add_argument('--bert-config', type=str, default="configs/config.velectra.json")
    parser.add_argument('--fold-id', type=int, default=0)
    parser.add_argument('--mode', type=str, default="nojoint")
    parser.add_argument('--n-blocks', type=int, default=0)
    parser.add_argument('--n-tokens', type=int, default=0)
    parser.add_argument('--use-biaffine', action='store_true', default=False)
    
    args = parser.parse_args()

    try:
        train(
            args.data_config,
            args.bert_config,
            args.mode,
            args.fold_id,
            n_blocks=args.n_blocks,
            n_tokens=args.n_tokens,
            biaffine=args.use_biaffine
        )
    except Exception as e:
        print("Error:", e)
