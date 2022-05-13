from network.trainer import train
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters for Vietnamese Text Normalization')
    parser.add_argument('--data-config', type=str, default="configs/config.norm.json")
    parser.add_argument('--bert-config', type=str, default="configs/config.velectra.json")
    parser.add_argument('--mode', type=str, default="nojoint")
    parser.add_argument('--surrounding-context', action='store_true')
    parser.add_argument('--use-biaffine', action='store_true')
    
    args = parser.parse_args()

    train(
        args.data_config,
        args.bert_config,
        args.mode,
        use_sc=args.surrounding_context,
        biaffine=args.use_biaffine
    )
