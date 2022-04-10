from network.trainer import train
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters for Vietnamese Text Normalization')
    parser.add_argument('--data-config', type=str, default="configs/config.norm.json")
    parser.add_argument('--bert-config', type=str, default="configs/config.velectra.json")
    parser.add_argument('--mode', type=str, default="nojoint")
    parser.add_argument('--extend-tokens', action='store_true', default=False)
    args = parser.parse_args()

    train(
        args.data_config,
        args.bert_config,
        args.mode,
        extend_tokens=args.extend_tokens
    )
