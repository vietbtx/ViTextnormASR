from network.trainer import train
import sys


if __name__=="__main__":
    fold_id = int(sys.argv[1])
    
    config = "configs/config.norm.json"
    mbert = "configs/config.mbert.json"
    vibert = "configs/config.vibert.json"
    velectra = "configs/config.velectra.json"

    train_data = [
        ("nojoint", 4, 50, True),
        ("norm_to_punc", 4, 50, True),
        ("punc_to_norm", 4, 50, True),

        ("nojoint", 0, 50, True),
        ("norm_to_punc", 0, 50, True),
        ("punc_to_norm", 0, 50, True),

        ("nojoint", 0, 0, True),
        ("norm_to_punc", 0, 0, True),
        ("punc_to_norm", 0, 0, True),

        ("nojoint", 4, 50, False),
        ("norm_to_punc", 4, 50, False),
        ("punc_to_norm", 4, 50, False),

        ("nojoint", 0, 50, False),
        ("norm_to_punc", 0, 50, False),
        ("punc_to_norm", 0, 50, False),

        ("nojoint", 0, 0, False),
        ("norm_to_punc", 0, 0, False),
        ("punc_to_norm", 0, 0, False),
    ]
    
    for mode, n_blocks, n_tokens, biaffine in train_data:
        try:
            train(config, velectra, mode, fold_id, n_blocks=n_blocks, n_tokens=n_tokens, biaffine=biaffine)
        except Exception as e:
            print("Error:", e)