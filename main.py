from network.trainer import train


if __name__=="__main__":
    mbert = "configs/config.mbert.json"
    vibert = "configs/config.vibert.json"
    velectra = "configs/config.velectra.json"

    train_data = [
        (0, "nojoint", 0, 0, False),
        (0, "norm_to_punc", 4, 50, True),
        (0, "punc_to_norm", 4, 50, True),
        (0, "norm_to_punc", 0, 50, True),
        (0, "punc_to_norm", 0, 50, True),
        (0, "norm_to_punc", 0, 0, True),
        (0, "punc_to_norm", 0, 0, True),
        (0, "norm_to_punc", 4, 50, False),
        (0, "punc_to_norm", 4, 50, False),
        (0, "norm_to_punc", 0, 50, False),
        (0, "punc_to_norm", 0, 50, False),
        (0, "norm_to_punc", 0, 0, False),
        (0, "punc_to_norm", 0, 0, False),

        (1, "nojoint", 0, 0, False),
        (1, "norm_to_punc", 4, 50, True),
        (1, "punc_to_norm", 4, 50, True),
        (1, "norm_to_punc", 0, 50, True),
        (1, "punc_to_norm", 0, 50, True),
        (1, "norm_to_punc", 0, 0, True),
        (1, "punc_to_norm", 0, 0, True),
        (1, "norm_to_punc", 4, 50, False),
        (1, "punc_to_norm", 4, 50, False),
        (1, "norm_to_punc", 0, 50, False),
        (1, "punc_to_norm", 0, 50, False),
        (1, "norm_to_punc", 0, 0, False),
        (1, "punc_to_norm", 0, 0, False),

        (2, "nojoint", 0, 0, False),
        (2, "norm_to_punc", 4, 50, True),
        (2, "punc_to_norm", 4, 50, True),
        (2, "norm_to_punc", 0, 50, True),
        (2, "punc_to_norm", 0, 50, True),
        (2, "norm_to_punc", 0, 0, True),
        (2, "punc_to_norm", 0, 0, True),
        (2, "norm_to_punc", 4, 50, False),
        (2, "punc_to_norm", 4, 50, False),
        (2, "norm_to_punc", 0, 50, False),
        (2, "punc_to_norm", 0, 50, False),
        (2, "norm_to_punc", 0, 0, False),
        (2, "punc_to_norm", 0, 0, False),

        (3, "nojoint", 0, 0, False),
        (3, "norm_to_punc", 4, 50, True),
        (3, "punc_to_norm", 4, 50, True),
        (3, "norm_to_punc", 0, 50, True),
        (3, "punc_to_norm", 0, 50, True),
        (3, "norm_to_punc", 0, 0, True),
        (3, "punc_to_norm", 0, 0, True),
        (3, "norm_to_punc", 4, 50, False),
        (3, "punc_to_norm", 4, 50, False),
        (3, "norm_to_punc", 0, 50, False),
        (3, "punc_to_norm", 0, 50, False),
        (3, "norm_to_punc", 0, 0, False),
        (3, "punc_to_norm", 0, 0, False),

        (4, "nojoint", 0, 0, False),
        (4, "norm_to_punc", 4, 50, True),
        (4, "punc_to_norm", 4, 50, True),
        (4, "norm_to_punc", 0, 50, True),
        (4, "punc_to_norm", 0, 50, True),
        (4, "norm_to_punc", 0, 0, True),
        (4, "punc_to_norm", 0, 0, True),
        (4, "norm_to_punc", 4, 50, False),
        (4, "punc_to_norm", 4, 50, False),
        (4, "norm_to_punc", 0, 50, False),
        (4, "punc_to_norm", 0, 50, False),
        (4, "norm_to_punc", 0, 0, False),
        (4, "punc_to_norm", 0, 0, False)
    ]
    
    for fold_id, mode, n_blocks, n_tokens, biaffine in train_data:
        config = f"configs/config.norm.fold_{fold_id}.json"
        train(config, velectra, mode, n_blocks=n_blocks, n_tokens=n_tokens, biaffine=biaffine)