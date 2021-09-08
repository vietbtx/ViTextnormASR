from network.trainer import train


if __name__=="__main__":
    config = "configs/config.norm.json"
    mbert = "configs/config.mbert.json"
    vibert = "configs/config.vibert.json"
    velectra = "configs/config.velectra.json"

    for biaffine in [True, False]:
        train(config, velectra, "nojoint", n_tokens=50, biaffine=biaffine)
        train(config, velectra, "norm_to_punc", n_tokens=50, biaffine=biaffine)
        train(config, velectra, "punc_to_norm", n_tokens=50, biaffine=biaffine)
        train(config, velectra, "nojoint", n_blocks=4, n_tokens=50, biaffine=biaffine)
        train(config, velectra, "norm_to_punc", n_blocks=4, n_tokens=50, biaffine=biaffine)
        train(config, velectra, "punc_to_norm", n_blocks=4, n_tokens=50, biaffine=biaffine)
        train(config, velectra, "nojoint", biaffine=biaffine)
        train(config, velectra, "norm_to_punc", biaffine=biaffine)
        train(config, velectra, "punc_to_norm", biaffine=biaffine)
