from network.trainer import train


if __name__=="__main__":
    config = "configs/config.norm.json"
    mbert = "configs/config.mbert.json"
    vibert = "configs/config.vibert.json"
    velectra = "configs/config.velectra.json"
    
    train(config, velectra, "nojoint")
    train(config, velectra, "norm_to_punc")
    train(config, velectra, "punc_to_norm")

    train(config, velectra, "nojoint", n_tokens=50)
    train(config, velectra, "norm_to_punc", n_tokens=50)
    train(config, velectra, "punc_to_norm", n_tokens=50)

    train(config, velectra, "nojoint", n_blocks=10, n_tokens=50)
    train(config, velectra, "norm_to_punc", n_blocks=10, n_tokens=50)
    train(config, velectra, "punc_to_norm", n_blocks=10, n_tokens=50)
