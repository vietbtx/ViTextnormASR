from utils.data import Data
from network.trainer import train

if __name__=="__main__":
    config = "configs/config.norm.cpu.json"
    mbert = "configs/config.mbert.json"
    vibert = "configs/config.vibert.json"
    velectra = "configs/config.velectra.json"

    # for fold_id in range(5):
    #     config = f"configs/config.norm.fold_{fold_id}.json"
    #     data = Data.from_config(config, velectra)
    
    train(config, velectra, "nojoint")
    train(config, velectra, "norm_to_punc", n_blocks=4, n_tokens=50)
    train(config, velectra, "punc_to_norm", n_blocks=4, n_tokens=50)
    
    train(config, velectra, "norm_to_punc", n_tokens=50)
    train(config, velectra, "punc_to_norm", n_tokens=50)

    train(config, velectra, "norm_to_punc")
    train(config, velectra, "punc_to_norm")
