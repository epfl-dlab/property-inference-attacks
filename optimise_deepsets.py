import json

from os import path, mkdir

from omegaconf import DictConfig, OmegaConf
import hydra

from src import TIMESTAMP, logger
from src.experiment import Experiment
from src.generator import GaussianGenerator, IndependentPropertyGenerator
from src.model import LogReg, MLP

CWD = path.dirname(__file__)

MODELS = {
    'LogReg': LogReg,
    'MLP': MLP
}

GENERATORS = {
    'GaussianGenerator': GaussianGenerator,
    'IndependentPropertyGenerator': IndependentPropertyGenerator
}


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    experiments = dict()
    gen = GaussianGenerator(1024)
    model = MLP
    exp = Experiment(gen, 'label', model, 256, 2048, cfg.models['MLP'])
    logger.info('Training targets...')
    exp.prepare_attacks()
    logger.info('Training shadows...')
    exp.run_shadows(MLP, cfg.models['MLP'])

    for latent_dim in [5, 10 ,20]:
        for epochs in [20, 50, 100, 250]:
            for lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
                for wd in [0, 1e-4, 1e-3, 1e-2]:
                    name = 'latent={} - epochs={} - lr={:.4f} - wd={:.4f}'.format(latent_dim, epochs, lr, wd)
                    hyperparams = {
                        'latent_dim': latent_dim,
                        'epochs': epochs,
                        'learning_rate': lr,
                        'weight_decay': wd
                    }
                    logger.info('Running {}...'.format(name))
                    experiments[name] = exp.run_whitebox_deepsets(hyperparams)
                    logger.info('Results {}: {:.2%}'.format(name, experiments[name]))


    # Output results
    outfile_name = 'results_PIA_' + TIMESTAMP + '.json'
    outdir = path.join(CWD, cfg.outdir)
    if not path.isdir(outdir):
        mkdir(outdir)
    with open(path.join(outdir, outfile_name), 'w') as f:
        json.dump(experiments, f)


if __name__ == "__main__":
    main()
