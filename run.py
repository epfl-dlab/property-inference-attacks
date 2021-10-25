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
    for gen in cfg.experiments.generators:
        generator = GENERATORS[gen](num_samples=cfg.generators.num_samples)
        for model in cfg.experiments.models:
            experiment = Experiment(generator, cfg.generators.label_col, MODELS[model], cfg.experiments.n_targets, cfg.experiments.n_shadows,
                                    cfg.models[model], cfg.experiments.n_queries)

            logger.info('Training target models: {} - {}'.format(gen, model))
            experiment.prepare_attacks()
            logger.info('Training shadow models: {} - {}'.format(gen, model))
            experiment.run_shadows(MODELS[model], cfg.models[model])

            runs = list(cfg.experiments.runs)
            if 'BlackBox' in runs:
                runs.remove('BlackBox')
                runs.append('BlackBox')

            for run in cfg.experiments.runs:
                name = '{} - {} - {}'.format(gen, model, run)
                logger.info('Running {}...'.format(name))
                if run == 'Naive':
                    experiments[name] = experiment.run_whitebox_sort(sort=False)
                elif run == 'Sort':
                    experiments[name] = experiment.run_whitebox_sort(sort=True)
                elif run == 'DeepSets':
                    experiments[name] = experiment.run_whitebox_deepsets(cfg.deepsets)
                elif run == 'GreyBox':
                    experiments[name] = experiment.run_blackbox()
                elif run == 'BlackBox':
                    logger.info('Training default shadow models: {} - {}'.format(gen, cfg.experiments.blackbox_model))
                    experiment.run_shadows(MODELS[cfg.experiments.blackbox_model], cfg.models[cfg.experiments.blackbox_model])
                    experiments[name] = experiment.run_blackbox()
                else:
                    raise AttributeError('Invalid run provided: should be Naive, Sort, DeepSets, GreyBox or BlackBox'
                                         ' - instead is {}'.format(run))

                logger.info('Result for {}: {:.2%}'.format(name, experiments[name]))

    # Output results
    outfile_name = 'results_PIA_' + TIMESTAMP + '.json'
    outdir = path.join(CWD, cfg.outdir)
    if not path.isdir(outdir):
        mkdir(outdir)
    with open(path.join(outdir, outfile_name), 'w') as f:
        json.dump(experiments, f)


if __name__ == "__main__":
    main()
