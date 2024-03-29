import json
import logging.config

from os import path

from omegaconf import DictConfig
import hydra

from propinfer import Experiment
from propinfer import GaussianGenerator, IndependentPropertyGenerator, ProbitGenerator, NonlinearGenerator
from propinfer import LogReg, MLP

CWD = path.dirname(__file__)

MODELS = {
    'LogReg': LogReg,
    'MLP': MLP
}

GENERATORS = {
    'GaussianGenerator': GaussianGenerator,
    'IndependentPropertyGenerator': IndependentPropertyGenerator,
    'ProbitGenerator': ProbitGenerator,
    'NonlinearGenerator': NonlinearGenerator
}

from os import path, mkdir
from time import strftime

TIMESTAMP = strftime('%d%m%y_%H:%M:%S')

config = path.abspath(path.join(path.dirname(__file__), 'logging.ini'))

logdir = path.abspath(path.join(path.dirname(__file__),"./logs"))
if not path.isdir(logdir):
    mkdir(logdir)
logfile = logdir + '/logs_property-inference-attacks_' + TIMESTAMP + '.txt'

logging.config.fileConfig(config, defaults={'logfilename': logfile})

# create logger
logger = logging.getLogger('propinfer')


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    experiments = dict()
    for gen in cfg.experiments.generators:
        generator = GENERATORS[gen](num_samples=cfg.generators.n_samples)
        for model in cfg.experiments.models:
            n_classes = cfg.experiments.n_classes if 'n_classes' in cfg.experiments.keys() else 2
            exp_range = None if n_classes > 1 else cfg.experiments.range
            experiment = Experiment(generator, cfg.generators.label_col, MODELS[model], cfg.experiments.n_targets, cfg.experiments.n_shadows,
                                    cfg.models[model], cfg.experiments.n_queries, n_classes=n_classes, range=exp_range)

            logger.info('Training target models: {} - {}'.format(gen, model))
            experiment.run_targets()
            logger.info('Training shadow models: {} - {}'.format(gen, model))
            experiment.run_shadows(MODELS[model], cfg.models[model])

            runs = list(cfg.experiments.runs)
            if 'BlackBox' in runs:
                runs.remove('BlackBox')
                runs.append('BlackBox')

            for run in cfg.experiments.runs:
                name = '{} - {} - {}'.format(gen, model, run)
                logger.info('Running {}...'.format(name))
                if run == 'LossTest':
                    experiments[name] = experiment.run_loss_test()
                elif run == 'ThresholdTest':
                    experiments[name] = experiment.run_threshold_test()
                elif run == 'Naive':
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

                if n_classes > 1:
                    logger.info('Attack accuracy for {}: {:.2%}'.format(name, experiments[name]))
                else:
                    logger.info('Mean absolute error for {}: {:.2f}'.format(name, experiments[name]))

    # Output results
    outfile_name = 'results_PIA_' + TIMESTAMP + '.json'
    outdir = path.join(CWD, cfg.outdir)
    if not path.isdir(outdir):
        mkdir(outdir)
    with open(path.join(outdir, outfile_name), 'w') as f:
        json.dump(experiments, f)


if __name__ == "__main__":
    main()
