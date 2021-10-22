import json

from argparse import ArgumentParser
from os import path, mkdir
from time import strftime

from src import logger
from src.experiment import Experiment
from src.generator import GaussianGenerator, IndependentPropertyGenerator
from src.model import LogReg, MLP

CWD = path.dirname(__file__)

def main():
    argparser = ArgumentParser()
    argparser.add_argument('--runconfig', '-RC', default='config.json', type=str,
                           help='Path relative to cwd of runconfig file')
    argparser.add_argument('--outdir', '-O', default='results', type=str,
                           help='Path relative to cwd for storing output files')

    args = argparser.parse_args()

    # Load runconfig
    with open(path.join(CWD, args.runconfig)) as f:
        runconfig = json.load(f)
    logger.debug('Runconfig: {}'.format(runconfig))

    # Define experiments
    experiments = dict()

    if 'sort_model_parameters' in runconfig.keys():
        sort_params = runconfig['sort_model_parameters']
    else:
        sort_params = False

    gen = GaussianGenerator()
    logreg = LogReg
    mlp = MLP

    """
    experiments['LogReg Multivariate Gaussian'] = Experiment(gen, 'label', logreg, runconfig['n_targets'],
                                                      runconfig['n_shadows'], runconfig['model_params'],
                                                      sort_params=False)


    experiments['MLP Multivariate Gaussian w/ Sort'] = Experiment(gen, 'label', mlp, runconfig['n_targets'],
                                                      runconfig['n_shadows'], runconfig['model_params'],
                                                      sort_params=True)
    """
    experiments['MLP Multivariate Gaussian w/o Sort'] = Experiment(gen, 'label', mlp, runconfig['n_targets'],
                                                      runconfig['n_shadows'], runconfig['model_params'],
                                                      sort_params=False)

    gen = IndependentPropertyGenerator()
    """
    experiments['LogReg Independent Property'] = Experiment(gen, 'label', logreg, runconfig['n_targets'],
                                                     runconfig['n_shadows'], runconfig['model_params'],
                                                     sort_params=False)
    

    experiments['MLP Independent Property'] = Experiment(gen, 'label', mlp, runconfig['n_targets'],
                                                     runconfig['n_shadows'], runconfig['model_params'],
                                                     sort_params=True)
                                                     
    """

    # Run experiments
    results = dict()
    for k, v in experiments.items():
        logger.info('Running {} Experiment'.format(k))
        results[k] = v.prepare_and_run_all(deepsets=True)
        logger.info('Results of {} Experiment: {}'.format(k, results[k]))

    # Output results
    outfile_name = 'results_PIA_' + strftime('%d%m%y_%H:%M:%S') + '.json'
    outdir = path.join(CWD, args.outdir)
    if not path.isdir(outdir):
        mkdir(outdir)
    with open(path.join(outdir, outfile_name), 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()