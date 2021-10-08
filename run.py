import json

from argparse import ArgumentParser
from os import path

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
    print('Runconfig:')
    print(runconfig)

    # Run experiments

    gen = GaussianGenerator()
    model = LogReg
    exp = Experiment(gen, 'label', model, runconfig['n_targets'], runconfig['n_shadows'], runconfig['model_params'])
    res_gauss = exp.prepare_and_run_all()
    print('Multivariate Gaussian Experiment:')
    print(res_gauss)

    gen = IndependentPropertyGenerator()
    model = LogReg
    exp = Experiment(gen, 'label', model, runconfig['n_targets'], runconfig['n_shadows'], runconfig['model_params'])
    res_indep = exp.prepare_and_run_all()
    print('Independent Property Experiment:')
    print(res_indep)

    # Output results



if __name__ == "__main__":
    main()