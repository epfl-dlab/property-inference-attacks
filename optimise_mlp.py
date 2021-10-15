import json
import numpy as np

from argparse import ArgumentParser
from os import path, mkdir
from time import strftime
from sklearn.metrics import accuracy_score

from src import logger
from src.generator import GaussianGenerator
from src.model import MLP, LogReg

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

    gen = GaussianGenerator()

    env = [True] * 5 + [False] * 5
    models = [LogReg('label', dict()).fit(gen.sample(b)) for b in env]
    mean_acc = np.mean([accuracy_score(data['label'], models[i].predict(data))
                        for i, data in enumerate([gen.sample(b) for b in env])])

    logger.info('LogReg - {:.2%}'.format(mean_acc))

    for hidden_size in [5, 10, 20]:
        for epochs in [5, 10, 20]:
            for lr in [1e-4, 1e-3, 1e-2, 1e-1]:
                hyperparams = {
                    'input_size': 4,
                    'num_classes': 2,
                    'batch_size': 32,
                    'hidden_size': hidden_size,
                    'epochs': epochs,
                    'learning_rate': lr
                }

                models = [MLP('label', hyperparams).fit(gen.sample(b)) for b in env]
                mean_acc = np.mean([accuracy_score(data['label'], models[i].predict(data))
                                    for i, data in enumerate([gen.sample(b) for b in env])])

                logger.info('hidden={};epochs={};lr={:.4f} - {:.2%}'.format(hidden_size, epochs, lr, mean_acc))

    # Output results


if __name__ == "__main__":
    main()