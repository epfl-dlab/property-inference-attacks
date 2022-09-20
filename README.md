# Property Inference Attacks

In this repository, we propose a modular framework to run Property Inference Attacks on Machine Learning models.

[![Continuous Integration](https://github.com/epfl-dlab/property-inference-framework/actions/workflows/python-app.yml/badge.svg)](https://github.com/epfl-dlab/property-inference-framework/actions/workflows/python-app.yml)
[![PyPI](https://img.shields.io/pypi/v/propinfer)](https://pypi.org/project/propinfer/)
[![Documentation](https://img.shields.io/badge/Documentation-v1.3.0-informational)](https://epfl-dlab.github.io/property-inference-attacks/)


## Installation

You can get this package directly from pip:

`python -m pip install propinfer`

Please note that PyTorch is required to run this framework. Please find installation instructions corresponding to you [here](https://pytorch.org/).

## Usage

This framework is made modular for any of your experiments: you simply should define subclasses of `Generator` and `Model`
to represent your data source and your evaluated model respectively.

From these, you can create a specific experiment configuration file. We suggest using [hydra](https://hydra.cc/docs/intro/) for your configurations, but parameters can also be passed in a standard `dict`.

Alternatively, you can extend the Experiment class.

## Threat models and attacks

### White-Box 
In this threat model, we have access to the model's parameters directly. In this case, [1] defines three different attacks:
 * Simple meta-classifier attack
 * Simple meta-classifier attack, with layer weights' sorting
 * DeepSets attack
 
They are respectively designated by the keywords `Naive`, `Sort`and `DeepSets`.

### Grey- and Black-Box
 
In this threat model, we have only query access to the model (we do not know its parameters). In the scope of the Grey-Box threat model, we know the model's architecture and hyperparameters - in the scope of Black-Box we do not.

For the Grey-Box case, [2] describes two simple attacks:
 * The Loss Test (represented by the `LossTest` keyword)
 * The Threshold Test (represented by the `ThresholdTest` keyword)
 
[3] also proposes a meta-classifier-based attack, that we use for both the Grey-Box and Black-Box cases: these are respectively represented by the `GreyBox` and `BlackBox` keywords. For the latter case, we simply default on a pre-defined model architecture.

## Unit tests

The framework is provided with a few, simple unit tests. Run them with:

`python -m unittest discover`

to check the correctness of your installation.

## Running an experiment

To run a simple experiment, please simply use the provided `run.py`. You can change any experiment parameter with the help of the yaml config files, inside the `config` folder.

To run an experiment using a specific `my_experiments.yaml` config file, you should place its yaml config file in `/config/experiments`, and then run:

`python run.py experiments=my_experiments`

Alternatively, you can instanciate an `Experiment` object using a specific `Generator` and `Model`, and then run both targets and shadows before performing an attack.

It is possible to provide a list as a model hyperparameter: in that case, the framework will automatically optimise between the given options.

## Citation

The research paper relative to this library is currently only available as a pre-print on [arXiv](https://arxiv.org/abs/2209.08541). If you use our library for your research, please cite our work:

```
V. Hartmann, L. Meynent, M. Peyrard, D. Dimitriadis, S. Tople, and R. West, Distribution inference risks: Identifying and mitigating sources of leakage. arXiv, 2022. doi: 10.48550/ARXIV.2209.08541. 
```

```
@misc{https://doi.org/10.48550/arxiv.2209.08541,
	title        = {Distribution inference risks: Identifying and mitigating sources of leakage},
	author       = {Hartmann, Valentin and Meynent, Léo and Peyrard, Maxime and Dimitriadis, Dimitrios and Tople, Shruti and West, Robert},
	year         = 2022,
	publisher    = {arXiv},
	doi          = {10.48550/ARXIV.2209.08541},
	url          = {https://arxiv.org/abs/2209.08541},
	copyright    = {arXiv.org perpetual, non-exclusive license},
	keywords     = {Cryptography and Security (cs.CR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences}
}
```

## References

[1] Karan Ganju, Qi Wang, Wei Yang, Carl A. Gunter, and Nikita Borisov. 2018. Property Inference Attacks on Fully Connected Neural Networks using Permutation Invariant Representations. In Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security (CCS '18). Association for Computing Machinery, New York, NY, USA, 619–633. DOI:https://doi.org/10.1145/3243734.3243834

[2] Anshuman Suri, David Evans. 2021. Formalizing Distribution Inference Risks. 2021 Workshop on Theory and Practice of Differential Privacy, ICML. https://arxiv.org/abs/2106.03699

[3] Wanrong Zhang, Shruti Tople, Olga Ohrimenko. 2021. Leakage of Dataset Properties in Multi-Party Machine Learning. https://arxiv.org/abs/2006.07267
