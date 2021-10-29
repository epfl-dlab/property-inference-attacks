# property-inference-framework

In this repository, we propose a modular framework to run Property Inference Attacks on Machine Learning models.

[![Continuous Integration](https://github.com/epfl-dlab/property-inference-framework/actions/workflows/python-app.yml/badge.svg)](https://github.com/epfl-dlab/property-inference-framework/actions/workflows/python-app.yml)

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

 
## Environment

The environment with required dependencies is provided for Anaconda in `env.yml`
To install it, simply use:

`conda env create -f env.yml`

Additionally, PyTorch is required to run this framework. Please find all useful installation information [here](https://pytorch.org/).

## Unit tests

The framework is provided with a few, simple unit tests. Run them with:

`python -m unittest discover`

to check the correctness of your installation.

## Running an experiment

To run a simple experiment, please simply use the provided `run.py`. You can change any experiment parameter with the help of the yaml config files, inside the `config` folder.

To run an experiment using a specific `my_experiments.yaml` config file, you should place its yaml config file in `/config/experiments`, and then run:

`python run.py experiments=my_experiments`

It is possible to provide a list as a model hyperparameter: in that case, the framework will automatically optimise between the given options.
 
## Framework usage

This framework is made modular for any of your experiments: you simply should define subclasses of `Generator` and `Model`
to represent your data source and your evaluated model respectively.

From these, you can create a specific experiment configuration file, or extend the Experiment class.

## References

[1] Karan Ganju, Qi Wang, Wei Yang, Carl A. Gunter, and Nikita Borisov. 2018. Property Inference Attacks on Fully Connected Neural Networks using Permutation Invariant Representations. In Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security (CCS '18). Association for Computing Machinery, New York, NY, USA, 619–633. DOI:https://doi.org/10.1145/3243734.3243834

[2] Anshuman Suri, David Evans. 2021. Formalizing Distribution Inference Risks. 2021 Workshop on Theory and Practice of Differential Privacy, ICML. https://arxiv.org/abs/2106.03699

[3] Wanrong Zhang, Shruti Tople, Olga Ohrimenko. 2021. Leakage of Dataset Properties in Multi-Party Machine Learning. https://arxiv.org/abs/2006.07267
