# property-inference-framework
Modular framework for property inference attacks on deep neural networks

## Attacks

 * __White-Box__: Threat model in which we have access to the model's parameters directly
 * __Grey-Box__: Threat model in which we have only query access to the model, but we know the model's architecture and hyperparameters
 * __Black-Box__: Threat model in which we do only have query access to the model, and no other information
 
## Environment

The environment with required dependencies is provided for Anaconda in `env.yml`
To install it, simply use:

`conda env create -f env.yml`

Additionally, PyTorch is required to run this framework. Please find all useful installation information [here](https://pytorch.org/).

## Unit tests

The framework is provided with a few, simple unit tests. Run them with:

`python -m unittest discover`

## Running an experiment

To run a simple experiment, please simply use the provided `run.py`. You can change any experiment parameter with the help of the yaml config files, inside the `config` folder.
 
## Framework usage

This framework is made modular for any of your experiments: you simply should define subclasses of `Generator` and `Model`
to represent your data source and your evaluated model respectively.

From these, you can create and run an instance of `Experiment` to run different kinds of Property Inference Attacks.