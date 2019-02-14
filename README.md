# Hybrid NN-HMM

This repository contains a Tensorflow implementation of the Hybrid NN-HMM model initially introduced in:

H. Bourlard and N. Morgan, “A continuous speech recognition system embedding MLP into HMM,” in Advances in neural information processing systems, 1990, pp. 186–193.

## Demonstration

A toy example with synthetic data is provided to help play around with the model.
To run it, follow the instructions below.

Install miniconda (https://conda.io/en/latest/miniconda.html).

For example on linux:

```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

From now on, conda installation path will be designated by `$CONDA_PATH`, by default `CONDA_PATH=~/miniconda3/bin`.

```
export CONDA_PATH=~/miniconda3/bin
```

Edit `~/.condarc` and add conda-forge as the prefered package source:

```
channels:
  - conda-forge
  - defaults
```

Setup and activate a new environment with a Python 3.6 interpreter:

```
$CONDA_PATH/conda create -p $PWD/venv python=3.6
source $CONDA_PATH/activate $PWD/venv
```

Install the necessary packages:

```
conda install jupyter notebook matplotlib numpy tensorflow
```

Start jupyter and open the demo notebook.

```
jupyter notebook
```
