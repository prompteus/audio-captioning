# Pipeline for MALACH 2023

This repository provides students of the course **Machine Learning and Audio: a challenge** with a simple ML4Audio
pipeline. For demonstration purposes the project uses a small example dataset consisting of 200 wav files in the folder 
![datasets/example_data/audio](datasets/example_data/audio). The main purpose of this project
is to demonstrate how to:
* set up a project using [PyTorch Lightening](https://pytorch-lightning.readthedocs.io/en/stable/) and [Weights & Biases](https://wandb.ai/site)
* load raw audio waveforms and process them to log mel spectrograms
* use a pytorch model to compute predictions for a log mel spectrogram
* apply simple data augmentation techniques to waveforms and to spectrograms

## Getting Started

We recommend to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for maintaining your project environment.

Start by creating a conda environment:
```
conda create -n malach23 python=3.8
```

Activate your environment:
```
conda activate malach23
```

Fork this GitHub Repo to your own GitHub account (select the 'Fork' symbol on menu in the upper right corner on this page).
Then clone your forked repo to your local file system. With your conda environment activated navigate to the root directory of 
your cloned GitHub repo (the ![requirements.txt](requirements.txt) file should be located in this folder) and run:

```
pip install -r requirements.txt
```

Now you should be able to run your first experiment using:

```
python ex_dcase.py
```

If you are not already logged in to a [Weights & Biases](https://wandb.ai/site) account, you will be asked to create a new account or log in to an existing
account. I highly recommend to use Weights & Biases, but of course feel free to use any other tool to log and visualize your experiments!

After several epochs of training and validation you should be able to find the logged metrics on your W&B account. Your plots
should look similar compared to [these](https://wandb.ai/florians/DCASE23/reports/Malach23-Pipeline-Test--VmlldzozNzM0MDA2?accessToken=bo8ps42w5vf42u6yyd23yqb3z0at1ti4lbb4qzfr5ww3kltd36hmbv8r8zqllq1e).

Don't expect any meaningful results, the only thing that proofs your setup to work is the decreasing training loss.

## How to proceed?

* Start with this codebase and modify it according to your needs (the task you have chosen)
* We recommend using [PyCharm](https://www.jetbrains.com/de-de/pycharm/) for programming
* Download the dataset corresponding to your task (pay attention to the recommended train-test splits)
* Try to reproduce the baseline system
