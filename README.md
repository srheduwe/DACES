DACES: Decision-based Adversarial Attacks via CMA-ES
==============================

Welcome to DACES, an algorithm for generating Adversarial Examples with CMA-ES to investigate robustness of Neural Networks. 

Getting started:
To create an Anaconda environment with the necessary dependencies follow these steps:
```
conda create -n DACES python=3.9.19
conda activate DACES
pip install -r requirements.txt
```
To run the attack, choose the experiment you want to run from the config files, e.g.:
```
python src/DACES.py config/CIFAR100/resnet34.yaml
```

For running experiments on ImageNet, follow the instructions for obtaining the dataset from the official [website](https://www.image-net.org). Then, place the dataset into data/raw.

If you want to rerun the HPO, you need to install [SMAC](https://automl.github.io/SMAC3/main/1_installation.html). Then, choose any version for which you want to do the HPO, e.g.:
```
python src/HPO.py config/ImageNet/BI/BI_HPO.yaml
```

We use the [EvoTorch](https://evotorch.ai) library as a basis for our work. Our adapted version can be found [here](https://github.com/srheduwe/DACES-with-evotorch.git).


Project Organization
------------

    ├── LICENSE
    ├── README.md                   <- The top-level README for developers using this project.
    ├── data
    │   └── correct indices         <- All indices of the images, that have been correctly classified by the used models in this study.
    │   └── results                 <- Results will be stored here.
    │   └── raw                     <- Raw data will be stored here.
    │
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment.
    │
    ├──  src                        <- Source code for use in this project.
    │    └── DACES.py               <- Main file for the DACES attack.
    │    └── HPO.py                 <- Hyperparameter Optimisation with SMAC3
    │    └── dim_reduction.py       <- Different dimensionality reduction techniques, incl. BI, NNI, SA and │grid downsizing
    │    └── evaluate_models.py     <- for obtaining the correct indices
    │    └── fitness                <- calculates the fitness values
    │    └── loader.py              <- For loading the different models and datasets
    │
    └── config                      <- The yaml files of all conducted experiments.
        └── ImageNet
        │   └── BI
        │   └── Grid
        │   └── NNI
        │   └── SA
        └── CIFAR100
