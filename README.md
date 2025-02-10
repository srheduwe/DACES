Decision-based Adversarial Attacks via CMA-ES
==============================

Generating Adversarial Examples with Evolutionary Algorithms to investigate robustness of Neural Networks

To create an Anaconda environment with the necessary dependencies follow these steps:
function test() {
  console.log("conda create -n DACES python=3.9.19
               conda activate DACES
               pip install git+https://github.com/srheduwe/DACES-with-evotorch.git
               pip install -r requirements.txt");
}
1. conda create -n DACES python=3.9.19
2. conda activate DACES
3. pip install git+https://github.com/srheduwe/DACES-with-evotorch.git
4. pip install -r requirements.txt

To run the attack, just choose the experiment you want to run from the config files:
python src/DACES.py config/ImageNet/BI/resnet50_BI.yaml


Project Organization
------------

    ├── LICENSE
    ├── README.md               <- The top-level README for developers using this project.
    ├── data
    │   └── correct indices     <- All indices of the images, that have been correctly classified by the used models in this study.
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment.
    │
    ├── config                  <- The yaml files of all conducted experiments.
    │
    └──  src                    <- Source code for use in this project.
         └── DACES.py           <- Main file for the DACES attack.
         └── HPO.py             <- Hyperparameter Optimisation with SMAC3
         └── dim_reduction.py   <- Different dimensionality reduction techniques, incl. BI, NNI, SA and grid downsizing
         └── evaluate_models.py <- for obtaining the correct indices
         └── fitness            <- calculates the fitness values
         └── fitness_no_query   <- for ablation studies
         └── loader.py          <- For loading the different models and datasets
