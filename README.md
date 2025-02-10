Decision-based Adversarial Attacks via CMA-ES
==============================

Generating Adversarial Examples with Evolutionary Algorithms to investigate robustness of Neural Networks

Project Organization
------------

    ├── LICENSE
    ├── README.md            <- The top-level README for developers using this project.
    ├── data
    │   └── correct indices  <- All indices of the images, that have been correctly classified by the used models in this study.
    │
    ├── requirements.txt     <- The requirements file for reproducing the analysis environment.
    │
    ├── config               <- The yaml files of all conducted experiments.
    │
    └──  src                 <- Source code for use in this project.
         └── AE_gen.py       <- Main file for the DACES attack.
         └── HPO.py          <- Hyperparameter Optimisation with SMAC3
         └── dim_reduction.py<- Different dimensionality reduction techniques, incl. BI, NNI, SA and grid downsizing
         └── evaluate_models.py <- for obtaining the correct indices
         └── l_norm           <- calculates the fitness values
         └── l_norm_no_query  <- for ablation studies
         └── loader.py        <- For loading the different models and datasets


--------
