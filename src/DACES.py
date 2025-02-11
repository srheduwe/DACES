import torch
import numpy as np
from evotorch.algorithms import CMAES
from evotorch.logging import StdOutLogger, PandasLogger
from evotorch.tools import set_default_logger_config
import logging
import warnings
import os
from fitness import fitness
import sys
from omegaconf import OmegaConf
from loader import loader
from dim_reduction import downsizer
from torch.utils.data import DataLoader
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def ae_gen(config, budget=None, seed_inc=0):
    np.random.seed(seed=config.experiment.seed_images)                          # Seed for images
    torch.manual_seed(config.experiment.seed_images + seed_inc)                 # Seed for experiment
    torch.cuda.manual_seed(config.experiment.seed_images + seed_inc)            # Seed for experiment

    set_default_logger_config(logger_level=logging.WARNING, override=True)
    
    net, dataset, dim = loader(model=config.experiment.model, split=config.experiment.split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    popsize = config.parameters.popsize
    query_budget = budget if budget else config.experiment.query_budget         # Use budget of HPO or budget of experiment

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        queries, norms = [], []

        indices = np.load(config.experiment.indices)                            # Indices of all correctly classified instances in testset
        sampled_indices = np.random.choice(a=indices, size=config.experiment.samples, replace=False)

        for i in sampled_indices:
            image, label = dataloader.dataset[i][0].to(device), dataloader.dataset[i][1]
            start = time.time()

            if label != net(image.unsqueeze(0)).argmax(1):
                print("Already Misclassified!")
                break

            _, height, width, upsizer = downsizer(config=config, image=image, dim=dim, popsize=popsize)

            problem = fitness(
                d=height*width*3,                               # Dimensionality of the problem                
                label=label,                                    # Label of the original image
                I=image,                                        # Original image
                dim=dim,                                        # Length/Width of original image (for ImageNet=224)  
                height=height,                                  # Height of lower image subspace
                width=width,                                    # Width of lower image subspace
                upsizer=upsizer,                                # upsizer for upscaling image subspace
                net=net,                                        # Neural Network for inference
                exp_update=None,                                # Whether e_sigma is used, defaults to None
                e_sigma=config.parameters.e_sigma,              # Step size enlarger of dynamic configuration
                device=device,
                start_time=start,
                query_budget=query_budget,                      # Query budget, most often 30.000
                abl_c=config.experiment.abl_c,                  # Whether ablation should be done on the dynamic configuration strategy
                norm=config.experiment.norm                     # Norm to be used, most often 2 (euclidian norm)
            )

            center_init = None if config.experiment.abl_c == True else torch.zeros(size=(height*width*3,))  # m^0 = 0 if not ablation is done

            searcher_args = {
                "problem": problem,
                "stdev_init": config.parameters.stdev_init if not config.experiment.abl_p else 0.1,
                "stdev_max": config.parameters.stdev_max if not config.experiment.abl_p else 5,
                "center_init": center_init,
                "separable": True, 
                "popsize": popsize,
            }

            if not config.experiment.abl_p:
                searcher_args.update({
                    "c_sigma": config.parameters.c_sigma,
                    "c_sigma_ratio": config.parameters.c_sigma_ratio,
                    "c_m": config.parameters.c_m,
                    "damp_sigma": config.parameters.damp_sigma,
                    "damp_sigma_ratio": config.parameters.damp_sigma_ratio,
                    "c_c": config.parameters.c_c,
                    "c_c_ratio": config.parameters.c_c_ratio,
                    "c_1": torch.tensor(config.parameters.c_1),
                    "c_1_ratio": config.parameters.c_1_ratio,
                    "c_mu": torch.tensor(config.parameters.c_mu),
                    "c_mu_ratio": config.parameters.c_mu_ratio
                })

            searcher = CMAES(**searcher_args)

            logger = StdOutLogger(searcher, interval=config.experiment.logging)
            pandas_logger = PandasLogger(searcher)

            searcher.run(10000) # Arbitrary high number, CMA-ES script has been change so that it stops based on query budget, not generations

            pop_best = searcher.status["pop_best_eval"]
            queries.append(searcher.problem._query_counter.cpu().numpy())
            norms.append(pandas_logger.to_dataframe()["pop_best_eval"].values)

            end = time.time()
            print(f"Instance {i} took {round(end - start, 3)} seconds")
            print("Final l2 norm: ", round(pop_best, 3))
            popsize = searcher.popsize

            if config.experiment.saving:
                folder = sys.argv[1][7:-5]
                path = f'data/results/{folder}_{seed_inc}/'
                os.makedirs(path, exist_ok=True)

                perturbation, Evals = searcher.population.access_values(keep_evals=True), searcher.population.access_evals()
                pandas_logger.to_dataframe().to_csv(path + f"{i}_df.csv")
                torch.save(searcher.problem._best_Instance, path + f"{i}_adv.pt")
                np.save(path + f"{i}_queries.npy", searcher.problem._query_counter.cpu().numpy())
                np.save(path + f"{i}_norms.npy", searcher.problem._best_norm.cpu().numpy())
                np.save(path + f"{i}_times.npy", searcher.problem._time.cpu().numpy())

                if upsizer: 
                    perturbation = perturbation.clone().reshape(popsize, 3, height, width).float()
                    perturbed_image = image.unsqueeze(0).repeat(popsize, 1, 1, 1).float()

                    if isinstance(upsizer, list):
                        upsizer[0] = np.array([[[[i]]] for i in range(popsize)])
                        perturbed_image[upsizer] += perturbation
                    else:
                        perturbation = upsizer(perturbation)
                        perturbed_image += perturbation

                    perturbed_image = torch.clamp(perturbed_image, min=0.0, max=1.0)
                    preds = net(perturbed_image.reshape(popsize, 3, dim, dim)).argmax(1) 

                else:
                    perturbation = perturbation.reshape(popsize, 3, height, width).float()
                    perturbation = torch.clamp(perturbation, min=0.0, max=1.0)
                    preds = net(perturbed_image.reshape(popsize, 3, dim, dim)).argmax(1) 

                print(f"Label changed from {label} to {preds[torch.argmin(Evals)]}")

    if __name__ != '__main__':
        # Calculate the AUC when running DACES as part of the HPO.py
        area_instances_stepwise_list = [query * norm for query, norm in zip(queries, norms)]
        area_instances = [area_instances_stepwise.sum() for area_instances_stepwise in area_instances_stepwise_list]
        average_area = np.mean(area_instances)
        return average_area

if __name__ == '__main__':
    experiment = str(sys.argv[1])
    config = OmegaConf.load(experiment)
    print(sys.argv)
    print(config.experiment)
    print(config.parameters)

    for i in range(config.experiment.number_seeds):
        start = time.time()
        ae_gen(config, seed_inc=i)
        end = time.time()
        print(end - start)